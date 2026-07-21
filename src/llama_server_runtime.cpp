#include "chromofold/llama_server_runtime.h"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

struct cf_llama_server_runtime {
    cf_llama_server_config config{};
    cf_multisequence_cache * host_cache = nullptr;
    cf_ms_cuda_resolver * cuda_resolver = nullptr;
    mutable std::mutex mutex;
    std::unordered_set<int64_t> live_sequences;
    cf_llama_server_stats stats{};
    bool shutting_down = false;
    std::string error;
};

namespace {

bool valid_config(const cf_llama_server_config & c) {
    return c.layer_count && c.query_head_count && c.kv_head_count && c.head_dim &&
           c.page_size && c.block_size && c.gqa_group_size &&
           c.query_head_count == c.kv_head_count * c.gqa_group_size &&
           c.llama_commit && c.model_sha256;
}

int fail(cf_llama_server_runtime * runtime, const char * message) {
    if (runtime) runtime->error = message ? message : "unknown ChromoFold server error";
    return -1;
}

bool sequence_is_live(cf_llama_server_runtime * runtime, int64_t sequence_id) {
    std::lock_guard<std::mutex> guard(runtime->mutex);
    return !runtime->shutting_down && runtime->live_sequences.count(sequence_id) != 0;
}

} // namespace

extern "C" cf_llama_server_runtime * cf_llama_server_runtime_create(const cf_llama_server_config * config) {
    if (!config || !valid_config(*config)) return nullptr;
    try {
        auto runtime = std::make_unique<cf_llama_server_runtime>();
        runtime->config = *config;
        cf_multisequence_config host{};
        host.layer_count = config->layer_count;
        host.kv_head_count = config->kv_head_count;
        host.query_head_count = config->query_head_count;
        host.head_dim = config->head_dim;
        host.page_size = config->page_size;
        host.gqa_group_size = config->gqa_group_size;
        host.use_async_allocator = 1;
        host.enable_invariant_auditor = 1;
        runtime->host_cache = cf_multisequence_create(&host);
        if (!runtime->host_cache) throw std::runtime_error("failed to create M10 host cache");
        cf_ms_cuda_config device{};
        device.layer_count = config->layer_count;
        device.kv_head_count = config->kv_head_count;
        device.query_head_count = config->query_head_count;
        device.head_dim = config->head_dim;
        device.page_size = config->page_size;
        device.gqa_group_size = config->gqa_group_size;
        device.use_async_allocator = 1;
        runtime->cuda_resolver = cf_ms_cuda_create(&device);
        if (!runtime->cuda_resolver) throw std::runtime_error("failed to create M10 CUDA resolver");
        return runtime.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void cf_llama_server_runtime_destroy(cf_llama_server_runtime * runtime, void * stream) {
    if (!runtime) return;
    cf_llama_server_shutdown_async(runtime, stream);
    if (runtime->cuda_resolver) cf_ms_cuda_destroy(runtime->cuda_resolver, stream);
    if (runtime->host_cache) cf_multisequence_destroy(runtime->host_cache, stream);
    delete runtime;
}

extern "C" int cf_llama_server_slot_create(cf_llama_server_runtime * runtime, int64_t sequence_id) {
    if (!runtime) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (runtime->shutting_down) return fail(runtime, "server runtime is shutting down");
    if (runtime->live_sequences.count(sequence_id)) return fail(runtime, "sequence already exists");
    if (cf_sequence_create(runtime->host_cache, sequence_id) != 0) return fail(runtime, cf_multisequence_last_error(runtime->host_cache));
    if (cf_ms_cuda_sequence_create(runtime->cuda_resolver, sequence_id) != 0) {
        cf_sequence_remove_async(runtime->host_cache, sequence_id, -1, -1, nullptr);
        return fail(runtime, cf_ms_cuda_last_error(runtime->cuda_resolver));
    }
    runtime->live_sequences.insert(sequence_id);
    runtime->stats.http_requests++;
    runtime->error.clear();
    return 0;
}

extern "C" int cf_llama_server_slot_reset_async(cf_llama_server_runtime * runtime, int64_t sequence_id, void * stream) {
    if (!runtime) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (!runtime->live_sequences.count(sequence_id)) return fail(runtime, "sequence does not exist");
    if (cf_ms_cuda_sequence_remove_async(runtime->cuda_resolver, sequence_id, stream) != 0) return fail(runtime, cf_ms_cuda_last_error(runtime->cuda_resolver));
    if (cf_sequence_remove_async(runtime->host_cache, sequence_id, -1, -1, stream) != 0) return fail(runtime, cf_multisequence_last_error(runtime->host_cache));
    runtime->live_sequences.erase(sequence_id);
    runtime->stats.slot_reuses++;
    runtime->error.clear();
    return 0;
}

extern "C" int cf_llama_server_cancel_async(cf_llama_server_runtime * runtime, int64_t sequence_id, void * stream) {
    if (!runtime) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (!runtime->live_sequences.count(sequence_id)) return fail(runtime, "sequence does not exist");
    if (cf_ms_cuda_cancel_sequence_async(runtime->cuda_resolver, sequence_id, stream) != 0) return fail(runtime, cf_ms_cuda_last_error(runtime->cuda_resolver));
    runtime->stats.cancelled_requests++;
    runtime->error.clear();
    return 0;
}

extern "C" int cf_llama_server_sequence_copy_async(cf_llama_server_runtime * runtime, int64_t source, int64_t destination,
                                                       uint32_t token_begin, uint32_t token_end, void * stream) {
    if (!runtime) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (runtime->shutting_down || !runtime->live_sequences.count(source)) return fail(runtime, "source sequence does not exist");
    if (cf_sequence_copy_async(runtime->host_cache, source, destination, token_begin, token_end, stream) != 0)
        return fail(runtime, cf_multisequence_last_error(runtime->host_cache));
    if (cf_ms_cuda_copy_sequence_async(runtime->cuda_resolver, source, destination, stream) != 0)
        return fail(runtime, cf_ms_cuda_last_error(runtime->cuda_resolver));
    runtime->live_sequences.insert(destination);
    runtime->error.clear();
    return 0;
}

extern "C" int cf_llama_server_sequence_remove_async(cf_llama_server_runtime * runtime, int64_t sequence_id,
                                                         int64_t position_begin, int64_t position_end, void * stream) {
    if (!runtime) return -1;
    if (position_begin < 0 && position_end < 0) return cf_llama_server_slot_reset_async(runtime, sequence_id, stream);
    if (!sequence_is_live(runtime, sequence_id)) return fail(runtime, "sequence does not exist");
    if (cf_sequence_remove_async(runtime->host_cache, sequence_id, position_begin, position_end, stream) != 0)
        return fail(runtime, cf_multisequence_last_error(runtime->host_cache));
    runtime->error.clear();
    return 0;
}

extern "C" int cf_llama_server_sequence_shift_async(cf_llama_server_runtime * runtime, int64_t sequence_id,
                                                        uint32_t position_begin, uint32_t position_end, int32_t delta, void * stream) {
    if (!runtime || !sequence_is_live(runtime, sequence_id)) return -1;
    if (cf_sequence_shift_async(runtime->host_cache, sequence_id, position_begin, position_end, delta, stream) != 0)
        return fail(runtime, cf_multisequence_last_error(runtime->host_cache));
    if (cf_ms_cuda_shift_sequence(runtime->cuda_resolver, sequence_id, position_begin, position_end, delta) != 0)
        return fail(runtime, cf_ms_cuda_last_error(runtime->cuda_resolver));
    std::lock_guard<std::mutex> guard(runtime->mutex);
    runtime->stats.context_shifts++;
    runtime->error.clear();
    return 0;
}

extern "C" int cf_llama_server_append_layer_async(cf_llama_server_runtime * runtime, int64_t sequence_id,
                                                      uint32_t layer_index, uint32_t token_begin,
                                                      const float * key, const float * value,
                                                      uint32_t token_count, void * stream) {
    if (!runtime || !key || !value || !token_count || !sequence_is_live(runtime, sequence_id)) return -1;
    if (cf_ms_cuda_stage_active_async(runtime->cuda_resolver, sequence_id, layer_index, token_begin,
                                      key, value, token_count, stream) != 0)
        return fail(runtime, cf_ms_cuda_last_error(runtime->cuda_resolver));
    runtime->error.clear();
    return 0;
}

extern "C" int cf_llama_server_execute_batch_async(cf_llama_server_runtime * runtime,
                                                       const cf_llama_server_batch * batch,
                                                       void * stream) {
    if (!runtime || !batch || !batch->attention_items || !batch->attention_item_count) return -1;
    std::vector<cf_ms_batch_item> items;
    items.reserve(batch->attention_item_count);
    std::unordered_set<int64_t> sequence_ids;
    {
        std::lock_guard<std::mutex> guard(runtime->mutex);
        if (runtime->shutting_down) return fail(runtime, "server runtime is shutting down");
        if (runtime->stats.dense_nodes_scheduled != 0)
            return fail(runtime, "dense attention node scheduled while ChromoFold backend is active");
        for (uint32_t i = 0; i < batch->attention_item_count; ++i) {
            const auto & source = batch->attention_items[i];
            if (!runtime->live_sequences.count(source.sequence_id)) return fail(runtime, "batch references unknown sequence");
            sequence_ids.insert(source.sequence_id);
            cf_ms_batch_item item{};
            item.sequence_id = source.sequence_id;
            item.layer = source.layer;
            item.query = source.query;
            item.output = source.output;
            item.output_type = source.output_type;
            item.output_token_stride_bytes = source.output_token_stride_bytes;
            item.output_head_stride_bytes = source.output_head_stride_bytes;
            item.output_element_stride_bytes = source.output_element_stride_bytes;
            item.query_token_begin = source.query_token_begin;
            item.softmax_scale = source.softmax_scale;
            item.causal_window = source.causal_window;
            items.push_back(item);
        }
        if (sequence_ids.size() > 1) runtime->stats.interleaved_batches++;
    }
    if (cf_ms_cuda_execute_batch_async(runtime->cuda_resolver, items.data(), static_cast<uint32_t>(items.size()), stream) != 0)
        return fail(runtime, cf_ms_cuda_last_error(runtime->cuda_resolver));
    runtime->error.clear();
    return 0;
}

extern "C" int cf_llama_server_shutdown_async(cf_llama_server_runtime * runtime, void * stream) {
    if (!runtime) return -1;
    std::vector<int64_t> sequences;
    {
        std::lock_guard<std::mutex> guard(runtime->mutex);
        if (runtime->shutting_down) return runtime->stats.shutdown_failures ? -1 : 0;
        runtime->shutting_down = true;
        sequences.assign(runtime->live_sequences.begin(), runtime->live_sequences.end());
        runtime->live_sequences.clear();
    }
    for (int64_t sequence_id : sequences) {
        cf_ms_cuda_cancel_sequence_async(runtime->cuda_resolver, sequence_id, stream);
        cf_ms_cuda_sequence_remove_async(runtime->cuda_resolver, sequence_id, stream);
        cf_sequence_remove_async(runtime->host_cache, sequence_id, -1, -1, stream);
    }
    cf_ms_cuda_reclaim_async(runtime->cuda_resolver, stream);
    cf_multisequence_reclaim_async(runtime->host_cache, stream);
    runtime->stats.shutdown_audits++;
    const bool valid = cf_ms_cuda_audit(runtime->cuda_resolver) == 0 && cf_multisequence_audit(runtime->host_cache) == 0;
    if (!valid) {
        runtime->stats.shutdown_failures++;
        return fail(runtime, "shutdown invariant audit failed");
    }
    runtime->error.clear();
    return 0;
}

extern "C" int cf_llama_server_get_stats(const cf_llama_server_runtime * runtime, cf_llama_server_stats * out) {
    if (!runtime || !out) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    *out = runtime->stats;
    return 0;
}

extern "C" const char * cf_llama_server_last_error(const cf_llama_server_runtime * runtime) {
    return runtime ? runtime->error.c_str() : "runtime is null";
}
