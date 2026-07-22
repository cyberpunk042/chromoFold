#include "chromofold_kv_adapter.h"

#include "chromofold/compressed_kv_cache.hpp"
#include "chromofold/kv_cuda.h"

#include <cuda_runtime.h>

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct cf_llama_kv_adapter {
    std::unique_ptr<chromofold::CompressedKvCache> cache;
    std::string last_error;
    uint32_t head_dim = 0;
};

namespace {

chromofold::KvCacheConfig to_config(const cf_llama_kv_options& options) {
    chromofold::KvCacheConfig config;
    config.layer_count = options.layer_count;
    config.kv_head_count = options.kv_head_count;
    config.query_head_count = options.query_head_count;
    config.head_dim = options.head_dim;
    config.page_size = options.page_size;
    config.gqa_group_size = options.gqa_group_size;
    return config;
}

template <typename Operation>
int guarded(cf_llama_kv_adapter* adapter, Operation&& operation) {
    if (adapter == nullptr || adapter->cache == nullptr) return -1;
    try {
        operation();
        adapter->last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        adapter->last_error = error.what();
        return -1;
    } catch (...) {
        adapter->last_error = "unknown ChromoFold KV error";
        return -1;
    }
}

}  // namespace

extern "C" int cf_llama_kv_options_validate(const cf_llama_kv_options* options) {
    if (options == nullptr || options->struct_size != sizeof(cf_llama_kv_options)) return -1;
    if (options->backend == CF_LLAMA_KV_BACKEND_DENSE) return 0;
    if (options->backend != CF_LLAMA_KV_BACKEND_CHROMOFOLD || options->layer_count == 0 ||
        options->kv_head_count == 0 || options->query_head_count == 0 || options->head_dim == 0 ||
        options->page_size == 0 || options->gqa_group_size == 0 ||
        options->query_head_count != options->kv_head_count * options->gqa_group_size) return -1;
    return 0;
}

extern "C" cf_llama_kv_adapter* cf_llama_kv_create(const cf_llama_kv_options* options) {
    if (cf_llama_kv_options_validate(options) != 0 || options->backend != CF_LLAMA_KV_BACKEND_CHROMOFOLD) return nullptr;
    try {
        auto adapter = std::make_unique<cf_llama_kv_adapter>();
        adapter->cache = std::make_unique<chromofold::CompressedKvCache>(to_config(*options));
        adapter->head_dim = options->head_dim;
        return adapter.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void cf_llama_kv_destroy(cf_llama_kv_adapter* adapter) { delete adapter; }

extern "C" int cf_llama_kv_append(cf_llama_kv_adapter* adapter,
                                    uint32_t layer,
                                    uint32_t kv_head,
                                    uint32_t token_begin,
                                    const float* keys,
                                    const float* values,
                                    uint32_t token_count,
                                    void* stream) {
    return guarded(adapter, [&] { adapter->cache->append(layer, kv_head, token_begin, keys, values, token_count, stream); });
}

// Serve attention for one layer from the compressed cache. queries/output are device pointers with layout
// [query_count, query_head_count, head_dim]; query head h maps to kv head floor(h / gqa_group_size). Covers
// sealed pages + the active tail (via attention_view). causal_window 0 == full causal.
extern "C" int cf_llama_kv_attention(cf_llama_kv_adapter* adapter,
                                     uint32_t layer,
                                     const float* queries_device,
                                     float* output_device,
                                     uint32_t query_count,
                                     uint32_t query_head_count,
                                     uint32_t gqa_group_size,
                                     uint32_t query_token_begin,
                                     float softmax_scale,
                                     uint32_t causal_window,
                                     void* stream) {
    return guarded(adapter, [&] {
        if (queries_device == nullptr || output_device == nullptr || gqa_group_size == 0 ||
            query_head_count == 0 || query_head_count % gqa_group_size != 0) {
            throw std::invalid_argument("invalid ChromoFold attention request");
        }
        chromofold::KvAttentionView view = adapter->cache->attention_view(layer, stream);
        cf_kv_paged_attention_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = CF_KV_CUDA_ABI_VERSION;
        desc.pages = view.device_descriptors;
        desc.page_count = view.page_count;
        desc.kv_head_count = query_head_count / gqa_group_size;
        desc.query_head_count = query_head_count;
        desc.gqa_group_size = gqa_group_size;
        desc.active_k = view.active_k;
        desc.active_v = view.active_v;
        desc.active_token_begin = view.active_token_begin;
        desc.active_token_count = view.active_token_count;
        desc.queries = queries_device;
        desc.output = output_device;
        desc.query_token_begin = query_token_begin;
        desc.query_count = query_count;
        desc.head_dim = adapter->head_dim;
        desc.causal_window = causal_window;
        desc.softmax_scale = softmax_scale;
        if (cf_kv_paged_attention_async(&desc, stream) != CF_OK) {
            throw std::runtime_error("ChromoFold paged attention launch rejected");
        }
    });
}

// Host-pointer convenience wrapper: uploads the queries, serves attention from the compressed cache, and
// downloads the result. queries_host/output_host are [query_count, query_head_count, head_dim] on the host.
// Lets a host-only caller (the ggml graph-eval callback) drive the replace without touching CUDA directly.
extern "C" int cf_llama_kv_attention_host(cf_llama_kv_adapter* adapter,
                                          uint32_t layer,
                                          const float* queries_host,
                                          float* output_host,
                                          uint32_t query_count,
                                          uint32_t query_head_count,
                                          uint32_t gqa_group_size,
                                          uint32_t query_token_begin,
                                          float softmax_scale,
                                          uint32_t causal_window,
                                          void* stream) {
    return guarded(adapter, [&] {
        if (queries_host == nullptr || output_host == nullptr) {
            throw std::invalid_argument("null host attention buffer");
        }
        const std::size_t n = static_cast<std::size_t>(query_count) * query_head_count * adapter->head_dim;
        const std::size_t bytes = n * sizeof(float);
        float* dq = nullptr;
        float* dout = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&dq), bytes) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&dout), bytes) != cudaSuccess) {
            if (dq != nullptr) cudaFree(dq);
            throw std::runtime_error("failed to allocate attention scratch");
        }
        cudaMemcpy(dq, queries_host, bytes, cudaMemcpyHostToDevice);
        int rc = cf_llama_kv_attention(adapter, layer, dq, dout, query_count, query_head_count,
                                       gqa_group_size, query_token_begin, softmax_scale, causal_window, stream);
        if (rc == 0) {
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
            cudaMemcpy(output_host, dout, bytes, cudaMemcpyDeviceToHost);
        }
        cudaFree(dq);
        cudaFree(dout);
        if (rc != 0) throw std::runtime_error(adapter->last_error.empty() ? "attention failed" : adapter->last_error);
    });
}

extern "C" int cf_llama_kv_flush(cf_llama_kv_adapter* adapter, void* stream) {
    return guarded(adapter, [&] { adapter->cache->flush(stream); });
}

extern "C" int cf_llama_kv_seq_rm(cf_llama_kv_adapter* adapter, uint64_t sequence_id) {
    return guarded(adapter, [&] { adapter->cache->remove_sequence(sequence_id); });
}

extern "C" int cf_llama_kv_seq_cp(cf_llama_kv_adapter* adapter, uint64_t source_id, uint64_t destination_id) {
    return guarded(adapter, [&] { adapter->cache->copy_sequence(source_id, destination_id); });
}

extern "C" int cf_llama_kv_seq_shift(cf_llama_kv_adapter* adapter, uint64_t sequence_id, int32_t delta) {
    return guarded(adapter, [&] { adapter->cache->shift_sequence(sequence_id, delta); });
}

extern "C" int cf_llama_kv_clear(cf_llama_kv_adapter* adapter) {
    return guarded(adapter, [&] { adapter->cache->clear(); });
}

extern "C" int cf_llama_kv_get_stats(const cf_llama_kv_adapter* adapter, cf_llama_kv_stats* stats) {
    if (adapter == nullptr || adapter->cache == nullptr || stats == nullptr) return -1;
    const auto source = adapter->cache->stats();
    stats->appended_tokens = source.appended_tokens;
    stats->sealed_tokens = source.sealed_tokens;
    stats->sealed_pages = source.sealed_pages;
    stats->active_tokens = source.active_tokens;
    stats->dense_active_bytes = source.dense_active_bytes;
    stats->compressed_bytes = source.compressed_bytes;
    stats->descriptor_bytes = source.descriptor_bytes;
    return 0;
}

extern "C" const char* cf_llama_kv_last_error(const cf_llama_kv_adapter* adapter) {
    return adapter == nullptr ? "invalid ChromoFold KV adapter" : adapter->last_error.c_str();
}
