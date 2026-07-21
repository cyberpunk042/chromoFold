#include "chromofold_kv_adapter.h"

#include "chromofold/compressed_kv_cache.hpp"

#include <exception>
#include <memory>
#include <string>

struct cf_llama_kv_adapter {
    std::unique_ptr<chromofold::CompressedKvCache> cache;
    std::string last_error;
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
