#ifndef CHROMOFOLD_COMPRESSED_KV_CACHE_HPP
#define CHROMOFOLD_COMPRESSED_KV_CACHE_HPP

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "chromofold/kv_cuda.h"
#include "chromofold/kv_cuda_owner.hpp"
#include "chromofold/kv_gpu_fixture.hpp"

namespace chromofold {

enum class KvCodecMode : std::uint32_t {
    fixed_int4_huffman = 0,
    adaptive_huffman = 1,
    fixed_int8_huffman = 2,
};

struct KvCacheConfig {
    std::uint32_t layer_count = 0;
    std::uint32_t kv_head_count = 0;
    std::uint32_t query_head_count = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t page_size = 128;
    std::uint32_t gqa_group_size = 1;
    KvCodecMode codec = KvCodecMode::fixed_int4_huffman;
};

struct KvCacheStats {
    std::uint64_t appended_tokens = 0;
    std::uint64_t sealed_tokens = 0;
    std::uint64_t sealed_pages = 0;
    std::uint64_t active_tokens = 0;
    std::uint64_t dense_active_bytes = 0;
    std::uint64_t compressed_bytes = 0;
    std::uint64_t descriptor_bytes = 0;
};

struct KvAttentionView {
    std::vector<cf_kv_device_page> host_descriptors;
    const cf_kv_device_page* device_descriptors = nullptr;
    std::uint32_t page_count = 0;
    const float* active_k = nullptr;
    const float* active_v = nullptr;
    std::uint32_t active_token_begin = 0;
    std::uint32_t active_token_count = 0;
};

class CompressedKvCache {
public:
    explicit CompressedKvCache(KvCacheConfig config);
    CompressedKvCache(const CompressedKvCache&) = delete;
    CompressedKvCache& operator=(const CompressedKvCache&) = delete;
    CompressedKvCache(CompressedKvCache&&) noexcept;
    CompressedKvCache& operator=(CompressedKvCache&&) noexcept;
    ~CompressedKvCache();

    void append(std::uint32_t layer,
                std::uint32_t kv_head,
                std::uint32_t token_begin,
                const float* keys,
                const float* values,
                std::uint32_t token_count,
                void* stream = nullptr);

    void seal_full_pages(void* stream = nullptr);
    void flush(void* stream = nullptr);
    KvAttentionView attention_view(std::uint32_t layer, void* stream = nullptr);

    void remove_sequence(std::uint64_t sequence_id);
    void copy_sequence(std::uint64_t source_sequence_id, std::uint64_t destination_sequence_id);
    void shift_sequence(std::uint64_t sequence_id, std::int32_t delta);
    void clear();

    const KvCacheConfig& config() const noexcept { return config_; }
    KvCacheStats stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    KvCacheConfig config_;
};

}  // namespace chromofold

#endif
