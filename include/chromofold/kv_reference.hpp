#ifndef CHROMOFOLD_KV_REFERENCE_HPP
#define CHROMOFOLD_KV_REFERENCE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace chromofold::reference {

struct SealedKvPage {
    uint32_t token_begin = 0;
    uint32_t token_count = 0;
    uint32_t layer = 0;
    uint32_t kv_head = 0;
    uint32_t head_dim = 0;

    // Symmetric signed-int4 reference representation. Two values are packed per byte.
    // K uses one scale per channel; V uses one scale per token, matching the M6/M9 codec contract.
    std::vector<uint8_t> packed_k;
    std::vector<uint8_t> packed_v;
    std::vector<float> k_scales;
    std::vector<float> v_scales;

    [[nodiscard]] std::size_t value_count() const noexcept;
    [[nodiscard]] std::size_t resident_bytes() const noexcept;
    [[nodiscard]] bool valid() const noexcept;
};

struct ActiveKvTail {
    uint32_t token_begin = 0;
    uint32_t token_count = 0;
    uint32_t layer = 0;
    uint32_t kv_head = 0;
    uint32_t head_dim = 0;
    const float *k = nullptr; // row-major [token_count, head_dim]
    const float *v = nullptr; // row-major [token_count, head_dim]

    [[nodiscard]] bool valid() const noexcept;
};

struct AttentionShape {
    uint32_t query_count = 0;
    uint32_t query_head_count = 0;
    uint32_t kv_head_count = 0;
    uint32_t head_dim = 0;
    uint32_t query_token_begin = 0;
    uint32_t causal_window = 0; // zero means all available history
    float score_scale = 1.0f;
};

struct AttentionDiagnostics {
    uint64_t attended_values = 0;
    uint64_t sealed_values = 0;
    uint64_t active_values = 0;
    uint64_t sealed_payload_bytes = 0;
};

// Seal dense fp32 K/V into a deterministic signed-int4 page. This is a correctness/reference
// encoder, not the final entropy encoder. It freezes scale orientation and page lifecycle before
// the CUDA/Huffman implementation is connected.
SealedKvPage seal_int4_page(const float *k, const float *v,
                            uint32_t token_begin, uint32_t token_count,
                            uint32_t layer, uint32_t kv_head, uint32_t head_dim);

void dequantize_page(const SealedKvPage &page, std::vector<float> *k, std::vector<float> *v);

// Dense oracle over sealed pages plus an optional active tail. Queries are row-major
// [query_count, query_head_count, head_dim]. Output has the same shape. GQA maps query head h to
// KV head floor(h * kv_head_count / query_head_count). Page order is irrelevant; overlapping
// pages for the same layer/head are rejected.
void paged_attention(const std::vector<SealedKvPage> &pages,
                     const std::vector<ActiveKvTail> &active_tails,
                     const float *queries, const AttentionShape &shape,
                     std::vector<float> *output,
                     AttentionDiagnostics *diagnostics = nullptr);

} // namespace chromofold::reference

#endif
