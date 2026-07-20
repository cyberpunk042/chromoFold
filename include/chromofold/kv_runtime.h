#ifndef CHROMOFOLD_KV_RUNTIME_H
#define CHROMOFOLD_KV_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "chromofold/chromofold.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CF_KV_RUNTIME_ABI_VERSION 0u
#define CF_KV_PAGE_FLAG_SEALED 0x1u
#define CF_KV_PAGE_FLAG_DEVICE_RESIDENT 0x2u

/* Host-visible descriptor for one immutable, entropy-coded KV page.
 *
 * The descriptor itself is host memory. All payload pointers are device pointers
 * when CF_KV_PAGE_FLAG_DEVICE_RESIDENT is set. Counts are explicit so the runtime
 * can validate ownership and account resident bytes without inspecting payloads.
 */
typedef struct cf_kv_page_view {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t layer;
    uint32_t kv_head;
    uint32_t token_begin;
    uint32_t token_count;
    uint32_t head_dim;
    uint32_t block_size;
    uint32_t zero_point;

    const uint32_t *k_words;
    uint64_t k_word_count;
    const int32_t *k_block_offsets;
    uint64_t k_block_count;
    const int32_t *k_lut;
    uint64_t k_lut_count;
    const float *k_scales;
    uint64_t k_scale_count;

    const uint32_t *v_words;
    uint64_t v_word_count;
    const int32_t *v_block_offsets;
    uint64_t v_block_count;
    const int32_t *v_lut;
    uint64_t v_lut_count;
    const float *v_scales;
    uint64_t v_scale_count;
} cf_kv_page_view;

/* A cache consists of immutable sealed pages plus one appendable active tail.
 * `pages` is a host pointer to `page_count` descriptors. The active tail remains
 * runtime-owned dense/quantized storage until it is sealed into a new page.
 */
typedef struct cf_kv_cache_layout {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t layer_count;
    uint32_t kv_head_count;
    uint32_t head_dim;
    uint32_t page_capacity_tokens;
    uint32_t active_token_begin;
    uint32_t active_token_count;
    uint32_t active_element_bytes;
    uint32_t reserved;

    const cf_kv_page_view *pages;
    uint64_t page_count;
    uint64_t active_k_bytes;
    uint64_t active_v_bytes;
} cf_kv_cache_layout;

typedef struct cf_kv_memory_stats {
    uint64_t sealed_payload_bytes;
    uint64_t sealed_descriptor_bytes;
    uint64_t active_payload_bytes;
    uint64_t total_resident_bytes;
    uint64_t sealed_tokens;
    uint64_t active_tokens;
    uint64_t page_count;
} cf_kv_memory_stats;

/* CPU correctness oracle state for numerically stable page-by-page attention.
 * `weighted_values` is caller-owned storage containing `dim` doubles.
 */
typedef struct cf_online_softmax_state {
    double max_score;
    double exp_sum;
    double *weighted_values;
    uint32_t dim;
    uint32_t initialized;
} cf_online_softmax_state;

cf_status cf_kv_validate_page(const cf_kv_page_view *page);
cf_status cf_kv_validate_layout(const cf_kv_cache_layout *layout);
cf_status cf_kv_page_resident_bytes(const cf_kv_page_view *page, uint64_t *out_bytes);
cf_status cf_kv_cache_memory_stats(const cf_kv_cache_layout *layout, cf_kv_memory_stats *out_stats);

cf_status cf_online_softmax_init(cf_online_softmax_state *state, double *weighted_values, uint32_t dim);
cf_status cf_online_softmax_accumulate(cf_online_softmax_state *state, double score, const float *value);
cf_status cf_online_softmax_merge(cf_online_softmax_state *dst, const cf_online_softmax_state *src);
cf_status cf_online_softmax_finalize(const cf_online_softmax_state *state, float *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CHROMOFOLD_KV_RUNTIME_H */
