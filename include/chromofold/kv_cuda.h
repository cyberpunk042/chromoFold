#ifndef CHROMOFOLD_KV_CUDA_H
#define CHROMOFOLD_KV_CUDA_H

#include <stdint.h>

#include "chromofold/chromofold.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CF_KV_CUDA_ABI_VERSION 0u
#define CF_KV_CUDA_MAX_PAGES_PER_LAUNCH 4096u

/* Device-resident immutable page descriptor.
 *
 * The descriptor array and every pointer referenced by a descriptor are device
 * pointers. K uses per-channel scales; V uses per-token scales. Bit offsets are
 * measured from the first bit of the corresponding packed word stream.
 */
typedef struct cf_kv_device_page {
    const uint32_t *k_words;
    const int32_t *k_block_offsets;
    const int32_t *k_lut;
    const float *k_scales;

    const uint32_t *v_words;
    const int32_t *v_block_offsets;
    const int32_t *v_lut;
    const float *v_scales;

    uint32_t token_begin;
    uint32_t token_count;
    uint32_t head_dim;
    uint32_t block_size;
    uint32_t k_max_code_length;
    uint32_t v_max_code_length;
    uint32_t zero_point;
    uint32_t kv_head;
} cf_kv_device_page;

/* Launch description for one layer across a batch of query tokens.
 *
 * queries and output are [query_count, query_head_count, head_dim]. The device
 * page array contains pages for all KV heads. Pages for one KV head must not
 * overlap. Query head h maps to KV head floor(h / gqa_group_size).
 *
 * active_k and active_v are optional dense fp32 tails with layout
 * [active_token_count, kv_head_count, head_dim]. Their logical token interval is
 * [active_token_begin, active_token_begin + active_token_count).
 */
typedef struct cf_kv_paged_attention_desc {
    uint32_t struct_size;
    uint32_t abi_version;

    const cf_kv_device_page *pages;
    uint32_t page_count;
    uint32_t kv_head_count;
    uint32_t query_head_count;
    uint32_t gqa_group_size;

    const float *active_k;
    const float *active_v;
    uint32_t active_token_begin;
    uint32_t active_token_count;

    const float *queries;
    float *output;
    uint32_t query_token_begin;
    uint32_t query_count;
    uint32_t head_dim;
    uint32_t causal_window;
    float softmax_scale;
} cf_kv_paged_attention_desc;

/* Host-side structural validation. This does not dereference device pointers. */
cf_status cf_kv_validate_paged_attention_desc(const cf_kv_paged_attention_desc *desc);

/* Asynchronous paged compressed attention.
 *
 * No allocation, synchronization, or host/device transfer is performed. The
 * caller owns every allocation and stream dependency. Historical K/V values are
 * decoded inside the consumer kernel and are never materialized as dense pages.
 */
cf_status cf_kv_paged_attention_async(const cf_kv_paged_attention_desc *desc, void *stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CHROMOFOLD_KV_CUDA_H */
