#pragma once

#include <stdint.h>
#include "chromofold/device_kv_seal_attention.h"
#include "chromofold/multisequence_cuda_resolver.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cf_attention_kernel_mode {
    CF_ATTENTION_REFERENCE = 0,
    CF_ATTENTION_OPTIMIZED = 1
} cf_attention_kernel_mode;

typedef struct cf_attention_tuning {
    uint32_t threads_per_block;
    uint32_t warps_per_block;
    uint32_t heads_per_block;
    uint32_t pages_per_work_unit;
    uint32_t vector_width;
    uint32_t enable_cuda_graphs;
    uint32_t enable_async_sealing;
} cf_attention_tuning;

typedef struct cf_attention_runtime_config {
    uint32_t layer_count;
    uint32_t query_head_count;
    uint32_t kv_head_count;
    uint32_t head_dim;
    uint32_t page_size;
    uint32_t block_size;
    uint32_t gqa_group_size;
    uint32_t cuda_device;
    uint32_t use_async_allocator;
    cf_attention_kernel_mode mode;
    cf_attention_tuning tuning;
} cf_attention_runtime_config;

typedef struct cf_attention_batch_item {
    int64_t sequence_id;
    uint32_t layer;
    cf_device_tensor_view query;
    void * output;
    uint32_t output_type;
    uint64_t output_token_stride_bytes;
    uint64_t output_head_stride_bytes;
    uint64_t output_element_stride_bytes;
    uint32_t query_token_begin;
    float softmax_scale;
    uint32_t causal_window;
    const cf_device_page_descriptor * sealed_pages;
    uint32_t sealed_page_count;
    const float * active_keys;
    const float * active_values;
    uint32_t active_token_begin;
    uint32_t active_token_count;
} cf_attention_batch_item;

typedef struct cf_attention_counters {
    uint64_t reference_launches;
    uint64_t optimized_launches;
    uint64_t decode_launches;
    uint64_t prefill_launches;
    uint64_t grouped_batches;
    uint64_t grouped_items;
    uint64_t kernel_launches_avoided;
    uint64_t graph_captures;
    uint64_t graph_replays;
    uint64_t graph_invalidations;
    uint64_t graph_replay_failures;
    uint64_t async_seal_submissions;
    uint64_t async_seal_overlaps;
    uint64_t descriptor_bytes_read;
    uint64_t compressed_payload_bytes_read;
    uint64_t scale_bytes_read;
    uint64_t allocation_count;
    uint64_t pool_hits;
    uint64_t pool_misses;
    uint64_t peak_reserved_bytes;
    uint64_t peak_live_bytes;
    uint64_t numerical_failures;
    uint64_t cross_sequence_contamination;
    uint64_t dense_fallback_launches;
    uint64_t cuda_errors;
} cf_attention_counters;

typedef struct cf_attention_runtime cf_attention_runtime;

cf_attention_runtime * cf_attention_runtime_create(const cf_attention_runtime_config * config);
void cf_attention_runtime_destroy(cf_attention_runtime * runtime, void * stream);
int cf_attention_execute_async(cf_attention_runtime * runtime,
                               const cf_attention_batch_item * items,
                               uint32_t item_count,
                               void * stream);
int cf_attention_compare_async(cf_attention_runtime * runtime,
                               const cf_attention_batch_item * items,
                               uint32_t item_count,
                               float max_abs_error,
                               float mean_abs_error,
                               void * stream);
int cf_attention_invalidate_graphs(cf_attention_runtime * runtime);
int cf_attention_get_counters(const cf_attention_runtime * runtime, cf_attention_counters * counters);
const char * cf_attention_last_error(const cf_attention_runtime * runtime);

#ifdef __cplusplus
}
#endif
