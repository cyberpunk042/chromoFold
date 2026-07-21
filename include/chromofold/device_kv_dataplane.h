#ifndef CHROMOFOLD_DEVICE_KV_DATAPLANE_H
#define CHROMOFOLD_DEVICE_KV_DATAPLANE_H

#include <stdint.h>
#include "chromofold/kv_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cf_device_scalar_type {
    CF_DEVICE_F16 = 1,
    CF_DEVICE_BF16 = 2,
    CF_DEVICE_F32 = 3,
} cf_device_scalar_type;

typedef struct cf_device_tensor_view {
    const void * data;
    uint32_t scalar_type;
    uint32_t token_count;
    uint32_t head_count;
    uint32_t head_dim;
    uint64_t token_stride_bytes;
    uint64_t head_stride_bytes;
    uint64_t element_stride_bytes;
    int32_t device_ordinal;
} cf_device_tensor_view;

typedef struct cf_device_kv_config {
    uint32_t layer_count;
    uint32_t kv_head_count;
    uint32_t query_head_count;
    uint32_t head_dim;
    uint32_t page_size;
    uint32_t gqa_group_size;
    uint32_t use_async_allocator;
} cf_device_kv_config;

typedef struct cf_device_kv_counters {
    uint64_t device_append_calls;
    uint64_t device_values_appended;
    uint64_t fp16_values_converted;
    uint64_t bf16_values_converted;
    uint64_t fp32_values_copied;
    uint64_t quantization_launches;
    uint64_t huffman_pack_launches;
    uint64_t seal_attempts;
    uint64_t seal_successes;
    uint64_t seal_failures;
    uint64_t descriptor_publications;
    uint64_t compressed_attention_launches;
    uint64_t sealed_values_consumed;
    uint64_t active_values_consumed;
    uint64_t query_conversion_launches;
    uint64_t output_conversion_launches;
    uint64_t dense_fallback_launches;
    uint64_t cuda_errors;
    uint64_t active_bytes;
    uint64_t compressed_bytes;
    uint64_t descriptor_bytes;
    uint64_t workspace_bytes;
    uint64_t retired_pending_bytes;
    uint64_t peak_owned_bytes;
} cf_device_kv_counters;

typedef struct cf_device_kv_dataplane cf_device_kv_dataplane;

int cf_device_tensor_validate(const cf_device_tensor_view * view);
cf_device_kv_dataplane * cf_device_kv_create(const cf_device_kv_config * config);
void cf_device_kv_destroy(cf_device_kv_dataplane * dataplane, void * stream);
int cf_device_kv_append_async(cf_device_kv_dataplane * dataplane,
                              uint32_t layer,
                              uint32_t token_begin,
                              const cf_device_tensor_view * keys,
                              const cf_device_tensor_view * values,
                              void * stream);
int cf_device_kv_seal_full_pages_async(cf_device_kv_dataplane * dataplane, void * stream);
int cf_device_kv_attention_async(cf_device_kv_dataplane * dataplane,
                                 uint32_t layer,
                                 const float * query,
                                 float * output,
                                 uint32_t query_token_count,
                                 uint32_t query_token_begin,
                                 float softmax_scale,
                                 uint32_t causal_window,
                                 void * stream);
int cf_device_kv_clear_async(cf_device_kv_dataplane * dataplane, void * stream);
int cf_device_kv_get_counters(const cf_device_kv_dataplane * dataplane, cf_device_kv_counters * counters);
const char * cf_device_kv_last_error(const cf_device_kv_dataplane * dataplane);

#ifdef __cplusplus
}
#endif
#endif
