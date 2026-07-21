#ifndef CHROMOFOLD_DEVICE_KV_SEAL_ATTENTION_H
#define CHROMOFOLD_DEVICE_KV_SEAL_ATTENTION_H

#include <stdint.h>
#include "chromofold/device_kv_dataplane.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cf_device_seal_state {
    CF_SEAL_ACTIVE_FILLING = 0,
    CF_SEAL_ACTIVE_FULL = 1,
    CF_SEAL_VALIDATING = 2,
    CF_SEAL_QUANTIZING = 3,
    CF_SEAL_PACKING = 4,
    CF_SEAL_PUBLISHING = 5,
    CF_SEAL_FAILED = 6,
} cf_device_seal_state;

typedef enum cf_device_seal_fault {
    CF_SEAL_FAULT_NONE = 0,
    CF_SEAL_FAULT_CANDIDATE_ALLOC = 1,
    CF_SEAL_FAULT_FINITE_VALIDATION = 2,
    CF_SEAL_FAULT_SCALE_ALLOC = 3,
    CF_SEAL_FAULT_QUANTIZE_LAUNCH = 4,
    CF_SEAL_FAULT_PAYLOAD_ALLOC = 5,
    CF_SEAL_FAULT_PACK_LAUNCH = 6,
    CF_SEAL_FAULT_DESCRIPTOR_VALIDATE = 7,
    CF_SEAL_FAULT_PAGE_TABLE_GROW = 8,
    CF_SEAL_FAULT_DESCRIPTOR_PUBLISH = 9,
    CF_SEAL_FAULT_ACTIVE_REPLACEMENT = 10,
} cf_device_seal_fault;

typedef struct cf_device_page_descriptor {
    const uint8_t * key_payload;
    const uint8_t * value_payload;
    const float * key_scales;
    const float * value_scales;
    uint32_t token_begin;
    uint32_t token_count;
    uint32_t kv_head_count;
    uint32_t head_dim;
    uint32_t block_size;
    uint32_t blocks_per_page;
    uint64_t key_payload_bytes;
    uint64_t value_payload_bytes;
    uint64_t resident_bytes;
    uint64_t generation;
} cf_device_page_descriptor;

typedef struct cf_device_seal_counters {
    uint64_t seal_attempts;
    uint64_t seal_successes;
    uint64_t seal_failures;
    uint64_t finite_validation_launches;
    uint64_t values_scanned;
    uint64_t nan_values;
    uint64_t inf_values;
    uint64_t invalid_pages_rejected;
    uint64_t quantization_launches;
    uint64_t values_quantized;
    uint64_t saturation_count;
    uint64_t packing_launches;
    uint64_t symbols_packed;
    uint64_t payload_bytes_written;
    uint64_t descriptor_validations;
    uint64_t descriptor_publications;
    uint64_t page_table_reallocations;
    uint64_t page_table_generation;
    uint64_t compressed_attention_launches;
    uint64_t sealed_pages_consumed;
    uint64_t sealed_values_consumed;
    uint64_t active_values_consumed;
    uint64_t query_conversion_launches;
    uint64_t output_conversion_launches;
    uint64_t cuda_errors;
    uint64_t compressed_bytes;
    uint64_t descriptor_bytes;
    uint64_t workspace_bytes;
    uint64_t peak_owned_bytes;
} cf_device_seal_counters;

typedef struct cf_device_seal_config {
    uint32_t layer_count;
    uint32_t kv_head_count;
    uint32_t query_head_count;
    uint32_t head_dim;
    uint32_t page_size;
    uint32_t gqa_group_size;
    uint32_t block_size;
    uint32_t use_async_allocator;
} cf_device_seal_config;

typedef struct cf_device_seal_engine cf_device_seal_engine;

cf_device_seal_engine * cf_device_seal_create(const cf_device_seal_config * config);
void cf_device_seal_destroy(cf_device_seal_engine * engine, void * stream);
int cf_device_seal_stage_active_page(cf_device_seal_engine * engine,
                                     uint32_t layer,
                                     uint32_t token_begin,
                                     const float * active_k,
                                     const float * active_v,
                                     uint32_t token_count,
                                     void * stream);
int cf_device_seal_full_page_async(cf_device_seal_engine * engine, uint32_t layer, void * stream);
int cf_device_seal_attention_async(cf_device_seal_engine * engine,
                                   uint32_t layer,
                                   const cf_device_tensor_view * query,
                                   void * output,
                                   uint32_t output_scalar_type,
                                   uint64_t output_token_stride_bytes,
                                   uint64_t output_head_stride_bytes,
                                   uint64_t output_element_stride_bytes,
                                   uint32_t query_token_begin,
                                   float softmax_scale,
                                   uint32_t causal_window,
                                   void * stream);
int cf_device_seal_clear_async(cf_device_seal_engine * engine, void * stream);
int cf_device_seal_set_fault(cf_device_seal_engine * engine, uint32_t fault);
int cf_device_seal_get_state(const cf_device_seal_engine * engine, uint32_t layer, uint32_t * state);
int cf_device_seal_get_counters(const cf_device_seal_engine * engine, cf_device_seal_counters * counters);
const char * cf_device_seal_last_error(const cf_device_seal_engine * engine);

#ifdef __cplusplus
}
#endif
#endif
