#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum cf_page_codec {
    CF_PAGE_INT2_BLOCKWISE = 2,
    CF_PAGE_INT4_BLOCKWISE = 4,
    CF_PAGE_INT8_BLOCKWISE = 8,
    CF_PAGE_FP16_RAW = 16,
} cf_page_codec;

typedef enum cf_compression_policy {
    CF_POLICY_FIXED_INT4 = 0,
    CF_POLICY_QUALITY_BUDGET = 1,
    CF_POLICY_MEMORY_BUDGET = 2,
    CF_POLICY_HYBRID = 3,
} cf_compression_policy;

typedef struct cf_codec_descriptor {
    uint32_t format_version;
    uint32_t codec;
    uint32_t block_size;
    uint32_t flags;
} cf_codec_descriptor;

typedef struct cf_page_analysis {
    float abs_max;
    float variance;
    float outlier_ratio;
    float estimated_int2_error;
    float estimated_int4_error;
    float estimated_int8_error;
    uint64_t nan_values;
    uint64_t inf_values;
} cf_page_analysis;

typedef struct cf_adaptive_config {
    uint32_t policy;
    uint32_t min_bits;
    uint32_t max_bits;
    uint32_t block_size;
    float max_page_error;
    float minimum_cosine_similarity;
    uint64_t memory_budget_bytes;
} cf_adaptive_config;

typedef struct cf_encoded_page {
    cf_codec_descriptor key_codec;
    cf_codec_descriptor value_codec;
    uint8_t * key_payload;
    uint8_t * value_payload;
    float * key_scales;
    float * value_scales;
    uint64_t key_payload_bytes;
    uint64_t value_payload_bytes;
    uint32_t value_count;
    uint64_t checksum;
} cf_encoded_page;

typedef struct cf_adaptive_counters {
    uint64_t int2_pages;
    uint64_t int4_pages;
    uint64_t int8_pages;
    uint64_t fp16_pages;
    uint64_t invalid_candidates_rejected;
    uint64_t quality_escalations;
    uint64_t recompression_attempts;
    uint64_t recompression_successes;
    uint64_t bytes_saved;
    uint64_t persistent_pages_written;
    uint64_t persistent_pages_loaded;
    uint64_t corrupted_records_rejected;
    uint64_t dense_fallback_launches;
} cf_adaptive_counters;

typedef struct cf_adaptive_runtime cf_adaptive_runtime;
cf_adaptive_runtime * cf_adaptive_create(const cf_adaptive_config * config);
void cf_adaptive_destroy(cf_adaptive_runtime * runtime);
int cf_adaptive_analyze(const float * values, uint32_t count, cf_page_analysis * out);
int cf_adaptive_encode(cf_adaptive_runtime * runtime, const float * keys, const float * values, uint32_t count, cf_encoded_page * out);
int cf_adaptive_decode(const cf_encoded_page * page, float * keys, float * values, uint32_t count);
int cf_adaptive_recompress(cf_adaptive_runtime * runtime, const cf_encoded_page * source, uint32_t target_codec, cf_encoded_page * out);
void cf_encoded_page_release(cf_encoded_page * page);
int cf_adaptive_get_counters(const cf_adaptive_runtime * runtime, cf_adaptive_counters * out);
const char * cf_adaptive_last_error(const cf_adaptive_runtime * runtime);
#ifdef __cplusplus
}
#endif
