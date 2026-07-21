#ifndef CHROMOFOLD_LLAMA_KV_ADAPTER_H
#define CHROMOFOLD_LLAMA_KV_ADAPTER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cf_llama_kv_backend {
    CF_LLAMA_KV_BACKEND_DENSE = 0,
    CF_LLAMA_KV_BACKEND_CHROMOFOLD = 1,
} cf_llama_kv_backend;

typedef struct cf_llama_kv_options {
    uint32_t struct_size;
    cf_llama_kv_backend backend;
    uint32_t layer_count;
    uint32_t kv_head_count;
    uint32_t query_head_count;
    uint32_t head_dim;
    uint32_t page_size;
    uint32_t gqa_group_size;
} cf_llama_kv_options;

typedef struct cf_llama_kv_stats {
    uint64_t appended_tokens;
    uint64_t sealed_tokens;
    uint64_t sealed_pages;
    uint64_t active_tokens;
    uint64_t dense_active_bytes;
    uint64_t compressed_bytes;
    uint64_t descriptor_bytes;
} cf_llama_kv_stats;

typedef struct cf_llama_kv_adapter cf_llama_kv_adapter;

int cf_llama_kv_options_validate(const cf_llama_kv_options * options);
cf_llama_kv_adapter * cf_llama_kv_create(const cf_llama_kv_options * options);
void cf_llama_kv_destroy(cf_llama_kv_adapter * adapter);

int cf_llama_kv_append(cf_llama_kv_adapter * adapter,
                       uint32_t layer,
                       uint32_t kv_head,
                       uint32_t token_begin,
                       const float * keys,
                       const float * values,
                       uint32_t token_count,
                       void * stream);

int cf_llama_kv_flush(cf_llama_kv_adapter * adapter, void * stream);
int cf_llama_kv_seq_rm(cf_llama_kv_adapter * adapter, uint64_t sequence_id);
int cf_llama_kv_seq_cp(cf_llama_kv_adapter * adapter, uint64_t source_id, uint64_t destination_id);
int cf_llama_kv_seq_shift(cf_llama_kv_adapter * adapter, uint64_t sequence_id, int32_t delta);
int cf_llama_kv_clear(cf_llama_kv_adapter * adapter);
int cf_llama_kv_get_stats(const cf_llama_kv_adapter * adapter, cf_llama_kv_stats * stats);

const char * cf_llama_kv_last_error(const cf_llama_kv_adapter * adapter);

#ifdef __cplusplus
}
#endif

#endif
