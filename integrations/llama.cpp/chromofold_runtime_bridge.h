#ifndef CHROMOFOLD_LLAMA_RUNTIME_BRIDGE_H
#define CHROMOFOLD_LLAMA_RUNTIME_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#include "chromofold_kv_adapter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cf_llama_scalar_type {
    CF_LLAMA_SCALAR_F32 = 0,
    CF_LLAMA_SCALAR_F16 = 1,
} cf_llama_scalar_type;

typedef struct cf_llama_tensor_view {
    const void * data;
    uint64_t token_stride_bytes;
    uint64_t head_stride_bytes;
    uint64_t element_stride_bytes;
    uint32_t token_count;
    uint32_t head_count;
    uint32_t head_dim;
    cf_llama_scalar_type scalar_type;
} cf_llama_tensor_view;

typedef struct cf_llama_runtime_options {
    uint32_t struct_size;
    cf_llama_kv_options kv;
    uint32_t causal_window;
    uint32_t require_cuda;
    uint32_t allow_dense_fallback;
    uint32_t emit_stats;
} cf_llama_runtime_options;

typedef struct cf_llama_runtime_counters {
    uint64_t append_calls;
    uint64_t appended_values;
    uint64_t compressed_attention_launches;
    uint64_t sealed_values_consumed;
    uint64_t active_values_consumed;
    uint64_t dense_fallback_launches;
    uint64_t rejected_operations;
    uint64_t cuda_errors;
} cf_llama_runtime_counters;

typedef struct cf_llama_runtime_snapshot {
    cf_llama_kv_stats kv;
    cf_llama_runtime_counters counters;
    uint32_t backend_selected;
    uint32_t initialized;
    uint32_t dense_fallback_allowed;
    uint32_t evidence_complete;
} cf_llama_runtime_snapshot;

typedef struct cf_llama_runtime cf_llama_runtime;

int cf_llama_runtime_options_validate(const cf_llama_runtime_options * options);
cf_llama_runtime * cf_llama_runtime_create(const cf_llama_runtime_options * options);
void cf_llama_runtime_destroy(cf_llama_runtime * runtime);

int cf_llama_runtime_append(cf_llama_runtime * runtime,
                            uint32_t layer,
                            uint32_t token_begin,
                            const cf_llama_tensor_view * keys,
                            const cf_llama_tensor_view * values,
                            void * stream);

int cf_llama_runtime_record_compressed_launch(cf_llama_runtime * runtime,
                                               uint64_t sealed_values,
                                               uint64_t active_values);
int cf_llama_runtime_record_dense_fallback(cf_llama_runtime * runtime);
int cf_llama_runtime_record_cuda_error(cf_llama_runtime * runtime);
int cf_llama_runtime_snapshot_get(const cf_llama_runtime * runtime,
                                  cf_llama_runtime_snapshot * snapshot);
int cf_llama_runtime_write_json(const cf_llama_runtime * runtime, const char * path);
const char * cf_llama_runtime_last_error(const cf_llama_runtime * runtime);

#ifdef __cplusplus
}
#endif

#endif
