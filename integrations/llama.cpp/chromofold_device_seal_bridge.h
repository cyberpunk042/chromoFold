#ifndef CHROMOFOLD_LLAMA_DEVICE_SEAL_BRIDGE_H
#define CHROMOFOLD_LLAMA_DEVICE_SEAL_BRIDGE_H

#include <stdint.h>
#include "chromofold/device_kv_seal_attention.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cf_llama_device_seal_bridge cf_llama_device_seal_bridge;

typedef struct cf_llama_device_seal_options {
    uint32_t struct_size;
    cf_device_seal_config seal;
    uint32_t require_cuda;
    uint32_t allow_dense_fallback;
} cf_llama_device_seal_options;

cf_llama_device_seal_bridge * cf_llama_device_seal_bridge_create(const cf_llama_device_seal_options * options);
void cf_llama_device_seal_bridge_destroy(cf_llama_device_seal_bridge * bridge, void * stream);
int cf_llama_device_seal_bridge_stage(cf_llama_device_seal_bridge * bridge,
                                      uint32_t layer,
                                      uint32_t token_begin,
                                      const float * active_k,
                                      const float * active_v,
                                      uint32_t token_count,
                                      void * stream);
int cf_llama_device_seal_bridge_seal(cf_llama_device_seal_bridge * bridge, uint32_t layer, void * stream);
int cf_llama_device_seal_bridge_attention(cf_llama_device_seal_bridge * bridge,
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
int cf_llama_device_seal_bridge_evidence(const cf_llama_device_seal_bridge * bridge,
                                          cf_device_seal_counters * counters,
                                          uint32_t * complete);
const char * cf_llama_device_seal_bridge_last_error(const cf_llama_device_seal_bridge * bridge);

#ifdef __cplusplus
}
#endif
#endif
