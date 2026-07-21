#ifndef CHROMOFOLD_LLAMA_DEVICE_KV_GRAPH_BRIDGE_H
#define CHROMOFOLD_LLAMA_DEVICE_KV_GRAPH_BRIDGE_H

#include "chromofold/device_kv_dataplane.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cf_llama_device_graph cf_llama_device_graph;

typedef struct cf_llama_device_graph_options {
    uint32_t struct_size;
    cf_device_kv_config kv;
    uint32_t causal;
    uint32_t linear_sequence_only;
    uint32_t dense_fallback_forbidden;
} cf_llama_device_graph_options;

cf_llama_device_graph * cf_llama_device_graph_create(const cf_llama_device_graph_options * options);
void cf_llama_device_graph_destroy(cf_llama_device_graph * graph, void * stream);
int cf_llama_device_graph_append_async(cf_llama_device_graph * graph,
                                       uint32_t layer,
                                       uint32_t token_begin,
                                       const cf_device_tensor_view * keys,
                                       const cf_device_tensor_view * values,
                                       void * stream);
int cf_llama_device_graph_attention_async(cf_llama_device_graph * graph,
                                          uint32_t layer,
                                          const float * query,
                                          float * output,
                                          uint32_t query_token_count,
                                          uint32_t query_token_begin,
                                          float softmax_scale,
                                          uint32_t causal_window,
                                          void * stream);
int cf_llama_device_graph_get_counters(const cf_llama_device_graph * graph, cf_device_kv_counters * counters);
const char * cf_llama_device_graph_last_error(const cf_llama_device_graph * graph);

#ifdef __cplusplus
}
#endif
#endif
