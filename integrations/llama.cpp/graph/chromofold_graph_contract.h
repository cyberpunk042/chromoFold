#ifndef CHROMOFOLD_LLAMA_GRAPH_CONTRACT_H
#define CHROMOFOLD_LLAMA_GRAPH_CONTRACT_H

#include <stddef.h>
#include <stdint.h>

#include "../chromofold_runtime_bridge.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cf_llama_model_topology {
    uint32_t struct_size;
    uint32_t layer_count;
    uint32_t query_head_count;
    uint32_t kv_head_count;
    uint32_t head_dim;
    uint32_t context_size;
    uint32_t uniform_kv_geometry;
    uint32_t causal;
    uint32_t recurrent;
} cf_llama_model_topology;

typedef struct cf_llama_graph_options {
    uint32_t struct_size;
    cf_llama_kv_backend backend;
    uint32_t page_size;
    uint32_t window;
    uint32_t emit_stats;
    const char * evidence_path;
} cf_llama_graph_options;

typedef struct cf_llama_graph_counters {
    uint64_t graph_append_calls;
    uint64_t graph_attention_dispatches;
    uint64_t graph_dense_dispatches;
    uint64_t query_layout_conversions;
    uint64_t output_layout_conversions;
    uint64_t unsupported_graphs;
} cf_llama_graph_counters;

typedef struct cf_llama_graph_state cf_llama_graph_state;

int cf_llama_graph_options_validate(const cf_llama_graph_options * options);
int cf_llama_model_topology_validate(const cf_llama_model_topology * topology);
cf_llama_graph_state * cf_llama_graph_create(const cf_llama_graph_options * options,
                                               const cf_llama_model_topology * topology);
void cf_llama_graph_destroy(cf_llama_graph_state * state);
int cf_llama_graph_append_device(cf_llama_graph_state * state,
                                 uint32_t layer,
                                 uint32_t token_begin,
                                 const cf_llama_tensor_view * keys,
                                 const cf_llama_tensor_view * values,
                                 void * stream);
int cf_llama_graph_record_attention(cf_llama_graph_state * state,
                                    uint64_t sealed_values,
                                    uint64_t active_values);
int cf_llama_graph_record_dense_dispatch(cf_llama_graph_state * state);
int cf_llama_graph_get_counters(const cf_llama_graph_state * state,
                                cf_llama_graph_counters * counters);
int cf_llama_graph_write_evidence(const cf_llama_graph_state * state);
const char * cf_llama_graph_last_error(const cf_llama_graph_state * state);

#ifdef __cplusplus
}
#endif

#endif
