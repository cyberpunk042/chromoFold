#include "device_kv_graph_bridge.h"

#include <memory>
#include <string>

struct cf_llama_device_graph {
    cf_llama_device_graph_options options{};
    cf_device_kv_dataplane * dataplane = nullptr;
    std::string last_error;
};

extern "C" cf_llama_device_graph * cf_llama_device_graph_create(const cf_llama_device_graph_options * options) {
    if (!options || options->struct_size != sizeof(cf_llama_device_graph_options) || options->causal == 0 ||
        options->linear_sequence_only == 0 || options->dense_fallback_forbidden == 0) return nullptr;
    auto graph = std::make_unique<cf_llama_device_graph>();
    graph->options = *options;
    graph->dataplane = cf_device_kv_create(&options->kv);
    if (!graph->dataplane) return nullptr;
    return graph.release();
}

extern "C" void cf_llama_device_graph_destroy(cf_llama_device_graph * graph, void * stream) {
    if (!graph) return;
    cf_device_kv_destroy(graph->dataplane, stream);
    delete graph;
}

extern "C" int cf_llama_device_graph_append_async(cf_llama_device_graph * graph,
                                                    uint32_t layer,
                                                    uint32_t token_begin,
                                                    const cf_device_tensor_view * keys,
                                                    const cf_device_tensor_view * values,
                                                    void * stream) {
    if (!graph || !graph->dataplane) return -1;
    const int result = cf_device_kv_append_async(graph->dataplane, layer, token_begin, keys, values, stream);
    if (result != 0) graph->last_error = cf_device_kv_last_error(graph->dataplane);
    else graph->last_error.clear();
    return result;
}

extern "C" int cf_llama_device_graph_attention_async(cf_llama_device_graph * graph,
                                                       uint32_t layer,
                                                       const float * query,
                                                       float * output,
                                                       uint32_t query_token_count,
                                                       uint32_t query_token_begin,
                                                       float softmax_scale,
                                                       uint32_t causal_window,
                                                       void * stream) {
    if (!graph || !graph->dataplane) return -1;
    const int result = cf_device_kv_attention_async(graph->dataplane, layer, query, output, query_token_count,
                                                     query_token_begin, softmax_scale, causal_window, stream);
    if (result != 0) graph->last_error = cf_device_kv_last_error(graph->dataplane);
    else graph->last_error.clear();
    return result;
}

extern "C" int cf_llama_device_graph_get_counters(const cf_llama_device_graph * graph,
                                                    cf_device_kv_counters * counters) {
    return graph ? cf_device_kv_get_counters(graph->dataplane, counters) : -1;
}

extern "C" const char * cf_llama_device_graph_last_error(const cf_llama_device_graph * graph) {
    return graph ? graph->last_error.c_str() : "null llama device graph";
}
