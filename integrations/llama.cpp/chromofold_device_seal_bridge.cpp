#include "chromofold_device_seal_bridge.h"

#include <memory>
#include <string>

struct cf_llama_device_seal_bridge {
    cf_llama_device_seal_options options{};
    cf_device_seal_engine * engine = nullptr;
    std::string error;
};

extern "C" cf_llama_device_seal_bridge * cf_llama_device_seal_bridge_create(const cf_llama_device_seal_options * options) {
    if (!options || options->struct_size != sizeof(cf_llama_device_seal_options) || !options->require_cuda || options->allow_dense_fallback)
        return nullptr;
    try {
        auto bridge = std::make_unique<cf_llama_device_seal_bridge>();
        bridge->options = *options;
        bridge->engine = cf_device_seal_create(&options->seal);
        if (!bridge->engine) return nullptr;
        return bridge.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void cf_llama_device_seal_bridge_destroy(cf_llama_device_seal_bridge * bridge, void * stream) {
    if (!bridge) return;
    cf_device_seal_destroy(bridge->engine, stream);
    delete bridge;
}

extern "C" int cf_llama_device_seal_bridge_stage(cf_llama_device_seal_bridge * bridge, uint32_t layer,
                                                   uint32_t token_begin, const float * active_k, const float * active_v,
                                                   uint32_t token_count, void * stream) {
    if (!bridge) return -1;
    const int rc = cf_device_seal_stage_active_page(bridge->engine, layer, token_begin, active_k, active_v, token_count, stream);
    bridge->error = rc == 0 ? "" : cf_device_seal_last_error(bridge->engine);
    return rc;
}

extern "C" int cf_llama_device_seal_bridge_seal(cf_llama_device_seal_bridge * bridge, uint32_t layer, void * stream) {
    if (!bridge) return -1;
    const int rc = cf_device_seal_full_page_async(bridge->engine, layer, stream);
    bridge->error = rc == 0 ? "" : cf_device_seal_last_error(bridge->engine);
    return rc;
}

extern "C" int cf_llama_device_seal_bridge_attention(cf_llama_device_seal_bridge * bridge, uint32_t layer,
                                                       const cf_device_tensor_view * query, void * output,
                                                       uint32_t output_scalar_type, uint64_t output_token_stride_bytes,
                                                       uint64_t output_head_stride_bytes, uint64_t output_element_stride_bytes,
                                                       uint32_t query_token_begin, float softmax_scale,
                                                       uint32_t causal_window, void * stream) {
    if (!bridge) return -1;
    const int rc = cf_device_seal_attention_async(bridge->engine, layer, query, output, output_scalar_type,
        output_token_stride_bytes, output_head_stride_bytes, output_element_stride_bytes,
        query_token_begin, softmax_scale, causal_window, stream);
    bridge->error = rc == 0 ? "" : cf_device_seal_last_error(bridge->engine);
    return rc;
}

extern "C" int cf_llama_device_seal_bridge_evidence(const cf_llama_device_seal_bridge * bridge,
                                                      cf_device_seal_counters * counters, uint32_t * complete) {
    if (!bridge || !counters || !complete) return -1;
    if (cf_device_seal_get_counters(bridge->engine, counters) != 0) return -1;
    *complete = counters->seal_successes > 0 &&
                counters->descriptor_publications == counters->seal_successes &&
                counters->compressed_attention_launches > 0 &&
                counters->sealed_values_consumed > 0 &&
                counters->cuda_errors == 0;
    return 0;
}

extern "C" const char * cf_llama_device_seal_bridge_last_error(const cf_llama_device_seal_bridge * bridge) {
    return bridge ? bridge->error.c_str() : "null llama device seal bridge";
}
