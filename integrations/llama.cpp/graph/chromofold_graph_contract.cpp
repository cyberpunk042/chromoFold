#include "chromofold_graph_contract.h"

#include <memory>
#include <stdexcept>
#include <string>

struct cf_llama_graph_state {
    cf_llama_graph_options options{};
    cf_llama_model_topology topology{};
    std::unique_ptr<cf_llama_runtime, void (*)(cf_llama_runtime *)> runtime{nullptr, cf_llama_runtime_destroy};
    cf_llama_graph_counters counters{};
    std::string evidence_path;
    std::string last_error;
};

namespace {
template <typename Operation>
int guarded(cf_llama_graph_state * state, Operation && operation) {
    if (state == nullptr) return -1;
    try {
        operation();
        state->last_error.clear();
        return 0;
    } catch (const std::exception & error) {
        state->last_error = error.what();
    } catch (...) {
        state->last_error = "unknown graph wiring error";
    }
    state->counters.unsupported_graphs++;
    return -1;
}
}

extern "C" int cf_llama_graph_options_validate(const cf_llama_graph_options * options) {
    if (options == nullptr || options->struct_size != sizeof(cf_llama_graph_options)) return -1;
    if (options->backend == CF_LLAMA_KV_BACKEND_DENSE) return 0;
    if (options->backend != CF_LLAMA_KV_BACKEND_CHROMOFOLD) return -1;
    if (options->page_size != 64 && options->page_size != 128 && options->page_size != 256) return -1;
    return 0;
}

extern "C" int cf_llama_model_topology_validate(const cf_llama_model_topology * topology) {
    if (topology == nullptr || topology->struct_size != sizeof(cf_llama_model_topology)) return -1;
    if (!topology->uniform_kv_geometry || !topology->causal || topology->recurrent) return -1;
    if (!topology->layer_count || !topology->query_head_count || !topology->kv_head_count || !topology->head_dim) return -1;
    if (topology->head_dim > 128 || topology->query_head_count % topology->kv_head_count != 0) return -1;
    return 0;
}

extern "C" cf_llama_graph_state * cf_llama_graph_create(const cf_llama_graph_options * options,
                                                           const cf_llama_model_topology * topology) {
    if (cf_llama_graph_options_validate(options) != 0 || cf_llama_model_topology_validate(topology) != 0) return nullptr;
    try {
        auto state = std::make_unique<cf_llama_graph_state>();
        state->options = *options;
        state->topology = *topology;
        state->evidence_path = options->evidence_path ? options->evidence_path : "";
        if (options->backend == CF_LLAMA_KV_BACKEND_CHROMOFOLD) {
            cf_llama_runtime_options runtime{};
            runtime.struct_size = sizeof(runtime);
            runtime.kv.struct_size = sizeof(runtime.kv);
            runtime.kv.backend = options->backend;
            runtime.kv.layer_count = topology->layer_count;
            runtime.kv.kv_head_count = topology->kv_head_count;
            runtime.kv.query_head_count = topology->query_head_count;
            runtime.kv.head_dim = topology->head_dim;
            runtime.kv.page_size = options->page_size;
            runtime.kv.gqa_group_size = topology->query_head_count / topology->kv_head_count;
            runtime.require_cuda = 1;
            runtime.allow_dense_fallback = 0;
            state->runtime.reset(cf_llama_runtime_create(&runtime));
            if (!state->runtime) return nullptr;
        }
        return state.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void cf_llama_graph_destroy(cf_llama_graph_state * state) { delete state; }

extern "C" int cf_llama_graph_append_device(cf_llama_graph_state * state,
                                              uint32_t layer,
                                              uint32_t token_begin,
                                              const cf_llama_tensor_view * keys,
                                              const cf_llama_tensor_view * values,
                                              void * stream) {
    return guarded(state, [&] {
        if (!state->runtime) throw std::runtime_error("ChromoFold graph backend is not active");
        if (cf_llama_runtime_append(state->runtime.get(), layer, token_begin, keys, values, stream) != 0)
            throw std::runtime_error(cf_llama_runtime_last_error(state->runtime.get()));
        state->counters.graph_append_calls++;
    });
}

extern "C" int cf_llama_graph_record_attention(cf_llama_graph_state * state,
                                                 uint64_t sealed_values,
                                                 uint64_t active_values) {
    return guarded(state, [&] {
        if (!state->runtime) throw std::runtime_error("compressed attention recorded without runtime");
        if (cf_llama_runtime_record_compressed_launch(state->runtime.get(), sealed_values, active_values) != 0)
            throw std::runtime_error(cf_llama_runtime_last_error(state->runtime.get()));
        state->counters.graph_attention_dispatches++;
    });
}

extern "C" int cf_llama_graph_record_dense_dispatch(cf_llama_graph_state * state) {
    return guarded(state, [&] {
        state->counters.graph_dense_dispatches++;
        if (state->options.backend == CF_LLAMA_KV_BACKEND_CHROMOFOLD) {
            if (state->runtime) cf_llama_runtime_record_dense_fallback(state->runtime.get());
            throw std::runtime_error("dense graph dispatch is forbidden for ChromoFold");
        }
    });
}

extern "C" int cf_llama_graph_get_counters(const cf_llama_graph_state * state,
                                             cf_llama_graph_counters * counters) {
    if (!state || !counters) return -1;
    *counters = state->counters;
    return 0;
}

extern "C" int cf_llama_graph_write_evidence(const cf_llama_graph_state * state) {
    if (!state || !state->runtime || state->evidence_path.empty()) return -1;
    return cf_llama_runtime_write_json(state->runtime.get(), state->evidence_path.c_str());
}

extern "C" const char * cf_llama_graph_last_error(const cf_llama_graph_state * state) {
    return state ? state->last_error.c_str() : "null graph state";
}
