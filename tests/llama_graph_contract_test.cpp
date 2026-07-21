#include "integrations/llama.cpp/graph/chromofold_graph_contract.h"

#include <cassert>

int main() {
    cf_llama_graph_options options{};
    options.struct_size = sizeof(options);
    options.backend = CF_LLAMA_KV_BACKEND_CHROMOFOLD;
    options.page_size = 128;
    assert(cf_llama_graph_options_validate(&options) == 0);
    options.page_size = 96;
    assert(cf_llama_graph_options_validate(&options) != 0);

    cf_llama_model_topology topology{};
    topology.struct_size = sizeof(topology);
    topology.layer_count = 32;
    topology.query_head_count = 32;
    topology.kv_head_count = 8;
    topology.head_dim = 128;
    topology.context_size = 8192;
    topology.uniform_kv_geometry = 1;
    topology.causal = 1;
    assert(cf_llama_model_topology_validate(&topology) == 0);
    topology.recurrent = 1;
    assert(cf_llama_model_topology_validate(&topology) != 0);
    return 0;
}
