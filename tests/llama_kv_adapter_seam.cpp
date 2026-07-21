#include "chromofold_kv_adapter.h"

#include <cassert>
#include <cstring>
#include <iostream>

int main() {
    cf_llama_kv_options dense{};
    dense.struct_size = sizeof(dense);
    dense.backend = CF_LLAMA_KV_BACKEND_DENSE;
    assert(cf_llama_kv_options_validate(&dense) == 0);
    assert(cf_llama_kv_create(&dense) == nullptr);

    cf_llama_kv_options options{};
    options.struct_size = sizeof(options);
    options.backend = CF_LLAMA_KV_BACKEND_CHROMOFOLD;
    options.layer_count = 2;
    options.kv_head_count = 2;
    options.query_head_count = 8;
    options.head_dim = 64;
    options.page_size = 128;
    options.gqa_group_size = 4;
    assert(cf_llama_kv_options_validate(&options) == 0);

    cf_llama_kv_adapter* adapter = cf_llama_kv_create(&options);
    assert(adapter != nullptr);

    cf_llama_kv_stats stats{};
    assert(cf_llama_kv_get_stats(adapter, &stats) == 0);
    assert(stats.appended_tokens == 0);
    assert(stats.sealed_pages == 0);

    assert(cf_llama_kv_seq_shift(adapter, 0, 0) == 0);
    assert(cf_llama_kv_seq_cp(adapter, 0, 1) != 0);
    assert(std::strstr(cf_llama_kv_last_error(adapter), "shared-page") != nullptr);
    assert(cf_llama_kv_clear(adapter) == 0);
    assert(cf_llama_kv_seq_rm(adapter, 0) == 0);

    cf_llama_kv_destroy(adapter);

    options.query_head_count = 7;
    assert(cf_llama_kv_options_validate(&options) != 0);

    std::cout << "llama.cpp ChromoFold KV adapter seam passed\n";
    return 0;
}
