#include "chromofold/production_attention.h"

#include <cassert>
#include <cstdint>
#include <iostream>

int main() {
    cf_attention_runtime_config config{};
    config.layer_count = 2;
    config.query_head_count = 4;
    config.kv_head_count = 2;
    config.head_dim = 64;
    config.page_size = 64;
    config.block_size = 32;
    config.gqa_group_size = 2;
    config.cuda_device = 0;
    config.use_async_allocator = 1;
    config.mode = CF_ATTENTION_OPTIMIZED;
    config.tuning.threads_per_block = 128;
    config.tuning.warps_per_block = 4;
    config.tuning.heads_per_block = 4;
    config.tuning.pages_per_work_unit = 2;
    config.tuning.vector_width = 4;
    config.tuning.enable_cuda_graphs = 1;
    config.tuning.enable_async_sealing = 1;

    cf_attention_runtime * runtime = cf_attention_runtime_create(&config);
    assert(runtime != nullptr);

    cf_attention_counters counters{};
    assert(cf_attention_get_counters(runtime, &counters) == 0);
    assert(counters.reference_launches == 0);
    assert(counters.optimized_launches == 0);
    assert(counters.dense_fallback_launches == 0);
    assert(counters.cuda_errors == 0);

    assert(cf_attention_invalidate_graphs(runtime) == 0);
    assert(cf_attention_get_counters(runtime, &counters) == 0);
    assert(counters.graph_invalidations == 1);

    cf_attention_runtime_destroy(runtime, nullptr);
    std::cout << "M12 production attention contract passed\n";
    return 0;
}
