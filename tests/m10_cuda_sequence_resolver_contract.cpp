#include "chromofold/multisequence_cuda_resolver.h"
#include <cassert>
#include <iostream>

int main() {
    cf_ms_cuda_config config{};
    config.layer_count = 2;
    config.kv_head_count = 2;
    config.query_head_count = 4;
    config.head_dim = 64;
    config.page_size = 64;
    config.gqa_group_size = 2;
    auto * resolver = cf_ms_cuda_create(&config);
    assert(resolver != nullptr);
    assert(cf_ms_cuda_sequence_create(resolver, 101) == 0);
    assert(cf_ms_cuda_sequence_create(resolver, 202) == 0);
    assert(cf_ms_cuda_sequence_create(resolver, 101) != 0);
    assert(cf_ms_cuda_shift_sequence(resolver, 101, 0, 0, 0) == 0);
    assert(cf_ms_cuda_cancel_sequence_async(resolver, 202, nullptr) == 0);
    assert(cf_ms_cuda_audit(resolver) == 0);
    cf_ms_cuda_counters counters{};
    assert(cf_ms_cuda_get_counters(resolver, &counters) == 0);
    assert(counters.cancellation_requests == 1);
    assert(counters.dense_fallback_launches == 0);
    assert(counters.cross_sequence_contamination == 0);
    cf_ms_cuda_destroy(resolver, nullptr);
    std::cout << "M10 CUDA sequence resolver contract passed\n";
    return 0;
}
