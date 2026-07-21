#include "chromofold/multisequence_cache.h"

#include <cassert>
#include <cstdint>
#include <iostream>

extern "C" int cf_device_tensor_validate(const cf_device_tensor_view * view) {
    return view && view->data && view->token_count && view->head_count && view->head_dim ? 0 : -1;
}

static cf_multisequence_config config() {
    cf_multisequence_config c{};
    c.layer_count = 2;
    c.kv_head_count = 2;
    c.query_head_count = 4;
    c.head_dim = 64;
    c.page_size = 64;
    c.gqa_group_size = 2;
    c.enable_invariant_auditor = 1;
    return c;
}

int main() {
    auto c = config();
    cf_multisequence_cache * cache = cf_multisequence_create(&c);
    assert(cache != nullptr);

    assert(cf_sequence_create(cache, 10) == 0);
    assert(cf_sequence_create(cache, 11) == 0);
    assert(cf_sequence_create(cache, 10) != 0);
    assert(cf_multisequence_audit(cache) == 0);

    const cf_speculation_id speculation = cf_sequence_speculation_begin(cache, 10);
    assert(speculation != 0);
    assert(cf_sequence_speculation_rollback(cache, 10, speculation, nullptr) == 0);

    assert(cf_sequence_shift_async(cache, 10, 0, 0, 0, nullptr) == 0);
    assert(cf_sequence_copy_async(cache, 10, 12, 0, 0, nullptr) == 0);
    assert(cf_multisequence_audit(cache) == 0);

    assert(cf_sequence_remove_async(cache, 11, -1, -1, nullptr) == 0);
    assert(cf_multisequence_reclaim_async(cache, nullptr) == 0);
    assert(cf_multisequence_audit(cache) == 0);

    cf_multisequence_counters counters{};
    assert(cf_multisequence_get_counters(cache, &counters) == 0);
    assert(counters.sequences_created == 2);
    assert(counters.sequences_copied == 1);
    assert(counters.speculative_transactions_started == 1);
    assert(counters.speculative_transactions_rolled_back == 1);
    assert(counters.dense_fallback_launches == 0);
    assert(counters.invariant_failures == 0);

    cf_multisequence_destroy(cache, nullptr);
    std::cout << "M10 multisequence contract passed\n";
    return 0;
}
