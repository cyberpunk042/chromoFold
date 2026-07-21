#include "chromofold/multisequence_cache.h"
#include <cassert>
#include <iostream>

extern "C" int cf_device_tensor_validate(const cf_device_tensor_view * view) {
    return view && view->data && view->token_count && view->head_count && view->head_dim ? 0 : -1;
}

int main() {
    cf_multisequence_config config{};
    config.layer_count = 2;
    config.kv_head_count = 2;
    config.query_head_count = 4;
    config.head_dim = 64;
    config.page_size = 64;
    config.gqa_group_size = 2;
    config.enable_invariant_auditor = 1;

    cf_multisequence_cache * cache = cf_multisequence_create(&config);
    assert(cache);
    assert(cf_sequence_create(cache, 10) == 0);
    assert(cf_sequence_create(cache, 11) == 0);
    assert(cf_sequence_create(cache, 10) != 0);

    const cf_speculation_id transaction = cf_sequence_speculation_begin(cache, 10);
    assert(transaction != 0);
    assert(cf_sequence_speculation_rollback(cache, 10, transaction, nullptr) == 0);
    assert(cf_sequence_copy_async(cache, 10, 12, 0, 0, nullptr) == 0);
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

    std::cout << "M10 multisequence host contract passed\n";
    return 0;
}
