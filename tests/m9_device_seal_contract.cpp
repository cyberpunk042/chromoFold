#include "chromofold/device_kv_seal_attention.h"

#include <cassert>
#include <cstdint>

int main() {
    cf_device_seal_config config{};
    config.layer_count = 2;
    config.kv_head_count = 4;
    config.query_head_count = 8;
    config.head_dim = 64;
    config.page_size = 64;
    config.gqa_group_size = 2;
    config.block_size = 32;
    config.use_async_allocator = 1;

    auto * engine = cf_device_seal_create(&config);
    assert(engine != nullptr);

    uint32_t state = 99;
    assert(cf_device_seal_get_state(engine, 0, &state) == 0);
    assert(state == CF_SEAL_ACTIVE_FILLING);

    for (uint32_t fault = CF_SEAL_FAULT_NONE; fault <= CF_SEAL_FAULT_ACTIVE_REPLACEMENT; ++fault)
        assert(cf_device_seal_set_fault(engine, fault) == 0);
    assert(cf_device_seal_set_fault(engine, CF_SEAL_FAULT_ACTIVE_REPLACEMENT + 1) != 0);

    cf_device_seal_counters counters{};
    assert(cf_device_seal_get_counters(engine, &counters) == 0);
    assert(counters.seal_attempts == 0);
    assert(counters.seal_successes == 0);
    assert(counters.descriptor_publications == 0);
    assert(counters.compressed_attention_launches == 0);

    cf_device_seal_destroy(engine, nullptr);

    cf_device_seal_config invalid = config;
    invalid.query_head_count = 7;
    assert(cf_device_seal_create(&invalid) == nullptr);

    invalid = config;
    invalid.block_size = 0;
    assert(cf_device_seal_create(&invalid) == nullptr);
    return 0;
}
