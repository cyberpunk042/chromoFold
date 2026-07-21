#include "chromofold/device_kv_dataplane.h"

#include <cassert>
#include <cstdint>
#include <iostream>

int main() {
    float fake = 0.0f;
    cf_device_tensor_view valid{};
    valid.data = &fake;
    valid.scalar_type = CF_DEVICE_F32;
    valid.token_count = 2;
    valid.head_count = 4;
    valid.head_dim = 8;
    valid.element_stride_bytes = sizeof(float);
    valid.head_stride_bytes = valid.head_dim * sizeof(float);
    valid.token_stride_bytes = valid.head_count * valid.head_stride_bytes;
    valid.device_ordinal = 0;
    assert(cf_device_tensor_validate(&valid) == 0);

    auto invalid = valid;
    invalid.scalar_type = 999;
    assert(cf_device_tensor_validate(&invalid) != 0);
    invalid = valid;
    invalid.element_stride_bytes = 1;
    assert(cf_device_tensor_validate(&invalid) != 0);
    invalid = valid;
    invalid.head_stride_bytes = 1;
    assert(cf_device_tensor_validate(&invalid) != 0);
    invalid = valid;
    invalid.token_stride_bytes = 1;
    assert(cf_device_tensor_validate(&invalid) != 0);

    cf_device_kv_config config{};
    config.layer_count = 2;
    config.kv_head_count = 4;
    config.query_head_count = 8;
    config.head_dim = 8;
    config.page_size = 64;
    config.gqa_group_size = 2;
    config.use_async_allocator = 1;
    auto * dataplane = cf_device_kv_create(&config);
    assert(dataplane != nullptr);
    cf_device_kv_counters counters{};
    assert(cf_device_kv_get_counters(dataplane, &counters) == 0);
    assert(counters.device_append_calls == 0);
    assert(counters.dense_fallback_launches == 0);
    cf_device_kv_destroy(dataplane, nullptr);

    config.query_head_count = 7;
    assert(cf_device_kv_create(&config) == nullptr);
    std::cout << "device KV dataplane contract valid\n";
    return 0;
}
