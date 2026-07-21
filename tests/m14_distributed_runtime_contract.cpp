#include "chromofold/distributed_runtime.h"

#include <cassert>
#include <cstdint>

int main() {
    const cf_device_descriptor devices[] = {
        {0, 24ull << 30, 20ull << 30, 8, 9, 1},
        {1, 24ull << 30, 20ull << 30, 8, 9, 1},
    };
    const cf_distributed_config config{
        CF_DISTRIBUTION_TENSOR_PARALLEL, 2, devices, 32, 2, 0, 1, 42,
    };
    cf_distributed_runtime * runtime = cf_distributed_create(&config);
    assert(runtime != nullptr);
    assert(cf_distributed_enable_peer_links(runtime) == 0);

    cf_sequence_placement placement{};
    assert(cf_distributed_place_sequence(runtime, 1001, &placement) == 0);
    assert(placement.shard_count == 2);

    int32_t layer_device = -1;
    assert(cf_distributed_route_layer(runtime, 31, &layer_device) == 0);
    assert(layer_device == 1);

    float query = 1.0f;
    float output = 0.0f;
    const cf_distributed_batch_item item{1001, 3, 0, 0, 1, &query, &output, 1};
    assert(cf_distributed_execute_batch_async(runtime, &item, 1, nullptr) == 0);
    assert(cf_distributed_migrate_sequence(runtime, 1001, 1, nullptr) == 0);
    assert(cf_distributed_shutdown_async(runtime, nullptr) == 0);

    cf_distributed_counters counters{};
    assert(cf_distributed_get_counters(runtime, &counters) == 0);
    assert(counters.devices_registered == 2);
    assert(counters.peer_links_enabled == 1);
    assert(counters.tensor_parallel_batches == 1);
    assert(counters.page_refs_at_shutdown == 0);
    assert(counters.snapshot_refs_at_shutdown == 0);
    assert(counters.dense_fallback_launches == 0);
    cf_distributed_destroy(runtime);
    return 0;
}
