#include "chromofold/multigpu_cuda_runtime.h"

#include <cassert>
#include <cstdint>

int main() {
    cf_multigpu_device_config devices[] = {
        {0, 1ull << 30, 1},
        {1, 1ull << 30, 1},
    };
    cf_multigpu_runtime_config config{};
    config.devices = devices;
    config.device_count = 2;
    config.layer_count = 32;
    config.tensor_parallel_size = 2;
    config.allow_host_staging = 1;
    config.require_peer_access = 0;
    config.collective = CF_MULTIGPU_COLLECTIVE_NCCL_ALL_REDUCE;

    cf_multigpu_cuda_runtime * runtime = cf_multigpu_cuda_create(&config);
    assert(runtime != nullptr);

    cf_multigpu_counters counters{};
    assert(cf_multigpu_cuda_get_counters(runtime, &counters) == 0);
    assert(counters.dense_fallback_launches == 0);
    assert(counters.cross_device_contamination == 0);

    assert(cf_multigpu_cuda_inject_device_failure(runtime, 1) == 0);
    assert(cf_multigpu_cuda_get_counters(runtime, &counters) == 0);
    assert(counters.device_failures_detected == 1);

    assert(cf_multigpu_cuda_shutdown(runtime) == 0);
    cf_multigpu_cuda_destroy(runtime);
    return 0;
}
