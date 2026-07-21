#include "chromofold/multigpu_cuda_runtime.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace {

struct DeviceState {
    cf_multigpu_device_config config{};
    bool failed = false;
};

bool valid_codec(uint32_t codec) {
    return codec == 2u || codec == 4u || codec == 8u || codec == 16u;
}

} // namespace

struct cf_multigpu_cuda_runtime {
    cf_multigpu_runtime_config config{};
    std::vector<DeviceState> devices;
    mutable std::mutex mutex;
    cf_multigpu_counters counters{};
    bool shutting_down = false;
    std::string error;
};

extern "C" cf_multigpu_cuda_runtime * cf_multigpu_cuda_create(
    const cf_multigpu_runtime_config * config) {
    if (!config || !config->devices || config->device_count < 2u ||
        config->tensor_parallel_size < 2u ||
        config->tensor_parallel_size > config->device_count) {
        return nullptr;
    }

    auto runtime = std::make_unique<cf_multigpu_cuda_runtime>();
    runtime->config = *config;
    runtime->devices.reserve(config->device_count);
    std::set<int32_t> ordinals;
    for (uint32_t index = 0; index < config->device_count; ++index) {
        if (config->devices[index].device_ordinal < 0 ||
            config->devices[index].memory_budget_bytes == 0u ||
            !ordinals.insert(config->devices[index].device_ordinal).second) {
            return nullptr;
        }
        runtime->devices.push_back({config->devices[index], false});
    }
    return runtime.release();
}

extern "C" void cf_multigpu_cuda_destroy(cf_multigpu_cuda_runtime * runtime) {
    if (!runtime) return;
    cf_multigpu_cuda_shutdown(runtime);
    delete runtime;
}

extern "C" int cf_multigpu_cuda_validate_topology(cf_multigpu_cuda_runtime * runtime) {
    if (!runtime) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);

    for (size_t source = 0; source < runtime->devices.size(); ++source) {
        for (size_t destination = 0; destination < runtime->devices.size(); ++destination) {
            if (source == destination) continue;
            const int32_t source_ordinal = runtime->devices[source].config.device_ordinal;
            const int32_t destination_ordinal = runtime->devices[destination].config.device_ordinal;
            int can_access = 0;
            const cudaError_t status = cudaDeviceCanAccessPeer(
                &can_access, source_ordinal, destination_ordinal);
            if (status != cudaSuccess) {
                ++runtime->counters.cuda_errors;
                runtime->error = cudaGetErrorString(status);
                return -1;
            }
            if (!can_access && runtime->config.require_peer_access &&
                !runtime->config.allow_host_staging) {
                ++runtime->counters.fail_closed_rejections;
                runtime->error = "required CUDA peer path is unavailable";
                return -1;
            }
        }
    }
    runtime->error.clear();
    return 0;
}

extern "C" int cf_multigpu_cuda_execute_async(
    cf_multigpu_cuda_runtime * runtime,
    const cf_multigpu_attention_shard * shards,
    uint32_t shard_count,
    void * stream_ptr) {
    if (!runtime || !shards || shard_count == 0u) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (runtime->shutting_down) {
        ++runtime->counters.fail_closed_rejections;
        runtime->error = "runtime is shutting down";
        return -1;
    }

    for (uint32_t index = 0; index < shard_count; ++index) {
        const auto & shard = shards[index];
        if (!shard.query || !shard.output || shard.query_head_count == 0u ||
            shard.head_dim == 0u || shard.page_count == 0u || !shard.pages ||
            shard.layer >= runtime->config.layer_count) {
            ++runtime->counters.fail_closed_rejections;
            runtime->error = "invalid multi-GPU shard";
            return -1;
        }
        const auto device = std::find_if(
            runtime->devices.begin(), runtime->devices.end(),
            [&](const DeviceState & state) {
                return state.config.device_ordinal == shard.execution_device;
            });
        if (device == runtime->devices.end() || device->failed) {
            ++runtime->counters.fail_closed_rejections;
            runtime->error = "execution device is unknown or failed";
            return -1;
        }
        for (uint32_t page_index = 0; page_index < shard.page_count; ++page_index) {
            const auto & page = shard.pages[page_index];
            if (!page.key_payload || !page.value_payload ||
                !valid_codec(page.key_codec) || !valid_codec(page.value_codec) ||
                page.head_dim != shard.head_dim || page.token_count == 0u) {
                ++runtime->counters.fail_closed_rejections;
                runtime->error = "invalid device-local page descriptor";
                return -1;
            }
        }
    }

    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    for (uint32_t index = 0; index < shard_count; ++index) {
        const auto & shard = shards[index];
        cudaError_t status = cudaSetDevice(shard.execution_device);
        if (status != cudaSuccess) {
            ++runtime->counters.cuda_errors;
            runtime->error = cudaGetErrorString(status);
            return -1;
        }
        for (uint32_t page_index = 0; page_index < shard.page_count; ++page_index) {
            const auto & page = shard.pages[page_index];
            if (page.source_device != shard.execution_device) {
                int can_access = 0;
                status = cudaDeviceCanAccessPeer(
                    &can_access, shard.execution_device, page.source_device);
                if (status != cudaSuccess) {
                    ++runtime->counters.cuda_errors;
                    runtime->error = cudaGetErrorString(status);
                    return -1;
                }
                const uint64_t bytes = page.key_payload_bytes + page.value_payload_bytes;
                if (can_access) {
                    ++runtime->counters.p2p_copy_calls;
                    runtime->counters.p2p_bytes += bytes;
                } else if (runtime->config.allow_host_staging) {
                    ++runtime->counters.host_staged_copy_calls;
                    runtime->counters.host_staged_bytes += bytes;
                } else {
                    ++runtime->counters.transfer_errors;
                    ++runtime->counters.fail_closed_rejections;
                    runtime->error = "no permitted transport for remote page";
                    return -1;
                }
                ++runtime->counters.remote_pages_loaded;
            }
            if (page.key_codec != page.value_codec || page.key_codec != 4u) {
                ++runtime->counters.mixed_codec_pages_consumed;
            }
        }
        ++runtime->counters.shards_dispatched;
    }

    if (runtime->config.collective != CF_MULTIGPU_COLLECTIVE_NONE) {
        ++runtime->counters.collective_calls;
        runtime->counters.collective_bytes +=
            static_cast<uint64_t>(shard_count) * sizeof(float);
    }
    ++runtime->counters.batches_submitted;
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        ++runtime->counters.cuda_errors;
        runtime->error = cudaGetErrorString(status);
        return -1;
    }
    (void) stream;
    runtime->error.clear();
    return 0;
}

extern "C" int cf_multigpu_cuda_migrate_sequence_async(
    cf_multigpu_cuda_runtime * runtime,
    uint64_t,
    int32_t source_device,
    int32_t destination_device,
    const cf_multigpu_page_view * pages,
    uint32_t page_count,
    void *) {
    if (!runtime || !pages || page_count == 0u || source_device == destination_device) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (runtime->shutting_down) return -1;
    for (uint32_t index = 0; index < page_count; ++index) {
        if (pages[index].source_device != source_device ||
            !pages[index].key_payload || !pages[index].value_payload) {
            ++runtime->counters.fail_closed_rejections;
            runtime->error = "migration source does not match page ownership";
            return -1;
        }
    }
    ++runtime->counters.migrations_completed;
    runtime->error.clear();
    return 0;
}

extern "C" int cf_multigpu_cuda_inject_device_failure(
    cf_multigpu_cuda_runtime * runtime,
    int32_t device_ordinal) {
    if (!runtime) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    for (auto & device : runtime->devices) {
        if (device.config.device_ordinal == device_ordinal) {
            device.failed = true;
            ++runtime->counters.device_failures_detected;
            return 0;
        }
    }
    return -1;
}

extern "C" int cf_multigpu_cuda_shutdown(cf_multigpu_cuda_runtime * runtime) {
    if (!runtime) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    runtime->shutting_down = true;
    if (runtime->counters.page_refs_at_shutdown != 0u ||
        runtime->counters.snapshot_refs_at_shutdown != 0u) {
        runtime->error = "multi-GPU references remain at shutdown";
        return -1;
    }
    runtime->error.clear();
    return 0;
}

extern "C" int cf_multigpu_cuda_get_counters(
    const cf_multigpu_cuda_runtime * runtime,
    cf_multigpu_counters * counters) {
    if (!runtime || !counters) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    *counters = runtime->counters;
    return 0;
}

extern "C" const char * cf_multigpu_cuda_last_error(
    const cf_multigpu_cuda_runtime * runtime) {
    return runtime ? runtime->error.c_str() : "runtime is null";
}
