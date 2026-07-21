#include "chromofold/distributed_runtime.h"

#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct cf_distributed_runtime {
    cf_distributed_config config{};
    std::vector<cf_device_descriptor> devices;
    std::unordered_map<int64_t, cf_sequence_placement> placements;
    cf_distributed_counters counters{};
    mutable std::mutex mutex;
    bool shutting_down = false;
    std::string error;
};

namespace {

int fail(cf_distributed_runtime * runtime, const char * message) {
    if (runtime) runtime->error = message ? message : "distributed runtime failure";
    return -1;
}

bool valid_config(const cf_distributed_config & config) {
    if (!config.device_count || !config.devices || !config.layer_count) return false;
    if (config.mode > CF_DISTRIBUTION_TENSOR_PARALLEL) return false;
    if (config.mode == CF_DISTRIBUTION_TENSOR_PARALLEL &&
        (!config.tensor_parallel_size || config.tensor_parallel_size > config.device_count)) return false;
    for (uint32_t index = 0; index < config.device_count; ++index) {
        if (config.devices[index].ordinal < 0 || !config.devices[index].memory_budget_bytes) return false;
    }
    return true;
}

uint64_t mix(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

} // namespace

extern "C" cf_distributed_runtime * cf_distributed_create(const cf_distributed_config * config) {
    if (!config || !valid_config(*config)) return nullptr;
    auto * runtime = new cf_distributed_runtime;
    runtime->config = *config;
    runtime->devices.assign(config->devices, config->devices + config->device_count);
    runtime->config.devices = runtime->devices.data();
    runtime->counters.devices_registered = config->device_count;
    return runtime;
}

extern "C" void cf_distributed_destroy(cf_distributed_runtime * runtime) {
    delete runtime;
}

extern "C" int cf_distributed_place_sequence(cf_distributed_runtime * runtime,
                                                int64_t sequence_id,
                                                cf_sequence_placement * out) {
    if (!runtime || !out) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (runtime->shutting_down) return fail(runtime, "runtime is shutting down");
    auto existing = runtime->placements.find(sequence_id);
    if (existing != runtime->placements.end()) {
        *out = existing->second;
        return 0;
    }

    const uint64_t hash = mix(static_cast<uint64_t>(sequence_id) ^ runtime->config.routing_seed);
    const uint32_t index = static_cast<uint32_t>(hash % runtime->devices.size());
    cf_sequence_placement placement{};
    placement.sequence_id = sequence_id;
    placement.primary_device = runtime->devices[index].ordinal;
    placement.generation = 1;
    placement.shard_count = runtime->config.mode == CF_DISTRIBUTION_TENSOR_PARALLEL
        ? runtime->config.tensor_parallel_size : 1;
    runtime->placements.emplace(sequence_id, placement);
    ++runtime->counters.sequences_placed;
    *out = placement;
    runtime->error.clear();
    return 0;
}

extern "C" int cf_distributed_migrate_sequence(cf_distributed_runtime * runtime,
                                                  int64_t sequence_id,
                                                  int32_t destination_device,
                                                  void *) {
    if (!runtime) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    auto placement = runtime->placements.find(sequence_id);
    if (placement == runtime->placements.end()) return fail(runtime, "sequence is not placed");
    const auto destination = std::find_if(runtime->devices.begin(), runtime->devices.end(),
        [destination_device](const cf_device_descriptor & device) { return device.ordinal == destination_device; });
    if (destination == runtime->devices.end()) return fail(runtime, "destination device is not registered");
    placement->second.primary_device = destination_device;
    ++placement->second.generation;
    ++runtime->counters.sequences_migrated;
    runtime->error.clear();
    return 0;
}

extern "C" int cf_distributed_route_layer(const cf_distributed_runtime * runtime,
                                             uint32_t layer,
                                             int32_t * device_out) {
    if (!runtime || !device_out || layer >= runtime->config.layer_count) return -1;
    const uint32_t index = runtime->config.mode == CF_DISTRIBUTION_SINGLE_DEVICE
        ? 0u : static_cast<uint32_t>((static_cast<uint64_t>(layer) * runtime->devices.size()) /
                                    runtime->config.layer_count);
    *device_out = runtime->devices[std::min(index, static_cast<uint32_t>(runtime->devices.size() - 1u))].ordinal;
    return 0;
}

extern "C" int cf_distributed_enable_peer_links(cf_distributed_runtime * runtime) {
    if (!runtime) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    for (size_t left = 0; left < runtime->devices.size(); ++left) {
        for (size_t right = left + 1; right < runtime->devices.size(); ++right) {
            if (runtime->devices[left].peer_group == runtime->devices[right].peer_group) {
                ++runtime->counters.peer_links_enabled;
            } else {
                ++runtime->counters.peer_links_rejected;
                if (runtime->config.strict_peer_access && !runtime->config.allow_host_staging)
                    return fail(runtime, "required peer link is unavailable");
            }
        }
    }
    runtime->error.clear();
    return 0;
}

extern "C" int cf_distributed_execute_batch_async(cf_distributed_runtime * runtime,
                                                     const cf_distributed_batch_item * items,
                                                     uint32_t count,
                                                     void *) {
    if (!runtime || !items || !count) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (runtime->shutting_down) return fail(runtime, "runtime is shutting down");
    for (uint32_t index = 0; index < count; ++index) {
        if (!items[index].query || !items[index].output || !items[index].token_count ||
            items[index].layer >= runtime->config.layer_count) {
            ++runtime->counters.routing_failures;
            return fail(runtime, "invalid distributed batch item");
        }
        if (runtime->placements.count(items[index].sequence_id) == 0) {
            ++runtime->counters.routing_failures;
            return fail(runtime, "batch references an unplaced sequence");
        }
        if (items[index].source_device != items[index].execution_device) {
            const auto source = std::find_if(runtime->devices.begin(), runtime->devices.end(), [&](const auto & device) {
                return device.ordinal == items[index].source_device;
            });
            const auto target = std::find_if(runtime->devices.begin(), runtime->devices.end(), [&](const auto & device) {
                return device.ordinal == items[index].execution_device;
            });
            if (source == runtime->devices.end() || target == runtime->devices.end())
                return fail(runtime, "batch references an unknown device");
            if (source->peer_group == target->peer_group) ++runtime->counters.peer_copies;
            else if (runtime->config.allow_host_staging) ++runtime->counters.host_staged_copies;
            else return fail(runtime, "cross-device transfer has no permitted transport");
        }
    }
    ++runtime->counters.batches_dispatched;
    if (runtime->config.mode == CF_DISTRIBUTION_TENSOR_PARALLEL) ++runtime->counters.tensor_parallel_batches;
    if (runtime->config.mode == CF_DISTRIBUTION_LAYER_PARALLEL) ++runtime->counters.layer_parallel_batches;
    runtime->error.clear();
    return 0;
}

extern "C" int cf_distributed_shutdown_async(cf_distributed_runtime * runtime, void *) {
    if (!runtime) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    runtime->shutting_down = true;
    runtime->placements.clear();
    runtime->counters.page_refs_at_shutdown = 0;
    runtime->counters.snapshot_refs_at_shutdown = 0;
    runtime->error.clear();
    return 0;
}

extern "C" int cf_distributed_get_counters(const cf_distributed_runtime * runtime,
                                              cf_distributed_counters * out) {
    if (!runtime || !out) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    *out = runtime->counters;
    return 0;
}

extern "C" const char * cf_distributed_last_error(const cf_distributed_runtime * runtime) {
    return runtime ? runtime->error.c_str() : "runtime is null";
}
