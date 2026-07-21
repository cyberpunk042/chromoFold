#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cf_distribution_mode {
    CF_DISTRIBUTION_SINGLE_DEVICE = 0,
    CF_DISTRIBUTION_LAYER_PARALLEL = 1,
    CF_DISTRIBUTION_TENSOR_PARALLEL = 2,
} cf_distribution_mode;

typedef enum cf_transfer_mode {
    CF_TRANSFER_DISABLED = 0,
    CF_TRANSFER_PEER = 1,
    CF_TRANSFER_HOST_STAGED = 2,
} cf_transfer_mode;

typedef struct cf_device_descriptor {
    int32_t ordinal;
    uint64_t total_memory_bytes;
    uint64_t memory_budget_bytes;
    uint32_t compute_major;
    uint32_t compute_minor;
    uint32_t peer_group;
} cf_device_descriptor;

typedef struct cf_distributed_config {
    uint32_t mode;
    uint32_t device_count;
    const cf_device_descriptor * devices;
    uint32_t layer_count;
    uint32_t tensor_parallel_size;
    uint32_t allow_host_staging;
    uint32_t strict_peer_access;
    uint64_t routing_seed;
} cf_distributed_config;

typedef struct cf_sequence_placement {
    int64_t sequence_id;
    int32_t primary_device;
    uint32_t generation;
    uint32_t shard_count;
} cf_sequence_placement;

typedef struct cf_distributed_batch_item {
    int64_t sequence_id;
    uint32_t layer;
    uint32_t shard;
    int32_t source_device;
    int32_t execution_device;
    const void * query;
    void * output;
    uint32_t token_count;
} cf_distributed_batch_item;

typedef struct cf_distributed_counters {
    uint64_t devices_registered;
    uint64_t peer_links_enabled;
    uint64_t peer_links_rejected;
    uint64_t sequences_placed;
    uint64_t sequences_migrated;
    uint64_t batches_dispatched;
    uint64_t tensor_parallel_batches;
    uint64_t layer_parallel_batches;
    uint64_t peer_copies;
    uint64_t peer_bytes;
    uint64_t host_staged_copies;
    uint64_t host_staged_bytes;
    uint64_t routing_failures;
    uint64_t device_failures;
    uint64_t cross_device_contamination;
    uint64_t dense_fallback_launches;
    uint64_t page_refs_at_shutdown;
    uint64_t snapshot_refs_at_shutdown;
} cf_distributed_counters;

typedef struct cf_distributed_runtime cf_distributed_runtime;

cf_distributed_runtime * cf_distributed_create(const cf_distributed_config * config);
void cf_distributed_destroy(cf_distributed_runtime * runtime);
int cf_distributed_place_sequence(cf_distributed_runtime * runtime, int64_t sequence_id, cf_sequence_placement * out);
int cf_distributed_migrate_sequence(cf_distributed_runtime * runtime, int64_t sequence_id, int32_t destination_device, void * stream);
int cf_distributed_route_layer(const cf_distributed_runtime * runtime, uint32_t layer, int32_t * device_out);
int cf_distributed_enable_peer_links(cf_distributed_runtime * runtime);
int cf_distributed_execute_batch_async(cf_distributed_runtime * runtime, const cf_distributed_batch_item * items, uint32_t count, void * stream);
int cf_distributed_shutdown_async(cf_distributed_runtime * runtime, void * stream);
int cf_distributed_get_counters(const cf_distributed_runtime * runtime, cf_distributed_counters * out);
const char * cf_distributed_last_error(const cf_distributed_runtime * runtime);

#ifdef __cplusplus
}
#endif
