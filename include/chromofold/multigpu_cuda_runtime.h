#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cf_multigpu_transport {
    CF_MULTIGPU_TRANSPORT_P2P = 1,
    CF_MULTIGPU_TRANSPORT_HOST_STAGED = 2,
} cf_multigpu_transport;

typedef enum cf_multigpu_collective {
    CF_MULTIGPU_COLLECTIVE_NONE = 0,
    CF_MULTIGPU_COLLECTIVE_NCCL_ALL_REDUCE = 1,
    CF_MULTIGPU_COLLECTIVE_HOST_REDUCE = 2,
} cf_multigpu_collective;

typedef struct cf_multigpu_device_config {
    int32_t device_ordinal;
    uint64_t memory_budget_bytes;
    uint32_t peer_group;
} cf_multigpu_device_config;

typedef struct cf_multigpu_runtime_config {
    const cf_multigpu_device_config * devices;
    uint32_t device_count;
    uint32_t layer_count;
    uint32_t tensor_parallel_size;
    uint32_t allow_host_staging;
    uint32_t require_peer_access;
    uint32_t collective;
} cf_multigpu_runtime_config;

typedef struct cf_multigpu_page_view {
    const void * key_payload;
    const void * value_payload;
    const void * key_scales;
    const void * value_scales;
    uint64_t key_payload_bytes;
    uint64_t value_payload_bytes;
    uint32_t key_codec;
    uint32_t value_codec;
    uint32_t token_count;
    uint32_t kv_head_count;
    uint32_t head_dim;
    uint32_t block_size;
    int32_t source_device;
    uint64_t generation;
} cf_multigpu_page_view;

typedef struct cf_multigpu_attention_shard {
    uint64_t sequence_id;
    uint32_t layer;
    uint32_t query_head_begin;
    uint32_t query_head_count;
    int32_t execution_device;
    const void * query;
    void * output;
    uint32_t query_type;
    uint32_t output_type;
    uint32_t head_dim;
    uint32_t page_count;
    const cf_multigpu_page_view * pages;
} cf_multigpu_attention_shard;

typedef struct cf_multigpu_counters {
    uint64_t batches_submitted;
    uint64_t shards_dispatched;
    uint64_t p2p_copy_calls;
    uint64_t p2p_bytes;
    uint64_t host_staged_copy_calls;
    uint64_t host_staged_bytes;
    uint64_t collective_calls;
    uint64_t collective_bytes;
    uint64_t mixed_codec_pages_consumed;
    uint64_t remote_pages_loaded;
    uint64_t migrations_completed;
    uint64_t device_failures_detected;
    uint64_t fail_closed_rejections;
    uint64_t cross_device_contamination;
    uint64_t dense_fallback_launches;
    uint64_t cuda_errors;
    uint64_t transfer_errors;
    uint64_t page_refs_at_shutdown;
    uint64_t snapshot_refs_at_shutdown;
} cf_multigpu_counters;

typedef struct cf_multigpu_cuda_runtime cf_multigpu_cuda_runtime;

cf_multigpu_cuda_runtime * cf_multigpu_cuda_create(const cf_multigpu_runtime_config * config);
void cf_multigpu_cuda_destroy(cf_multigpu_cuda_runtime * runtime);
int cf_multigpu_cuda_validate_topology(cf_multigpu_cuda_runtime * runtime);
int cf_multigpu_cuda_execute_async(cf_multigpu_cuda_runtime * runtime,
                                   const cf_multigpu_attention_shard * shards,
                                   uint32_t shard_count,
                                   void * stream);
int cf_multigpu_cuda_migrate_sequence_async(cf_multigpu_cuda_runtime * runtime,
                                            uint64_t sequence_id,
                                            int32_t source_device,
                                            int32_t destination_device,
                                            const cf_multigpu_page_view * pages,
                                            uint32_t page_count,
                                            void * stream);
int cf_multigpu_cuda_inject_device_failure(cf_multigpu_cuda_runtime * runtime,
                                           int32_t device_ordinal);
int cf_multigpu_cuda_shutdown(cf_multigpu_cuda_runtime * runtime);
int cf_multigpu_cuda_get_counters(const cf_multigpu_cuda_runtime * runtime,
                                  cf_multigpu_counters * counters);
const char * cf_multigpu_cuda_last_error(const cf_multigpu_cuda_runtime * runtime);

#ifdef __cplusplus
}
#endif
