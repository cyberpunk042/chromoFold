#ifndef CHROMOFOLD_MULTISEQUENCE_CUDA_RESOLVER_H
#define CHROMOFOLD_MULTISEQUENCE_CUDA_RESOLVER_H

#include <stdint.h>
#include "chromofold/device_kv_dataplane.h"
#include "chromofold/device_kv_seal_attention.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cf_ms_cuda_config {
    uint32_t layer_count;
    uint32_t kv_head_count;
    uint32_t query_head_count;
    uint32_t head_dim;
    uint32_t page_size;
    uint32_t gqa_group_size;
    uint32_t use_async_allocator;
} cf_ms_cuda_config;

typedef struct cf_ms_page_publication {
    int64_t sequence_id;
    uint32_t layer;
    cf_device_page_descriptor descriptor;
    uint64_t resident_bytes;
    uint64_t seal_generation;
} cf_ms_page_publication;

typedef struct cf_ms_batch_item {
    int64_t sequence_id;
    uint32_t layer;
    cf_device_tensor_view query;
    void * output;
    uint32_t output_type;
    uint64_t output_token_stride_bytes;
    uint64_t output_head_stride_bytes;
    uint64_t output_element_stride_bytes;
    uint32_t query_token_begin;
    float softmax_scale;
    uint32_t causal_window;
} cf_ms_batch_item;

typedef struct cf_ms_cuda_counters {
    uint64_t pages_registered;
    uint64_t page_registration_failures;
    uint64_t descriptor_tables_built;
    uint64_t descriptor_table_rebuilds;
    uint64_t snapshots_acquired;
    uint64_t snapshots_released;
    uint64_t in_flight_page_references;
    uint64_t snapshot_release_events;
    uint64_t stale_snapshot_rejections;
    uint64_t server_batches_received;
    uint64_t compressed_batches_executed;
    uint64_t sequence_attention_launches;
    uint64_t prefill_items;
    uint64_t decode_items;
    uint64_t tokens_per_batch_peak;
    uint64_t copy_on_write_tail_copies;
    uint64_t cancellation_requests;
    uint64_t pages_retired;
    uint64_t pages_reclaimed;
    uint64_t pending_reclamation_bytes;
    uint64_t pending_reclamation_bytes_peak;
    uint64_t cross_sequence_contamination;
    uint64_t dense_fallback_launches;
    uint64_t cuda_errors;
} cf_ms_cuda_counters;

typedef struct cf_ms_cuda_resolver cf_ms_cuda_resolver;

cf_ms_cuda_resolver * cf_ms_cuda_create(const cf_ms_cuda_config * config);
void cf_ms_cuda_destroy(cf_ms_cuda_resolver * resolver, void * stream);
int cf_ms_cuda_sequence_create(cf_ms_cuda_resolver * resolver, int64_t sequence_id);
int cf_ms_cuda_sequence_remove_async(cf_ms_cuda_resolver * resolver, int64_t sequence_id, void * stream);
int cf_ms_cuda_publish_page_async(cf_ms_cuda_resolver * resolver, const cf_ms_page_publication * publication, void * stream);
int cf_ms_cuda_stage_active_async(cf_ms_cuda_resolver * resolver, int64_t sequence_id, uint32_t layer,
                                  uint32_t token_begin, const float * keys, const float * values,
                                  uint32_t token_count, void * stream);
int cf_ms_cuda_copy_sequence_async(cf_ms_cuda_resolver * resolver, int64_t source_sequence,
                                   int64_t destination_sequence, void * stream);
int cf_ms_cuda_shift_sequence(cf_ms_cuda_resolver * resolver, int64_t sequence_id,
                              uint32_t position_begin, uint32_t position_end, int32_t delta);
int cf_ms_cuda_execute_batch_async(cf_ms_cuda_resolver * resolver,
                                   const cf_ms_batch_item * items,
                                   uint32_t item_count,
                                   void * stream);
int cf_ms_cuda_cancel_sequence_async(cf_ms_cuda_resolver * resolver, int64_t sequence_id, void * stream);
int cf_ms_cuda_reclaim_async(cf_ms_cuda_resolver * resolver, void * stream);
int cf_ms_cuda_audit(const cf_ms_cuda_resolver * resolver);
int cf_ms_cuda_get_counters(const cf_ms_cuda_resolver * resolver, cf_ms_cuda_counters * counters);
const char * cf_ms_cuda_last_error(const cf_ms_cuda_resolver * resolver);

#ifdef __cplusplus
}
#endif

#endif
