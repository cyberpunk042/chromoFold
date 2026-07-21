#ifndef CHROMOFOLD_MULTISEQUENCE_CACHE_H
#define CHROMOFOLD_MULTISEQUENCE_CACHE_H

#include <stdint.h>
#include "chromofold/device_kv_dataplane.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cf_multisequence_cache cf_multisequence_cache;
typedef uint64_t cf_speculation_id;

typedef struct cf_multisequence_config {
    uint32_t layer_count;
    uint32_t kv_head_count;
    uint32_t query_head_count;
    uint32_t head_dim;
    uint32_t page_size;
    uint32_t gqa_group_size;
    uint32_t use_async_allocator;
    uint32_t enable_invariant_auditor;
} cf_multisequence_config;

typedef struct cf_multisequence_counters {
    uint64_t sequences_created;
    uint64_t sequences_copied;
    uint64_t sequences_removed;
    uint64_t sequences_shifted;
    uint64_t active_sequences;
    uint64_t active_sequences_peak;
    uint64_t shared_pages;
    uint64_t shared_page_references;
    uint64_t page_ref_increments;
    uint64_t page_ref_decrements;
    uint64_t pages_reclaimed;
    uint64_t pages_pending_reclamation;
    uint64_t pending_reclamation_bytes;
    uint64_t pending_reclamation_bytes_peak;
    uint64_t speculative_transactions_started;
    uint64_t speculative_transactions_committed;
    uint64_t speculative_transactions_rolled_back;
    uint64_t batches_dispatched;
    uint64_t sequences_per_batch_peak;
    uint64_t invariant_audits;
    uint64_t invariant_failures;
    uint64_t unsupported_operations;
    uint64_t dense_fallback_launches;
} cf_multisequence_counters;

typedef struct cf_sequence_attention_item {
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
} cf_sequence_attention_item;

cf_multisequence_cache * cf_multisequence_create(const cf_multisequence_config * config);
void cf_multisequence_destroy(cf_multisequence_cache * cache, void * stream);

int cf_sequence_create(cf_multisequence_cache * cache, int64_t sequence_id);
int cf_sequence_copy_async(cf_multisequence_cache * cache,
                           int64_t source_sequence,
                           int64_t destination_sequence,
                           uint32_t token_begin,
                           uint32_t token_end,
                           void * stream);
int cf_sequence_remove_async(cf_multisequence_cache * cache,
                             int64_t sequence_id,
                             int64_t position_begin,
                             int64_t position_end,
                             void * stream);
int cf_sequence_shift_async(cf_multisequence_cache * cache,
                            int64_t sequence_id,
                            uint32_t position_begin,
                            uint32_t position_end,
                            int32_t delta,
                            void * stream);
int cf_sequence_append_async(cf_multisequence_cache * cache,
                             int64_t sequence_id,
                             uint32_t layer,
                             uint32_t token_begin,
                             const cf_device_tensor_view * keys,
                             const cf_device_tensor_view * values,
                             void * stream);
int cf_multisequence_attention_async(cf_multisequence_cache * cache,
                                     const cf_sequence_attention_item * items,
                                     uint32_t item_count,
                                     void * stream);

cf_speculation_id cf_sequence_speculation_begin(cf_multisequence_cache * cache, int64_t sequence_id);
int cf_sequence_speculation_commit(cf_multisequence_cache * cache,
                                   int64_t sequence_id,
                                   cf_speculation_id speculation_id,
                                   uint32_t accepted_tokens,
                                   void * stream);
int cf_sequence_speculation_rollback(cf_multisequence_cache * cache,
                                     int64_t sequence_id,
                                     cf_speculation_id speculation_id,
                                     void * stream);

int cf_multisequence_reclaim_async(cf_multisequence_cache * cache, void * stream);
int cf_multisequence_audit(const cf_multisequence_cache * cache);
int cf_multisequence_get_counters(const cf_multisequence_cache * cache, cf_multisequence_counters * counters);
const char * cf_multisequence_last_error(const cf_multisequence_cache * cache);

#ifdef __cplusplus
}
#endif

#endif
