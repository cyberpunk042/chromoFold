#pragma once

#include <cstddef>
#include <cstdint>

#include "chromofold/device_kv_seal_attention.h"
#include "chromofold/multisequence_cache.h"
#include "chromofold/multisequence_cuda_resolver.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cf_llama_server_runtime cf_llama_server_runtime;

typedef struct cf_llama_server_config {
    uint32_t layer_count;
    uint32_t query_head_count;
    uint32_t kv_head_count;
    uint32_t head_dim;
    uint32_t page_size;
    uint32_t block_size;
    uint32_t context_window;
    uint32_t gqa_group_size;
    uint32_t cuda_device;
    uint32_t strict_mode;
    const char * evidence_path;
    const char * llama_commit;
    const char * model_sha256;
} cf_llama_server_config;

typedef struct cf_llama_server_batch_token {
    int64_t sequence_id;
    uint32_t batch_index;
    uint32_t absolute_position;
    uint32_t token_count;
} cf_llama_server_batch_token;

typedef struct cf_llama_server_batch {
    const cf_llama_server_batch_token * tokens;
    uint32_t token_item_count;
    const cf_multisequence_attention_item * attention_items;
    uint32_t attention_item_count;
} cf_llama_server_batch;

typedef struct cf_llama_server_stats {
    uint64_t http_requests;
    uint64_t interleaved_batches;
    uint64_t slot_reuses;
    uint64_t cancelled_requests;
    uint64_t context_shifts;
    uint64_t speculative_rollbacks;
    uint64_t dense_nodes_scheduled;
    uint64_t shutdown_audits;
    uint64_t shutdown_failures;
} cf_llama_server_stats;

cf_llama_server_runtime * cf_llama_server_runtime_create(const cf_llama_server_config * config);
void cf_llama_server_runtime_destroy(cf_llama_server_runtime * runtime, void * stream);

int cf_llama_server_slot_create(cf_llama_server_runtime * runtime, int64_t sequence_id);
int cf_llama_server_slot_reset_async(cf_llama_server_runtime * runtime, int64_t sequence_id, void * stream);
int cf_llama_server_cancel_async(cf_llama_server_runtime * runtime, int64_t sequence_id, void * stream);
int cf_llama_server_sequence_copy_async(cf_llama_server_runtime * runtime, int64_t source, int64_t destination,
                                        uint32_t token_begin, uint32_t token_end, void * stream);
int cf_llama_server_sequence_remove_async(cf_llama_server_runtime * runtime, int64_t sequence_id,
                                          int64_t position_begin, int64_t position_end, void * stream);
int cf_llama_server_sequence_shift_async(cf_llama_server_runtime * runtime, int64_t sequence_id,
                                         uint32_t position_begin, uint32_t position_end, int32_t delta, void * stream);

int cf_llama_server_append_layer_async(cf_llama_server_runtime * runtime,
                                       int64_t sequence_id,
                                       uint32_t layer_index,
                                       uint32_t token_begin,
                                       const float * key,
                                       const float * value,
                                       uint32_t token_count,
                                       void * stream);

int cf_llama_server_execute_batch_async(cf_llama_server_runtime * runtime,
                                        const cf_llama_server_batch * batch,
                                        void * stream);

int cf_llama_server_shutdown_async(cf_llama_server_runtime * runtime, void * stream);
int cf_llama_server_get_stats(const cf_llama_server_runtime * runtime, cf_llama_server_stats * out);
const char * cf_llama_server_last_error(const cf_llama_server_runtime * runtime);

#ifdef __cplusplus
}
#endif
