#ifndef CHROMOFOLD_LLAMA_SERVER_MULTISEQUENCE_BRIDGE_H
#define CHROMOFOLD_LLAMA_SERVER_MULTISEQUENCE_BRIDGE_H

#include <stdint.h>
#include "chromofold/multisequence_cache.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cf_llama_server_multisequence cf_llama_server_multisequence;

typedef struct cf_llama_server_multisequence_config {
    cf_multisequence_config cache;
    uint32_t strict_no_fallback;
    uint32_t enable_prefix_sharing;
    uint32_t enable_speculative_transactions;
} cf_llama_server_multisequence_config;

typedef struct cf_llama_server_slot_event {
    int64_t slot_id;
    int64_t sequence_id;
    uint32_t token_begin;
    uint32_t token_end;
} cf_llama_server_slot_event;

cf_llama_server_multisequence * cf_llama_server_multisequence_create(
    const cf_llama_server_multisequence_config * config);
void cf_llama_server_multisequence_destroy(cf_llama_server_multisequence * bridge, void * stream);

int cf_llama_server_slot_create(cf_llama_server_multisequence * bridge,
                                int64_t slot_id,
                                int64_t sequence_id);
int cf_llama_server_slot_reset_async(cf_llama_server_multisequence * bridge,
                                     int64_t slot_id,
                                     void * stream);
int cf_llama_server_sequence_copy_async(cf_llama_server_multisequence * bridge,
                                        int64_t source_sequence,
                                        int64_t destination_sequence,
                                        uint32_t token_begin,
                                        uint32_t token_end,
                                        void * stream);
int cf_llama_server_sequence_remove_async(cf_llama_server_multisequence * bridge,
                                          int64_t sequence_id,
                                          int64_t position_begin,
                                          int64_t position_end,
                                          void * stream);
int cf_llama_server_context_shift_async(cf_llama_server_multisequence * bridge,
                                        int64_t sequence_id,
                                        uint32_t position_begin,
                                        uint32_t position_end,
                                        int32_t delta,
                                        void * stream);
int cf_llama_server_cancel_sequence_async(cf_llama_server_multisequence * bridge,
                                          int64_t sequence_id,
                                          void * stream);
int cf_llama_server_batch_attention_async(cf_llama_server_multisequence * bridge,
                                          const cf_sequence_attention_item * items,
                                          uint32_t item_count,
                                          void * stream);
int cf_llama_server_audit(const cf_llama_server_multisequence * bridge);
const char * cf_llama_server_multisequence_last_error(const cf_llama_server_multisequence * bridge);

#ifdef __cplusplus
}
#endif

#endif
