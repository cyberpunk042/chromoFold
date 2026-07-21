#ifndef CHROMOFOLD_LLAMA_SERVER_CUDA_RESOLVER_BRIDGE_H
#define CHROMOFOLD_LLAMA_SERVER_CUDA_RESOLVER_BRIDGE_H

#include "chromofold/multisequence_cuda_resolver.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cf_llama_server_cuda_bridge cf_llama_server_cuda_bridge;

cf_llama_server_cuda_bridge * cf_llama_server_cuda_bridge_create(const cf_ms_cuda_config * config);
void cf_llama_server_cuda_bridge_destroy(cf_llama_server_cuda_bridge * bridge, void * stream);
int cf_llama_server_cuda_slot_create(cf_llama_server_cuda_bridge * bridge, int64_t slot_id);
int cf_llama_server_cuda_slot_copy_async(cf_llama_server_cuda_bridge * bridge, int64_t source_slot, int64_t destination_slot, void * stream);
int cf_llama_server_cuda_slot_cancel_async(cf_llama_server_cuda_bridge * bridge, int64_t slot_id, void * stream);
int cf_llama_server_cuda_slot_reset_async(cf_llama_server_cuda_bridge * bridge, int64_t slot_id, void * stream);
int cf_llama_server_cuda_dispatch_batch_async(cf_llama_server_cuda_bridge * bridge, const cf_ms_batch_item * items, uint32_t item_count, void * stream);
int cf_llama_server_cuda_shutdown_audit(cf_llama_server_cuda_bridge * bridge);
const char * cf_llama_server_cuda_last_error(const cf_llama_server_cuda_bridge * bridge);

#ifdef __cplusplus
}
#endif

#endif
