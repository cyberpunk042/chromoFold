#include "llama_server_cuda_resolver_bridge.h"

#include <memory>
#include <string>

struct cf_llama_server_cuda_bridge {
    cf_ms_cuda_resolver * resolver = nullptr;
    std::string error;
};

extern "C" cf_llama_server_cuda_bridge * cf_llama_server_cuda_bridge_create(const cf_ms_cuda_config * config) {
    auto bridge = std::make_unique<cf_llama_server_cuda_bridge>();
    bridge->resolver = cf_ms_cuda_create(config);
    if (!bridge->resolver) return nullptr;
    return bridge.release();
}

extern "C" void cf_llama_server_cuda_bridge_destroy(cf_llama_server_cuda_bridge * bridge, void * stream) {
    if (!bridge) return;
    cf_ms_cuda_destroy(bridge->resolver, stream);
    delete bridge;
}

extern "C" int cf_llama_server_cuda_slot_create(cf_llama_server_cuda_bridge * bridge, int64_t slot_id) {
    if (!bridge) return -1;
    const int result = cf_ms_cuda_sequence_create(bridge->resolver, slot_id);
    if (result != 0) bridge->error = cf_ms_cuda_last_error(bridge->resolver);
    return result;
}

extern "C" int cf_llama_server_cuda_slot_copy_async(cf_llama_server_cuda_bridge * bridge, int64_t source_slot, int64_t destination_slot, void * stream) {
    if (!bridge) return -1;
    const int result = cf_ms_cuda_copy_sequence_async(bridge->resolver, source_slot, destination_slot, stream);
    if (result != 0) bridge->error = cf_ms_cuda_last_error(bridge->resolver);
    return result;
}

extern "C" int cf_llama_server_cuda_slot_cancel_async(cf_llama_server_cuda_bridge * bridge, int64_t slot_id, void * stream) {
    if (!bridge) return -1;
    const int result = cf_ms_cuda_cancel_sequence_async(bridge->resolver, slot_id, stream);
    if (result != 0) bridge->error = cf_ms_cuda_last_error(bridge->resolver);
    return result;
}

extern "C" int cf_llama_server_cuda_slot_reset_async(cf_llama_server_cuda_bridge * bridge, int64_t slot_id, void * stream) {
    if (!bridge) return -1;
    const int result = cf_ms_cuda_sequence_remove_async(bridge->resolver, slot_id, stream);
    if (result != 0) bridge->error = cf_ms_cuda_last_error(bridge->resolver);
    return result;
}

extern "C" int cf_llama_server_cuda_dispatch_batch_async(cf_llama_server_cuda_bridge * bridge, const cf_ms_batch_item * items, uint32_t item_count, void * stream) {
    if (!bridge) return -1;
    const int result = cf_ms_cuda_execute_batch_async(bridge->resolver, items, item_count, stream);
    if (result != 0) bridge->error = cf_ms_cuda_last_error(bridge->resolver);
    return result;
}

extern "C" int cf_llama_server_cuda_shutdown_audit(cf_llama_server_cuda_bridge * bridge) {
    if (!bridge) return -1;
    const int result = cf_ms_cuda_audit(bridge->resolver);
    if (result != 0) bridge->error = "CUDA multi-sequence shutdown audit failed";
    return result;
}

extern "C" const char * cf_llama_server_cuda_last_error(const cf_llama_server_cuda_bridge * bridge) {
    return bridge ? bridge->error.c_str() : "null llama-server CUDA bridge";
}
