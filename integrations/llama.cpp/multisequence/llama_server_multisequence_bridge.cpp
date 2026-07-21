#include "llama_server_multisequence_bridge.h"

#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

struct cf_llama_server_multisequence {
    cf_llama_server_multisequence_config config{};
    cf_multisequence_cache * cache = nullptr;
    std::map<int64_t, int64_t> slots;
    mutable std::mutex mutex;
    std::string last_error;
};

namespace {

void set_error(cf_llama_server_multisequence * bridge, const std::string & message) {
    bridge->last_error = message;
}

int remove_sequence(cf_llama_server_multisequence * bridge, int64_t sequence_id, void * stream) {
    return cf_sequence_remove_async(bridge->cache, sequence_id, -1, -1, stream);
}

} // namespace

extern "C" cf_llama_server_multisequence * cf_llama_server_multisequence_create(
    const cf_llama_server_multisequence_config * config) {
    if (!config || config->strict_no_fallback == 0) return nullptr;
    try {
        auto bridge = std::make_unique<cf_llama_server_multisequence>();
        bridge->config = *config;
        bridge->cache = cf_multisequence_create(&config->cache);
        if (!bridge->cache) throw std::runtime_error("failed to create multisequence cache");
        return bridge.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void cf_llama_server_multisequence_destroy(cf_llama_server_multisequence * bridge, void * stream) {
    if (!bridge) return;
    std::lock_guard<std::mutex> guard(bridge->mutex);
    cf_multisequence_destroy(bridge->cache, stream);
    bridge->cache = nullptr;
    bridge->slots.clear();
    delete bridge;
}

extern "C" int cf_llama_server_slot_create(cf_llama_server_multisequence * bridge,
                                              int64_t slot_id,
                                              int64_t sequence_id) {
    if (!bridge) return -1;
    std::lock_guard<std::mutex> guard(bridge->mutex);
    try {
        if (bridge->slots.count(slot_id)) throw std::runtime_error("slot already exists");
        if (cf_sequence_create(bridge->cache, sequence_id) != 0)
            throw std::runtime_error(cf_multisequence_last_error(bridge->cache));
        bridge->slots.emplace(slot_id, sequence_id);
        bridge->last_error.clear();
        return 0;
    } catch (const std::exception & error) {
        set_error(bridge, error.what());
        return -1;
    }
}

extern "C" int cf_llama_server_slot_reset_async(cf_llama_server_multisequence * bridge,
                                                   int64_t slot_id,
                                                   void * stream) {
    if (!bridge) return -1;
    std::lock_guard<std::mutex> guard(bridge->mutex);
    try {
        auto slot = bridge->slots.find(slot_id);
        if (slot == bridge->slots.end()) throw std::runtime_error("slot does not exist");
        if (remove_sequence(bridge, slot->second, stream) != 0)
            throw std::runtime_error(cf_multisequence_last_error(bridge->cache));
        bridge->slots.erase(slot);
        if (cf_multisequence_reclaim_async(bridge->cache, stream) != 0)
            throw std::runtime_error(cf_multisequence_last_error(bridge->cache));
        bridge->last_error.clear();
        return 0;
    } catch (const std::exception & error) {
        set_error(bridge, error.what());
        return -1;
    }
}

extern "C" int cf_llama_server_sequence_copy_async(cf_llama_server_multisequence * bridge,
                                                      int64_t source_sequence,
                                                      int64_t destination_sequence,
                                                      uint32_t token_begin,
                                                      uint32_t token_end,
                                                      void * stream) {
    if (!bridge) return -1;
    std::lock_guard<std::mutex> guard(bridge->mutex);
    const int status = cf_sequence_copy_async(
        bridge->cache, source_sequence, destination_sequence, token_begin, token_end, stream);
    if (status != 0) set_error(bridge, cf_multisequence_last_error(bridge->cache));
    else bridge->last_error.clear();
    return status;
}

extern "C" int cf_llama_server_sequence_remove_async(cf_llama_server_multisequence * bridge,
                                                        int64_t sequence_id,
                                                        int64_t position_begin,
                                                        int64_t position_end,
                                                        void * stream) {
    if (!bridge) return -1;
    std::lock_guard<std::mutex> guard(bridge->mutex);
    const int status = cf_sequence_remove_async(
        bridge->cache, sequence_id, position_begin, position_end, stream);
    if (status != 0) set_error(bridge, cf_multisequence_last_error(bridge->cache));
    else bridge->last_error.clear();
    return status;
}

extern "C" int cf_llama_server_context_shift_async(cf_llama_server_multisequence * bridge,
                                                      int64_t sequence_id,
                                                      uint32_t position_begin,
                                                      uint32_t position_end,
                                                      int32_t delta,
                                                      void * stream) {
    if (!bridge) return -1;
    std::lock_guard<std::mutex> guard(bridge->mutex);
    const int status = cf_sequence_shift_async(
        bridge->cache, sequence_id, position_begin, position_end, delta, stream);
    if (status != 0) set_error(bridge, cf_multisequence_last_error(bridge->cache));
    else bridge->last_error.clear();
    return status;
}

extern "C" int cf_llama_server_cancel_sequence_async(cf_llama_server_multisequence * bridge,
                                                        int64_t sequence_id,
                                                        void * stream) {
    if (!bridge) return -1;
    std::lock_guard<std::mutex> guard(bridge->mutex);
    for (auto it = bridge->slots.begin(); it != bridge->slots.end();) {
        if (it->second == sequence_id) it = bridge->slots.erase(it);
        else ++it;
    }
    const int status = remove_sequence(bridge, sequence_id, stream);
    if (status != 0) set_error(bridge, cf_multisequence_last_error(bridge->cache));
    else bridge->last_error.clear();
    return status;
}

extern "C" int cf_llama_server_batch_attention_async(cf_llama_server_multisequence * bridge,
                                                        const cf_sequence_attention_item * items,
                                                        uint32_t item_count,
                                                        void * stream) {
    if (!bridge) return -1;
    std::lock_guard<std::mutex> guard(bridge->mutex);
    const int status = cf_multisequence_attention_async(bridge->cache, items, item_count, stream);
    if (status != 0) set_error(bridge, cf_multisequence_last_error(bridge->cache));
    else bridge->last_error.clear();
    return status;
}

extern "C" int cf_llama_server_audit(const cf_llama_server_multisequence * bridge) {
    if (!bridge) return -1;
    std::lock_guard<std::mutex> guard(bridge->mutex);
    return cf_multisequence_audit(bridge->cache);
}

extern "C" const char * cf_llama_server_multisequence_last_error(
    const cf_llama_server_multisequence * bridge) {
    return bridge ? bridge->last_error.c_str() : "null llama-server multisequence bridge";
}
