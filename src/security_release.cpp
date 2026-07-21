#include "chromofold/security_release.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>

namespace {

uint64_t mix64(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31;
    return value;
}

void digest32(uint8_t out[32], const uint64_t * words, size_t count, uint32_t key_generation) {
    uint64_t state = mix64(0x4348524f4d4f464full ^ key_generation);
    for (size_t index = 0; index < count; ++index) {
        state = mix64(state ^ words[index] ^ static_cast<uint64_t>(index));
    }
    for (size_t index = 0; index < 4; ++index) {
        const uint64_t word = mix64(state ^ (index * 0x9e3779b97f4a7c15ull));
        std::memcpy(out + index * sizeof(word), &word, sizeof(word));
    }
}

uint64_t replay_hash(const cf_replay_token & token) {
    return mix64(token.peer_worker_id) ^ mix64(token.nonce) ^
           mix64(token.timestamp_millis) ^ mix64(token.operation_hash);
}

bool is_runtime_role(uint32_t role) {
    return role >= CF_SECURITY_ROUTER && role <= CF_SECURITY_ADMIN;
}

bool role_allows(uint32_t role, uint32_t operation) {
    switch (operation) {
        case CF_OP_REGISTER_WORKER:
            return role == CF_SECURITY_ROUTER || role == CF_SECURITY_SCHEDULER || role == CF_SECURITY_ADMIN;
        case CF_OP_PUBLISH_MANIFEST:
            return role == CF_SECURITY_PREFILL || role == CF_SECURITY_DECODE;
        case CF_OP_READ_PAGE:
            return role == CF_SECURITY_PREFILL || role == CF_SECURITY_DECODE || role == CF_SECURITY_ROUTER;
        case CF_OP_TRANSFER_LEASE:
            return role == CF_SECURITY_PREFILL || role == CF_SECURITY_DECODE || role == CF_SECURITY_SCHEDULER;
        case CF_OP_ROUTE_REQUEST:
            return role == CF_SECURITY_ROUTER || role == CF_SECURITY_SCHEDULER;
        case CF_OP_UPDATE_POLICY:
        case CF_OP_DRAIN_WORKER:
        case CF_OP_INSPECT_USAGE:
            return role == CF_SECURITY_ADMIN;
        default:
            return false;
    }
}

} // namespace

struct cf_security_runtime {
    cf_security_config config{};
    mutable std::mutex mutex;
    std::unordered_set<uint64_t> replay_cache;
    cf_security_counters counters{};
    uint64_t rotation_started_millis = 0;
    uint64_t audit_chain = 0;
    std::string error;
};

extern "C" cf_security_runtime * cf_security_create(const cf_security_config * config) {
    if (!config || config->protocol_version != 1 || config->current_key_generation == 0 ||
        config->replay_window_millis == 0 || config->max_replay_entries == 0) {
        return nullptr;
    }
    auto * runtime = new cf_security_runtime;
    runtime->config = *config;
    return runtime;
}

extern "C" void cf_security_destroy(cf_security_runtime * runtime) {
    delete runtime;
}

extern "C" int cf_security_authenticate(cf_security_runtime * runtime,
                                           const cf_peer_identity * peer,
                                           uint64_t now_millis) {
    if (!runtime || !peer || peer->worker_id == 0 || !is_runtime_role(peer->role)) {
        return -1;
    }
    std::lock_guard<std::mutex> guard(runtime->mutex);
    const bool current_key = peer->key_generation == runtime->config.current_key_generation;
    const bool next_key = runtime->config.next_key_generation != 0 &&
                          peer->key_generation == runtime->config.next_key_generation;
    const bool within_grace = runtime->rotation_started_millis != 0 &&
                              now_millis <= runtime->rotation_started_millis + runtime->config.rotation_grace_millis;
    if ((!current_key && !(next_key && within_grace)) || now_millis < peer->not_before_millis ||
        now_millis > peer->not_after_millis) {
        ++runtime->counters.authentication_rejections;
        if (!current_key && !next_key) {
            ++runtime->counters.stale_key_rejections;
        }
        return -1;
    }
    ++runtime->counters.peers_authenticated;
    return 0;
}

extern "C" int cf_security_authorize(cf_security_runtime * runtime,
                                        const cf_authorization_request * request) {
    if (!runtime || !request || !role_allows(request->peer.role, request->operation)) {
        if (runtime) {
            std::lock_guard<std::mutex> guard(runtime->mutex);
            ++runtime->counters.authorization_rejections;
        }
        return -1;
    }
    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (runtime->config.require_tenant_binding && request->target_tenant_namespace != 0 &&
        request->peer.tenant_namespace != request->target_tenant_namespace &&
        request->peer.role != CF_SECURITY_ADMIN) {
        ++runtime->counters.authorization_rejections;
        ++runtime->counters.tenant_substitution_rejections;
        return -1;
    }
    ++runtime->counters.authorization_grants;
    return 0;
}

extern "C" int cf_security_sign_manifest(cf_security_runtime * runtime,
                                            cf_signed_manifest * manifest) {
    if (!runtime || !manifest || manifest->tenant_namespace == 0 || manifest->payload_bytes == 0 ||
        manifest->protocol_version != runtime->config.protocol_version) {
        return -1;
    }
    std::lock_guard<std::mutex> guard(runtime->mutex);
    manifest->key_generation = runtime->config.current_key_generation;
    const uint64_t words[] = {
        manifest->tenant_namespace,
        manifest->model_namespace,
        manifest->page_identity_hash,
        manifest->payload_bytes,
        manifest->payload_checksum,
        manifest->lease_generation,
        manifest->owner_worker_id,
        (static_cast<uint64_t>(manifest->protocol_version) << 32) | manifest->codec_version,
    };
    digest32(manifest->signature, words, sizeof(words) / sizeof(words[0]), manifest->key_generation);
    return 0;
}

extern "C" int cf_security_verify_manifest(cf_security_runtime * runtime,
                                              const cf_signed_manifest * manifest,
                                              uint64_t expected_tenant) {
    if (!runtime || !manifest || expected_tenant == 0) {
        return -1;
    }
    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (runtime->config.require_tenant_binding && manifest->tenant_namespace != expected_tenant) {
        ++runtime->counters.tenant_substitution_rejections;
        return -1;
    }
    if (manifest->protocol_version != runtime->config.protocol_version ||
        (manifest->key_generation != runtime->config.current_key_generation &&
         manifest->key_generation != runtime->config.next_key_generation)) {
        ++runtime->counters.manifest_tamper_rejections;
        return -1;
    }
    const uint64_t words[] = {
        manifest->tenant_namespace,
        manifest->model_namespace,
        manifest->page_identity_hash,
        manifest->payload_bytes,
        manifest->payload_checksum,
        manifest->lease_generation,
        manifest->owner_worker_id,
        (static_cast<uint64_t>(manifest->protocol_version) << 32) | manifest->codec_version,
    };
    uint8_t expected[32]{};
    digest32(expected, words, sizeof(words) / sizeof(words[0]), manifest->key_generation);
    if (std::memcmp(expected, manifest->signature, sizeof(expected)) != 0) {
        ++runtime->counters.manifest_tamper_rejections;
        return -1;
    }
    ++runtime->counters.manifests_verified;
    return 0;
}

extern "C" int cf_security_check_replay(cf_security_runtime * runtime,
                                           const cf_replay_token * token,
                                           uint64_t now_millis) {
    if (!runtime || !token || token->peer_worker_id == 0 || token->nonce == 0) {
        return -1;
    }
    std::lock_guard<std::mutex> guard(runtime->mutex);
    const uint64_t oldest = now_millis > runtime->config.replay_window_millis
                                ? now_millis - runtime->config.replay_window_millis
                                : 0;
    if (token->timestamp_millis < oldest || token->timestamp_millis > now_millis + runtime->config.replay_window_millis) {
        ++runtime->counters.replay_rejections;
        return -1;
    }
    const uint64_t hash = replay_hash(*token);
    if (runtime->replay_cache.count(hash) != 0) {
        ++runtime->counters.replay_rejections;
        return -1;
    }
    if (runtime->replay_cache.size() >= runtime->config.max_replay_entries) {
        runtime->replay_cache.clear();
    }
    runtime->replay_cache.insert(hash);
    return 0;
}

extern "C" int cf_security_rotate_keys(cf_security_runtime * runtime,
                                          uint32_t next_generation,
                                          uint64_t now_millis) {
    if (!runtime || next_generation <= runtime->config.current_key_generation) {
        return -1;
    }
    std::lock_guard<std::mutex> guard(runtime->mutex);
    runtime->config.next_key_generation = next_generation;
    runtime->rotation_started_millis = now_millis;
    ++runtime->counters.key_rotations;
    return 0;
}

extern "C" int cf_security_append_audit(cf_security_runtime * runtime,
                                           uint64_t event_hash,
                                           uint64_t * chain_hash) {
    if (!runtime || !chain_hash || event_hash == 0) {
        return -1;
    }
    std::lock_guard<std::mutex> guard(runtime->mutex);
    runtime->audit_chain = mix64(runtime->audit_chain ^ event_hash ^ runtime->counters.audit_records);
    ++runtime->counters.audit_records;
    *chain_hash = runtime->audit_chain;
    return 0;
}

extern "C" int cf_security_verify_release(cf_security_runtime * runtime,
                                             const cf_release_manifest * manifest) {
    if (!runtime || !manifest || !manifest->release_version || !manifest->git_commit ||
        !manifest->platform || manifest->protocol_version != runtime->config.protocol_version ||
        manifest->component_count == 0 || !manifest->components) {
        if (runtime) {
            std::lock_guard<std::mutex> guard(runtime->mutex);
            ++runtime->counters.release_verification_failures;
        }
        return -1;
    }
    for (size_t index = 0; index < manifest->component_count; ++index) {
        const auto & component = manifest->components[index];
        if (!component.name || !component.version || !component.sha256 || !component.license ||
            std::strlen(component.sha256) != 64) {
            std::lock_guard<std::mutex> guard(runtime->mutex);
            ++runtime->counters.release_verification_failures;
            return -1;
        }
    }
    std::lock_guard<std::mutex> guard(runtime->mutex);
    ++runtime->counters.release_verifications;
    return 0;
}

extern "C" int cf_security_get_counters(const cf_security_runtime * runtime,
                                           cf_security_counters * counters) {
    if (!runtime || !counters) {
        return -1;
    }
    std::lock_guard<std::mutex> guard(runtime->mutex);
    *counters = runtime->counters;
    return 0;
}

extern "C" const char * cf_security_last_error(const cf_security_runtime * runtime) {
    return runtime ? runtime->error.c_str() : "runtime is null";
}
