#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cf_security_role {
    CF_SECURITY_ROUTER = 1,
    CF_SECURITY_PREFILL = 2,
    CF_SECURITY_DECODE = 3,
    CF_SECURITY_SCHEDULER = 4,
    CF_SECURITY_ADMIN = 5,
} cf_security_role;

typedef enum cf_security_operation {
    CF_OP_REGISTER_WORKER = 1,
    CF_OP_PUBLISH_MANIFEST = 2,
    CF_OP_READ_PAGE = 3,
    CF_OP_TRANSFER_LEASE = 4,
    CF_OP_ROUTE_REQUEST = 5,
    CF_OP_UPDATE_POLICY = 6,
    CF_OP_DRAIN_WORKER = 7,
    CF_OP_INSPECT_USAGE = 8,
} cf_security_operation;

typedef struct cf_security_config {
    uint32_t protocol_version;
    uint32_t current_key_generation;
    uint32_t next_key_generation;
    uint64_t rotation_grace_millis;
    uint64_t replay_window_millis;
    uint32_t require_mtls;
    uint32_t require_manifest_signatures;
    uint32_t require_tenant_binding;
    uint32_t max_replay_entries;
} cf_security_config;

typedef struct cf_peer_identity {
    uint64_t worker_id;
    uint64_t tenant_namespace;
    uint32_t role;
    uint32_t key_generation;
    uint64_t not_before_millis;
    uint64_t not_after_millis;
    uint8_t certificate_fingerprint[32];
} cf_peer_identity;

typedef struct cf_authorization_request {
    cf_peer_identity peer;
    uint32_t operation;
    uint64_t target_tenant_namespace;
    uint64_t model_namespace;
} cf_authorization_request;

typedef struct cf_signed_manifest {
    uint64_t tenant_namespace;
    uint64_t model_namespace;
    uint64_t page_identity_hash;
    uint64_t payload_bytes;
    uint64_t payload_checksum;
    uint64_t lease_generation;
    uint64_t owner_worker_id;
    uint32_t protocol_version;
    uint32_t codec_version;
    uint32_t key_generation;
    uint8_t signature[32];
} cf_signed_manifest;

typedef struct cf_replay_token {
    uint64_t peer_worker_id;
    uint64_t nonce;
    uint64_t timestamp_millis;
    uint64_t operation_hash;
} cf_replay_token;

typedef struct cf_release_component {
    const char * name;
    const char * version;
    const char * sha256;
    const char * license;
} cf_release_component;

typedef struct cf_release_manifest {
    const char * release_version;
    const char * git_commit;
    const char * platform;
    const char * cuda_version;
    uint32_t protocol_version;
    uint32_t codec_version;
    uint32_t checkpoint_version;
    const cf_release_component * components;
    size_t component_count;
    uint8_t signature[32];
} cf_release_manifest;

typedef struct cf_security_counters {
    uint64_t peers_authenticated;
    uint64_t authentication_rejections;
    uint64_t authorization_grants;
    uint64_t authorization_rejections;
    uint64_t manifests_verified;
    uint64_t manifest_tamper_rejections;
    uint64_t replay_rejections;
    uint64_t tenant_substitution_rejections;
    uint64_t key_rotations;
    uint64_t stale_key_rejections;
    uint64_t audit_records;
    uint64_t audit_chain_failures;
    uint64_t release_verifications;
    uint64_t release_verification_failures;
} cf_security_counters;

typedef struct cf_security_runtime cf_security_runtime;

cf_security_runtime * cf_security_create(const cf_security_config * config);
void cf_security_destroy(cf_security_runtime * runtime);
int cf_security_authenticate(cf_security_runtime * runtime, const cf_peer_identity * peer, uint64_t now_millis);
int cf_security_authorize(cf_security_runtime * runtime, const cf_authorization_request * request);
int cf_security_sign_manifest(cf_security_runtime * runtime, cf_signed_manifest * manifest);
int cf_security_verify_manifest(cf_security_runtime * runtime, const cf_signed_manifest * manifest, uint64_t expected_tenant);
int cf_security_check_replay(cf_security_runtime * runtime, const cf_replay_token * token, uint64_t now_millis);
int cf_security_rotate_keys(cf_security_runtime * runtime, uint32_t next_generation, uint64_t now_millis);
int cf_security_append_audit(cf_security_runtime * runtime, uint64_t event_hash, uint64_t * chain_hash);
int cf_security_verify_release(cf_security_runtime * runtime, const cf_release_manifest * manifest);
int cf_security_get_counters(const cf_security_runtime * runtime, cf_security_counters * counters);
const char * cf_security_last_error(const cf_security_runtime * runtime);

#ifdef __cplusplus
}
#endif
