#include "chromofold/security_release.h"

#include <cassert>
#include <cstring>

int main() {
    cf_security_config config{1, 1, 0, 60000, 30000, 1, 1, 1, 128};
    cf_security_runtime * runtime = cf_security_create(&config);
    assert(runtime);

    cf_peer_identity router{};
    router.worker_id = 10;
    router.tenant_namespace = 7;
    router.role = CF_SECURITY_ROUTER;
    router.key_generation = 1;
    router.not_before_millis = 1;
    router.not_after_millis = 100000;
    assert(cf_security_authenticate(runtime, &router, 1000) == 0);

    cf_authorization_request route{router, CF_OP_ROUTE_REQUEST, 7, 11};
    assert(cf_security_authorize(runtime, &route) == 0);
    route.operation = CF_OP_UPDATE_POLICY;
    assert(cf_security_authorize(runtime, &route) != 0);

    cf_signed_manifest manifest{};
    manifest.tenant_namespace = 7;
    manifest.model_namespace = 11;
    manifest.page_identity_hash = 22;
    manifest.payload_bytes = 4096;
    manifest.payload_checksum = 33;
    manifest.lease_generation = 4;
    manifest.owner_worker_id = 12;
    manifest.protocol_version = 1;
    manifest.codec_version = 1;
    assert(cf_security_sign_manifest(runtime, &manifest) == 0);
    assert(cf_security_verify_manifest(runtime, &manifest, 7) == 0);
    cf_signed_manifest tampered = manifest;
    ++tampered.payload_bytes;
    assert(cf_security_verify_manifest(runtime, &tampered, 7) != 0);
    assert(cf_security_verify_manifest(runtime, &manifest, 8) != 0);

    cf_replay_token replay{10, 99, 1000, 55};
    assert(cf_security_check_replay(runtime, &replay, 1000) == 0);
    assert(cf_security_check_replay(runtime, &replay, 1000) != 0);

    assert(cf_security_rotate_keys(runtime, 2, 1200) == 0);
    cf_peer_identity next = router;
    next.key_generation = 2;
    assert(cf_security_authenticate(runtime, &next, 1300) == 0);

    uint64_t chain1 = 0;
    uint64_t chain2 = 0;
    assert(cf_security_append_audit(runtime, 100, &chain1) == 0);
    assert(cf_security_append_audit(runtime, 101, &chain2) == 0);
    assert(chain1 != 0 && chain2 != chain1);

    cf_release_component component{"chromofold", "0.18.0", "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", "Apache-2.0"};
    cf_release_manifest release{"0.18.0", "abcdef", "linux-amd64", "12.8", 1, 1, 1, &component, 1, {}};
    assert(cf_security_verify_release(runtime, &release) == 0);

    cf_security_counters counters{};
    assert(cf_security_get_counters(runtime, &counters) == 0);
    assert(counters.peers_authenticated == 2);
    assert(counters.authorization_grants == 1);
    assert(counters.authorization_rejections == 1);
    assert(counters.manifests_verified == 1);
    assert(counters.manifest_tamper_rejections == 1);
    assert(counters.tenant_substitution_rejections == 1);
    assert(counters.replay_rejections == 1);
    assert(counters.key_rotations == 1);
    assert(counters.audit_records == 2);
    assert(counters.release_verifications == 1);

    cf_security_destroy(runtime);
    return 0;
}
