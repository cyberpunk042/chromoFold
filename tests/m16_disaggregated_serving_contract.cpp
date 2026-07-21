#include "chromofold/disaggregated_serving.h"
#include <cassert>
#include <cstring>

int main() {
    cf_cluster_config config{1, 2, 1, 1, 0, 8, 4, 32, 10000, 1ull << 30};
    cf_cluster_runtime * runtime = cf_cluster_create(&config);
    assert(runtime);
    cf_worker_descriptor prefill{1, CF_WORKER_PREFILL, CF_TRANSPORT_TCP, 1, 1, 16ull << 30, 0, 0, 7};
    cf_worker_descriptor decode1{2, CF_WORKER_DECODE, CF_TRANSPORT_TCP, 1, 1, 16ull << 30, 0, 0, 7};
    cf_worker_descriptor decode2{3, CF_WORKER_DECODE, CF_TRANSPORT_TCP, 1, 1, 8ull << 30, 0, 1, 7};
    assert(cf_cluster_register_worker(runtime, &prefill) == 0);
    assert(cf_cluster_register_worker(runtime, &decode1) == 0);
    assert(cf_cluster_register_worker(runtime, &decode2) == 0);

    cf_cluster_page_id page{}; page.layer = 1; page.page_index = 0; page.page_size = 16; page.codec_version = 1;
    page.model_sha256[0] = 1; page.tokenizer_sha256[0] = 2; page.token_sequence_sha256[0] = 3;
    cf_lease lease{}; assert(cf_cluster_acquire_lease(runtime, &page, 1, &lease) == 0);
    cf_page_manifest manifest{}; manifest.page_id = page; manifest.lease = lease; manifest.payload_bytes = 4096;
    manifest.checksum = 42; manifest.key_codec = 4; manifest.value_codec = 2; manifest.replica_count = 2;
    assert(cf_cluster_publish_manifest(runtime, &manifest) == 0);
    assert(cf_cluster_record_transfer(runtime, &manifest, 4096, 1, 1, 1) == 0);
    assert(cf_cluster_record_transfer(runtime, &manifest, 1024, 0, 1, 1) != 0);

    uint32_t tokens[32]{}; cf_router_request request{tokens, 32, 1, 99}; cf_router_decision decision{};
    assert(cf_cluster_route_request(runtime, &request, &decision) == 0);
    assert(decision.prefill_worker_id == 1 && decision.decode_worker_id == 2);
    assert(decision.cached_prefix_tokens == 16 && decision.suffix_tokens == 16);

    cf_lease transferred{}; assert(cf_cluster_transfer_lease(runtime, &page, &lease, 2, &transferred) == 0);
    assert(transferred.generation == lease.generation + 1);
    assert(cf_cluster_renew_lease(runtime, &page, &lease, &transferred) != 0);
    assert(cf_cluster_mark_worker_failed(runtime, 2) == 0);
    assert(cf_cluster_shutdown_audit(runtime) == 0);
    cf_cluster_counters counters{}; assert(cf_cluster_get_counters(runtime, &counters) == 0);
    assert(counters.prefix_cache_hits == 1 && counters.prefill_tokens_avoided == 16);
    assert(counters.partial_transfers_rejected == 1 && counters.replica_recoveries >= 1);
    assert(counters.dense_fallback_launches == 0 && counters.cross_request_contamination == 0);
    cf_cluster_destroy(runtime);
}
