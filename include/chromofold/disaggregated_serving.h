#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum cf_worker_role {
    CF_WORKER_STANDALONE = 0,
    CF_WORKER_ROUTER = 1,
    CF_WORKER_PREFILL = 2,
    CF_WORKER_DECODE = 3,
} cf_worker_role;

typedef enum cf_transport_kind {
    CF_TRANSPORT_TCP = 1,
    CF_TRANSPORT_UNIX = 2,
    CF_TRANSPORT_PINNED_HOST = 3,
    CF_TRANSPORT_CUDA_IPC = 4,
    CF_TRANSPORT_RDMA = 5,
} cf_transport_kind;

typedef enum cf_message_type {
    CF_MSG_HELLO = 1,
    CF_MSG_CAPABILITIES = 2,
    CF_MSG_PREFILL_REQUEST = 3,
    CF_MSG_PAGE_MANIFEST = 4,
    CF_MSG_PAGE_DATA = 5,
    CF_MSG_PAGE_ACK = 6,
    CF_MSG_LEASE_ACQUIRE = 7,
    CF_MSG_LEASE_RENEW = 8,
    CF_MSG_LEASE_TRANSFER = 9,
    CF_MSG_DECODE_ATTACH = 10,
    CF_MSG_CANCEL = 11,
    CF_MSG_ERROR = 12,
} cf_message_type;

typedef struct cf_cluster_page_id {
    uint8_t model_sha256[32];
    uint8_t tokenizer_sha256[32];
    uint8_t token_sequence_sha256[32];
    uint64_t rope_hash;
    uint64_t compression_profile_hash;
    uint32_t layer;
    uint32_t page_index;
    uint32_t page_size;
    uint32_t codec_version;
} cf_cluster_page_id;

typedef struct cf_worker_descriptor {
    uint64_t worker_id;
    uint32_t role;
    uint32_t transport;
    uint32_t healthy;
    uint32_t model_loaded;
    uint64_t free_vram_bytes;
    uint64_t active_sequences;
    uint64_t decode_backlog;
    uint64_t cache_namespace;
} cf_worker_descriptor;

typedef struct cf_lease {
    uint64_t owner_worker_id;
    uint64_t generation;
    uint64_t expires_at_millis;
    uint8_t token[32];
} cf_lease;

typedef struct cf_page_manifest {
    cf_cluster_page_id page_id;
    cf_lease lease;
    uint64_t payload_bytes;
    uint64_t checksum;
    uint32_t key_codec;
    uint32_t value_codec;
    uint32_t replica_count;
} cf_page_manifest;

typedef struct cf_router_request {
    const uint32_t * tokens;
    uint32_t token_count;
    uint64_t model_hash;
    uint64_t request_id;
} cf_router_request;

typedef struct cf_router_decision {
    uint64_t prefill_worker_id;
    uint64_t decode_worker_id;
    uint32_t cached_prefix_tokens;
    uint32_t suffix_tokens;
    uint64_t estimated_transfer_bytes;
} cf_router_decision;

typedef struct cf_cluster_config {
    uint32_t protocol_version;
    uint32_t replication_factor;
    uint32_t require_authentication;
    uint32_t require_integrity;
    uint32_t allow_host_staging;
    uint32_t max_workers;
    uint32_t max_inflight_transfers;
    uint32_t max_queued_requests;
    uint64_t lease_duration_millis;
    uint64_t max_pinned_host_bytes;
} cf_cluster_config;

typedef struct cf_cluster_counters {
    uint64_t workers_registered;
    uint64_t prefix_cache_hits;
    uint64_t prefill_tokens_avoided;
    uint64_t page_manifests_published;
    uint64_t page_transfers_started;
    uint64_t page_transfers_completed;
    uint64_t transfer_bytes;
    uint64_t partial_transfers_rejected;
    uint64_t authentication_failures;
    uint64_t integrity_failures;
    uint64_t replay_rejections;
    uint64_t lease_acquires;
    uint64_t lease_renews;
    uint64_t lease_transfers;
    uint64_t stale_generation_rejections;
    uint64_t replica_recoveries;
    uint64_t router_retries;
    uint64_t overload_rejections;
    uint64_t cross_request_contamination;
    uint64_t dense_fallback_launches;
    uint64_t page_refs_at_shutdown;
    uint64_t snapshot_refs_at_shutdown;
} cf_cluster_counters;

typedef struct cf_cluster_runtime cf_cluster_runtime;
cf_cluster_runtime * cf_cluster_create(const cf_cluster_config * config);
void cf_cluster_destroy(cf_cluster_runtime * runtime);
int cf_cluster_register_worker(cf_cluster_runtime * runtime, const cf_worker_descriptor * worker);
int cf_cluster_route_request(cf_cluster_runtime * runtime, const cf_router_request * request, cf_router_decision * decision);
int cf_cluster_publish_manifest(cf_cluster_runtime * runtime, const cf_page_manifest * manifest);
int cf_cluster_acquire_lease(cf_cluster_runtime * runtime, const cf_cluster_page_id * page_id, uint64_t worker_id, cf_lease * lease);
int cf_cluster_renew_lease(cf_cluster_runtime * runtime, const cf_cluster_page_id * page_id, const cf_lease * current, cf_lease * renewed);
int cf_cluster_transfer_lease(cf_cluster_runtime * runtime, const cf_cluster_page_id * page_id, const cf_lease * current, uint64_t next_worker_id, cf_lease * transferred);
int cf_cluster_record_transfer(cf_cluster_runtime * runtime, const cf_page_manifest * manifest, uint64_t bytes, uint32_t complete, uint32_t authenticated, uint32_t integrity_ok);
int cf_cluster_mark_worker_failed(cf_cluster_runtime * runtime, uint64_t worker_id);
int cf_cluster_shutdown_audit(cf_cluster_runtime * runtime);
int cf_cluster_get_counters(const cf_cluster_runtime * runtime, cf_cluster_counters * counters);
const char * cf_cluster_last_error(const cf_cluster_runtime * runtime);

#ifdef __cplusplus
}
#endif
