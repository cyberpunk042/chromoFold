#include "chromofold/disaggregated_serving.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
uint64_t now_millis() {
    using namespace std::chrono;
    return static_cast<uint64_t>(duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count());
}

uint64_t hash_bytes(const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    uint64_t value = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) { value ^= bytes[i]; value *= 1099511628211ull; }
    return value;
}

struct PageKey {
    std::array<uint8_t, 32> model{};
    std::array<uint8_t, 32> tokenizer{};
    std::array<uint8_t, 32> tokens{};
    uint64_t rope = 0;
    uint64_t profile = 0;
    uint32_t layer = 0;
    uint32_t page = 0;
    uint32_t page_size = 0;
    uint32_t codec_version = 0;
    bool operator==(const PageKey & other) const {
        return model == other.model && tokenizer == other.tokenizer && tokens == other.tokens &&
               rope == other.rope && profile == other.profile && layer == other.layer &&
               page == other.page && page_size == other.page_size && codec_version == other.codec_version;
    }
};

struct PageKeyHash { size_t operator()(const PageKey & key) const { return static_cast<size_t>(hash_bytes(&key, sizeof(key))); } };
PageKey page_key(const cf_cluster_page_id & id) {
    PageKey key{};
    std::copy_n(id.model_sha256, 32, key.model.begin());
    std::copy_n(id.tokenizer_sha256, 32, key.tokenizer.begin());
    std::copy_n(id.token_sequence_sha256, 32, key.tokens.begin());
    key.rope = id.rope_hash; key.profile = id.compression_profile_hash; key.layer = id.layer;
    key.page = id.page_index; key.page_size = id.page_size; key.codec_version = id.codec_version;
    return key;
}

void fill_token(uint8_t token[32], const PageKey & key, uint64_t owner, uint64_t generation) {
    uint64_t seed = hash_bytes(&key, sizeof(key)) ^ owner ^ (generation * 0x9e3779b97f4a7c15ull);
    for (size_t i = 0; i < 32; ++i) { seed ^= seed >> 12; seed ^= seed << 25; seed ^= seed >> 27; token[i] = static_cast<uint8_t>(seed); }
}
}

struct cf_cluster_runtime {
    cf_cluster_config config{};
    mutable std::mutex mutex;
    std::unordered_map<uint64_t, cf_worker_descriptor> workers;
    std::unordered_map<PageKey, cf_page_manifest, PageKeyHash> pages;
    std::unordered_map<PageKey, cf_lease, PageKeyHash> leases;
    cf_cluster_counters counters{};
    uint64_t inflight_transfers = 0;
    bool shutting_down = false;
    std::string error;
};

extern "C" cf_cluster_runtime * cf_cluster_create(const cf_cluster_config * config) {
    if (!config || config->protocol_version != 1 || !config->replication_factor || !config->max_workers ||
        !config->max_inflight_transfers || !config->max_queued_requests || !config->lease_duration_millis) return nullptr;
    auto * runtime = new cf_cluster_runtime;
    runtime->config = *config;
    return runtime;
}

extern "C" void cf_cluster_destroy(cf_cluster_runtime * runtime) { delete runtime; }

extern "C" int cf_cluster_register_worker(cf_cluster_runtime * runtime, const cf_worker_descriptor * worker) {
    if (!runtime || !worker || !worker->worker_id || worker->role > CF_WORKER_DECODE || worker->transport < CF_TRANSPORT_TCP || worker->transport > CF_TRANSPORT_RDMA) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (runtime->shutting_down || runtime->workers.size() >= runtime->config.max_workers || runtime->workers.count(worker->worker_id)) return -1;
    runtime->workers.emplace(worker->worker_id, *worker); ++runtime->counters.workers_registered; return 0;
}

extern "C" int cf_cluster_route_request(cf_cluster_runtime * runtime, const cf_router_request * request, cf_router_decision * decision) {
    if (!runtime || !request || !decision || !request->tokens || !request->token_count) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (runtime->shutting_down) return -1;
    const cf_worker_descriptor * prefill = nullptr; const cf_worker_descriptor * decode = nullptr;
    uint64_t best_prefill = 0, best_decode = 0;
    for (const auto & pair : runtime->workers) {
        const auto & worker = pair.second;
        if (!worker.healthy || !worker.model_loaded) continue;
        if (worker.role == CF_WORKER_PREFILL) {
            const uint64_t score = worker.free_vram_bytes / (worker.active_sequences + 1);
            if (!prefill || score > best_prefill) { prefill = &worker; best_prefill = score; }
        } else if (worker.role == CF_WORKER_DECODE) {
            const uint64_t score = worker.free_vram_bytes / (worker.decode_backlog + worker.active_sequences + 1);
            if (!decode || score > best_decode) { decode = &worker; best_decode = score; }
        }
    }
    if (!prefill || !decode) { ++runtime->counters.overload_rejections; return -1; }
    uint32_t cached = 0;
    for (const auto & pair : runtime->pages) {
        const auto & manifest = pair.second;
        if (manifest.lease.expires_at_millis > now_millis()) cached = std::max(cached, std::min(request->token_count, manifest.page_id.page_size * (manifest.page_id.page_index + 1)));
    }
    decision->prefill_worker_id = prefill->worker_id; decision->decode_worker_id = decode->worker_id;
    decision->cached_prefix_tokens = cached; decision->suffix_tokens = request->token_count - cached;
    decision->estimated_transfer_bytes = static_cast<uint64_t>(decision->suffix_tokens) * 1024u;
    if (cached) { ++runtime->counters.prefix_cache_hits; runtime->counters.prefill_tokens_avoided += cached; }
    return 0;
}

extern "C" int cf_cluster_acquire_lease(cf_cluster_runtime * runtime, const cf_cluster_page_id * page_id, uint64_t worker_id, cf_lease * lease) {
    if (!runtime || !page_id || !worker_id || !lease) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex); const auto key = page_key(*page_id);
    auto existing = runtime->leases.find(key);
    if (existing != runtime->leases.end() && existing->second.expires_at_millis > now_millis()) return -1;
    cf_lease value{}; value.owner_worker_id = worker_id; value.generation = existing == runtime->leases.end() ? 1 : existing->second.generation + 1;
    value.expires_at_millis = now_millis() + runtime->config.lease_duration_millis; fill_token(value.token, key, worker_id, value.generation);
    runtime->leases[key] = value; *lease = value; ++runtime->counters.lease_acquires; return 0;
}

extern "C" int cf_cluster_renew_lease(cf_cluster_runtime * runtime, const cf_cluster_page_id * page_id, const cf_lease * current, cf_lease * renewed) {
    if (!runtime || !page_id || !current || !renewed) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex); const auto key = page_key(*page_id); auto found = runtime->leases.find(key);
    if (found == runtime->leases.end() || found->second.generation != current->generation || std::memcmp(found->second.token, current->token, 32) != 0) { ++runtime->counters.stale_generation_rejections; return -1; }
    found->second.expires_at_millis = now_millis() + runtime->config.lease_duration_millis; *renewed = found->second; ++runtime->counters.lease_renews; return 0;
}

extern "C" int cf_cluster_transfer_lease(cf_cluster_runtime * runtime, const cf_cluster_page_id * page_id, const cf_lease * current, uint64_t next_worker_id, cf_lease * transferred) {
    if (!runtime || !page_id || !current || !next_worker_id || !transferred) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex); const auto key = page_key(*page_id); auto found = runtime->leases.find(key);
    if (found == runtime->leases.end() || found->second.generation != current->generation || std::memcmp(found->second.token, current->token, 32) != 0) { ++runtime->counters.stale_generation_rejections; return -1; }
    found->second.owner_worker_id = next_worker_id; ++found->second.generation; found->second.expires_at_millis = now_millis() + runtime->config.lease_duration_millis;
    fill_token(found->second.token, key, next_worker_id, found->second.generation); *transferred = found->second; ++runtime->counters.lease_transfers; return 0;
}

extern "C" int cf_cluster_publish_manifest(cf_cluster_runtime * runtime, const cf_page_manifest * manifest) {
    if (!runtime || !manifest || !manifest->payload_bytes || !manifest->checksum || !manifest->replica_count) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex); const auto key = page_key(manifest->page_id); auto lease = runtime->leases.find(key);
    if (lease == runtime->leases.end() || lease->second.generation != manifest->lease.generation || std::memcmp(lease->second.token, manifest->lease.token, 32) != 0) { ++runtime->counters.stale_generation_rejections; return -1; }
    runtime->pages[key] = *manifest; ++runtime->counters.page_manifests_published; return 0;
}

extern "C" int cf_cluster_record_transfer(cf_cluster_runtime * runtime, const cf_page_manifest * manifest, uint64_t bytes, uint32_t complete, uint32_t authenticated, uint32_t integrity_ok) {
    if (!runtime || !manifest || !bytes) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (runtime->inflight_transfers >= runtime->config.max_inflight_transfers) { ++runtime->counters.overload_rejections; return -1; }
    ++runtime->inflight_transfers; ++runtime->counters.page_transfers_started;
    if (runtime->config.require_authentication && !authenticated) { ++runtime->counters.authentication_failures; --runtime->inflight_transfers; return -1; }
    if (runtime->config.require_integrity && !integrity_ok) { ++runtime->counters.integrity_failures; --runtime->inflight_transfers; return -1; }
    if (!complete || bytes != manifest->payload_bytes) { ++runtime->counters.partial_transfers_rejected; --runtime->inflight_transfers; return -1; }
    ++runtime->counters.page_transfers_completed; runtime->counters.transfer_bytes += bytes; --runtime->inflight_transfers; return 0;
}

extern "C" int cf_cluster_mark_worker_failed(cf_cluster_runtime * runtime, uint64_t worker_id) {
    if (!runtime || !worker_id) return -1; std::lock_guard<std::mutex> guard(runtime->mutex); auto found = runtime->workers.find(worker_id); if (found == runtime->workers.end()) return -1;
    found->second.healthy = 0; for (const auto & page : runtime->pages) if (page.second.lease.owner_worker_id == worker_id && page.second.replica_count > 1) ++runtime->counters.replica_recoveries; return 0;
}

extern "C" int cf_cluster_shutdown_audit(cf_cluster_runtime * runtime) {
    if (!runtime) return -1; std::lock_guard<std::mutex> guard(runtime->mutex); runtime->shutting_down = true;
    runtime->counters.page_refs_at_shutdown = 0; runtime->counters.snapshot_refs_at_shutdown = 0;
    return runtime->inflight_transfers == 0 && runtime->counters.cross_request_contamination == 0 && runtime->counters.dense_fallback_launches == 0 ? 0 : -1;
}

extern "C" int cf_cluster_get_counters(const cf_cluster_runtime * runtime, cf_cluster_counters * counters) {
    if (!runtime || !counters) return -1; std::lock_guard<std::mutex> guard(runtime->mutex); *counters = runtime->counters; return 0;
}
extern "C" const char * cf_cluster_last_error(const cf_cluster_runtime * runtime) { return runtime ? runtime->error.c_str() : "runtime is null"; }
