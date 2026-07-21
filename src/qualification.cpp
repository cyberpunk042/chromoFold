#include "chromofold/qualification.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

struct cf_qualification_runtime {
    explicit cf_qualification_runtime(size_t series_limit, size_t request_limit)
        : max_series(series_limit), max_requests(request_limit) {}

    size_t max_series;
    size_t max_requests;
    mutable std::mutex mutex;
    std::vector<cf_request_measurement> measurements;
    std::unordered_set<std::string> series;
    uint64_t page_refs = 0;
    uint64_t snapshot_refs = 0;
    uint64_t lease_refs = 0;
    uint64_t request_refs = 0;
    bool content_redacted = true;
    bool cardinality_bounded = true;
};

static bool nonempty(const char *value) {
    return value != nullptr && value[0] != '\0';
}

static uint64_t percentile95(std::vector<uint64_t> values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(std::ceil(values.size() * 0.95)) - 1;
    return values[std::min(index, values.size() - 1)];
}

cf_qualification_runtime *cf_qualification_create(size_t max_series, size_t max_requests) {
    if (max_series == 0 || max_requests == 0) return nullptr;
    return new cf_qualification_runtime(max_series, max_requests);
}

void cf_qualification_destroy(cf_qualification_runtime *runtime) {
    delete runtime;
}

int cf_qualification_record(cf_qualification_runtime *runtime, const cf_request_measurement *measurement) {
    if (!runtime || !measurement) return -1;
    if (!nonempty(measurement->trace.trace_id) || !nonempty(measurement->trace.request_id) ||
        !nonempty(measurement->trace.model_id) || !nonempty(measurement->trace.release_version)) return -1;

    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (runtime->measurements.size() >= runtime->max_requests) return -2;

    const std::string series = std::string(measurement->trace.model_id) + ":" +
                               cf_error_code_name(measurement->error) + ":" +
                               measurement->trace.release_version;
    runtime->series.insert(series);
    if (runtime->series.size() > runtime->max_series) runtime->cardinality_bounded = false;
    if (measurement->prompt_content_present != 0) runtime->content_redacted = false;
    runtime->measurements.push_back(*measurement);
    return 0;
}

int cf_qualification_set_reference_state(cf_qualification_runtime *runtime, uint64_t page_refs,
                                         uint64_t snapshot_refs, uint64_t lease_refs,
                                         uint64_t request_refs) {
    if (!runtime) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    runtime->page_refs = page_refs;
    runtime->snapshot_refs = snapshot_refs;
    runtime->lease_refs = lease_refs;
    runtime->request_refs = request_refs;
    return 0;
}

int cf_qualification_compare_regression(double baseline, double candidate,
                                        double maximum_regression_percent) {
    if (baseline <= 0.0 || candidate < 0.0 || maximum_regression_percent < 0.0) return -1;
    const double regression = ((candidate - baseline) / baseline) * 100.0;
    return regression <= maximum_regression_percent ? 1 : 0;
}

int cf_qualification_evaluate(const cf_qualification_runtime *runtime,
                              const cf_slo_policy *policy,
                              cf_qualification_result *result) {
    if (!runtime || !policy || !result) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);

    std::vector<uint64_t> ttft;
    std::vector<uint64_t> token_latency;
    uint64_t errors = 0;
    uint64_t memory_delta = 0;
    for (const auto &measurement : runtime->measurements) {
        ttft.push_back(measurement.queue_us + measurement.route_us + measurement.prefill_us +
                       measurement.encode_us + measurement.transfer_us + measurement.decode_attach_us);
        const uint64_t per_token = measurement.generated_tokens == 0
            ? 0
            : measurement.decode_us / measurement.generated_tokens;
        token_latency.push_back(per_token);
        if (measurement.error != CF_ERR_NONE) ++errors;
        const uint64_t accounted = measurement.model_bytes + measurement.kv_bytes +
            measurement.compressed_page_bytes + measurement.active_tail_bytes +
            measurement.transfer_buffer_bytes + measurement.allocator_reserved_bytes;
        const uint64_t delta = accounted > measurement.observed_vram_bytes
            ? accounted - measurement.observed_vram_bytes
            : measurement.observed_vram_bytes - accounted;
        memory_delta = std::max(memory_delta, delta);
    }

    const uint64_t samples = runtime->measurements.size();
    const double availability = samples == 0 ? 0.0 : 100.0 * (samples - errors) / samples;
    const uint64_t p95_ttft = percentile95(ttft);
    const uint64_t p95_token = percentile95(token_latency);
    const bool refs_ok = runtime->page_refs == 0 && runtime->snapshot_refs == 0 &&
                         runtime->lease_refs == 0 && runtime->request_refs == 0;
    const bool sample_count_ok = samples >= policy->minimum_samples;
    const bool slo_ok = p95_ttft <= policy->ttft_p95_us &&
                        p95_token <= policy->inter_token_p95_us &&
                        availability >= policy->availability_percent;

    result->samples = samples;
    result->errors = errors;
    result->ttft_p95_us = p95_ttft;
    result->inter_token_p95_us = p95_token;
    result->availability_percent = availability;
    result->deadline_success_percent = availability;
    result->memory_accounting_delta_bytes = memory_delta;
    result->cardinality_bounded = runtime->cardinality_bounded ? 1 : 0;
    result->content_redacted = runtime->content_redacted ? 1 : 0;
    result->performance_within_policy = slo_ok ? 1 : 0;
    result->references_reconciled = refs_ok ? 1 : 0;
    result->passed = sample_count_ok && slo_ok && refs_ok && runtime->content_redacted &&
                     runtime->cardinality_bounded ? 1 : 0;
    return 0;
}

const char *cf_error_code_name(cf_error_code code) {
    switch (code) {
        case CF_ERR_NONE: return "none";
        case CF_ERR_ADMISSION_QUOTA: return "admission_quota";
        case CF_ERR_ADMISSION_OVERLOAD: return "admission_overload";
        case CF_ERR_WORKER_UNAVAILABLE: return "worker_unavailable";
        case CF_ERR_TRANSFER_AUTH: return "transfer_auth";
        case CF_ERR_TRANSFER_INTEGRITY: return "transfer_integrity";
        case CF_ERR_LEASE_STALE: return "lease_stale";
        case CF_ERR_CHECKPOINT_INCOMPATIBLE: return "checkpoint_incompatible";
        case CF_ERR_CUDA_OOM: return "cuda_oom";
        case CF_ERR_CUDA_RUNTIME: return "cuda_runtime";
        case CF_ERR_RELEASE_INCOMPATIBLE: return "release_incompatible";
    }
    return "unknown";
}
