#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cf_error_code {
    CF_ERR_NONE = 0,
    CF_ERR_ADMISSION_QUOTA,
    CF_ERR_ADMISSION_OVERLOAD,
    CF_ERR_WORKER_UNAVAILABLE,
    CF_ERR_TRANSFER_AUTH,
    CF_ERR_TRANSFER_INTEGRITY,
    CF_ERR_LEASE_STALE,
    CF_ERR_CHECKPOINT_INCOMPATIBLE,
    CF_ERR_CUDA_OOM,
    CF_ERR_CUDA_RUNTIME,
    CF_ERR_RELEASE_INCOMPATIBLE
} cf_error_code;

typedef struct cf_trace_context {
    const char *trace_id;
    const char *request_id;
    const char *tenant_id;
    const char *model_id;
    const char *sequence_id;
    const char *worker_id;
    const char *release_version;
} cf_trace_context;

typedef struct cf_request_measurement {
    cf_trace_context trace;
    uint64_t queue_us;
    uint64_t route_us;
    uint64_t prefill_us;
    uint64_t encode_us;
    uint64_t transfer_us;
    uint64_t decode_attach_us;
    uint64_t decode_us;
    uint64_t output_us;
    uint64_t prompt_tokens;
    uint64_t generated_tokens;
    uint64_t model_bytes;
    uint64_t kv_bytes;
    uint64_t compressed_page_bytes;
    uint64_t active_tail_bytes;
    uint64_t transfer_buffer_bytes;
    uint64_t allocator_reserved_bytes;
    uint64_t observed_vram_bytes;
    cf_error_code error;
    int prompt_content_present;
} cf_request_measurement;

typedef struct cf_slo_policy {
    uint64_t ttft_p95_us;
    uint64_t inter_token_p95_us;
    double availability_percent;
    double deadline_success_percent;
    uint64_t minimum_samples;
} cf_slo_policy;

typedef struct cf_qualification_result {
    uint64_t samples;
    uint64_t errors;
    uint64_t ttft_p95_us;
    uint64_t inter_token_p95_us;
    double availability_percent;
    double deadline_success_percent;
    uint64_t memory_accounting_delta_bytes;
    int cardinality_bounded;
    int content_redacted;
    int performance_within_policy;
    int references_reconciled;
    int passed;
} cf_qualification_result;

typedef struct cf_qualification_runtime cf_qualification_runtime;

cf_qualification_runtime *cf_qualification_create(size_t max_series, size_t max_requests);
void cf_qualification_destroy(cf_qualification_runtime *runtime);
int cf_qualification_record(cf_qualification_runtime *runtime, const cf_request_measurement *measurement);
int cf_qualification_set_reference_state(cf_qualification_runtime *runtime, uint64_t page_refs, uint64_t snapshot_refs, uint64_t lease_refs, uint64_t request_refs);
int cf_qualification_compare_regression(double baseline, double candidate, double maximum_regression_percent);
int cf_qualification_evaluate(const cf_qualification_runtime *runtime, const cf_slo_policy *policy, cf_qualification_result *result);
const char *cf_error_code_name(cf_error_code code);

#ifdef __cplusplus
}
#endif
