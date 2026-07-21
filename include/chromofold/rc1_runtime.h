#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cf_rc1_decision {
  CF_RC1_PASS = 0,
  CF_RC1_FAIL = 1,
  CF_RC1_INCOMPLETE = 2
} cf_rc1_decision;

typedef struct cf_rc1_runtime cf_rc1_runtime;

typedef struct cf_rc1_config {
  std::size_t max_requests;
  std::size_t max_metric_series;
  double max_ttft_p95_ms;
  double max_inter_token_p95_ms;
  double max_memory_error_ratio;
} cf_rc1_config;

typedef struct cf_rc1_request_event {
  const char *trace_id;
  const char *request_id;
  const char *tenant_id;
  const char *model_id;
  const char *worker_id;
  const char *release_digest;
  double queue_ms;
  double prefill_ms;
  double transfer_ms;
  double decode_ms;
  double ttft_ms;
  double inter_token_ms;
  std::uint64_t prompt_tokens;
  std::uint64_t completion_tokens;
  int prompt_content_present;
  int correctness_failure;
  int cuda_error;
  int cross_tenant_contamination;
} cf_rc1_request_event;

typedef struct cf_rc1_memory_sample {
  std::uint64_t model_bytes;
  std::uint64_t kv_bytes;
  std::uint64_t compressed_page_bytes;
  std::uint64_t active_tail_bytes;
  std::uint64_t transfer_buffer_bytes;
  std::uint64_t allocator_reserved_bytes;
  std::uint64_t observed_vram_bytes;
} cf_rc1_memory_sample;

typedef struct cf_rc1_reference_sample {
  std::uint64_t page_refs;
  std::uint64_t snapshot_refs;
  std::uint64_t lease_refs;
  std::uint64_t request_refs;
} cf_rc1_reference_sample;

typedef struct cf_rc1_summary {
  std::size_t requests;
  double ttft_p95_ms;
  double inter_token_p95_ms;
  double memory_error_ratio;
  std::uint64_t prompt_leaks;
  std::uint64_t correctness_failures;
  std::uint64_t cuda_errors;
  std::uint64_t cross_tenant_contamination;
  std::uint64_t unreconciled_references;
  cf_rc1_decision decision;
} cf_rc1_summary;

cf_rc1_runtime *cf_rc1_create(const cf_rc1_config *config);
void cf_rc1_destroy(cf_rc1_runtime *runtime);
int cf_rc1_record_request(cf_rc1_runtime *runtime, const cf_rc1_request_event *event);
int cf_rc1_record_memory(cf_rc1_runtime *runtime, const cf_rc1_memory_sample *sample);
int cf_rc1_record_references(cf_rc1_runtime *runtime, const cf_rc1_reference_sample *sample);
int cf_rc1_finalize(cf_rc1_runtime *runtime, cf_rc1_summary *summary);
const char *cf_rc1_decision_name(cf_rc1_decision decision);

#ifdef __cplusplus
}
#endif
