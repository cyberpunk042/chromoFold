#include "chromofold/rc1_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <string>
#include <vector>

struct cf_rc1_runtime {
  cf_rc1_config config{};
  std::vector<double> ttft;
  std::vector<double> inter_token;
  std::uint64_t prompt_leaks = 0;
  std::uint64_t correctness_failures = 0;
  std::uint64_t cuda_errors = 0;
  std::uint64_t cross_tenant_contamination = 0;
  std::uint64_t unreconciled_references = 0;
  double memory_error_ratio = 0.0;
  bool memory_recorded = false;
};

namespace {
bool text_present(const char *value) { return value != nullptr && value[0] != '\0'; }

double percentile95(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(std::ceil(values.size() * 0.95)) - 1;
  return values[std::min(index, values.size() - 1)];
}
}

extern "C" cf_rc1_runtime *cf_rc1_create(const cf_rc1_config *config) {
  if (config == nullptr || config->max_requests == 0 || config->max_metric_series == 0) return nullptr;
  auto *runtime = new (std::nothrow) cf_rc1_runtime;
  if (runtime == nullptr) return nullptr;
  runtime->config = *config;
  runtime->ttft.reserve(config->max_requests);
  runtime->inter_token.reserve(config->max_requests);
  return runtime;
}

extern "C" void cf_rc1_destroy(cf_rc1_runtime *runtime) { delete runtime; }

extern "C" int cf_rc1_record_request(cf_rc1_runtime *runtime, const cf_rc1_request_event *event) {
  if (runtime == nullptr || event == nullptr) return -1;
  if (runtime->ttft.size() >= runtime->config.max_requests) return -2;
  if (!text_present(event->trace_id) || !text_present(event->request_id) || !text_present(event->tenant_id) ||
      !text_present(event->model_id) || !text_present(event->worker_id) || !text_present(event->release_digest)) return -3;
  if (event->ttft_ms < 0.0 || event->inter_token_ms < 0.0 || event->queue_ms < 0.0 || event->prefill_ms < 0.0 ||
      event->transfer_ms < 0.0 || event->decode_ms < 0.0) return -4;
  runtime->ttft.push_back(event->ttft_ms);
  runtime->inter_token.push_back(event->inter_token_ms);
  runtime->prompt_leaks += event->prompt_content_present != 0;
  runtime->correctness_failures += event->correctness_failure != 0;
  runtime->cuda_errors += event->cuda_error != 0;
  runtime->cross_tenant_contamination += event->cross_tenant_contamination != 0;
  return 0;
}

extern "C" int cf_rc1_record_memory(cf_rc1_runtime *runtime, const cf_rc1_memory_sample *sample) {
  if (runtime == nullptr || sample == nullptr || sample->observed_vram_bytes == 0) return -1;
  const long double accounted = static_cast<long double>(sample->model_bytes) + sample->kv_bytes +
      sample->compressed_page_bytes + sample->active_tail_bytes + sample->transfer_buffer_bytes +
      sample->allocator_reserved_bytes;
  const long double observed = static_cast<long double>(sample->observed_vram_bytes);
  runtime->memory_error_ratio = static_cast<double>(std::fabs(accounted - observed) / observed);
  runtime->memory_recorded = true;
  return 0;
}

extern "C" int cf_rc1_record_references(cf_rc1_runtime *runtime, const cf_rc1_reference_sample *sample) {
  if (runtime == nullptr || sample == nullptr) return -1;
  runtime->unreconciled_references = sample->page_refs + sample->snapshot_refs + sample->lease_refs + sample->request_refs;
  return 0;
}

extern "C" int cf_rc1_finalize(cf_rc1_runtime *runtime, cf_rc1_summary *summary) {
  if (runtime == nullptr || summary == nullptr) return -1;
  std::memset(summary, 0, sizeof(*summary));
  summary->requests = runtime->ttft.size();
  summary->ttft_p95_ms = percentile95(runtime->ttft);
  summary->inter_token_p95_ms = percentile95(runtime->inter_token);
  summary->memory_error_ratio = runtime->memory_error_ratio;
  summary->prompt_leaks = runtime->prompt_leaks;
  summary->correctness_failures = runtime->correctness_failures;
  summary->cuda_errors = runtime->cuda_errors;
  summary->cross_tenant_contamination = runtime->cross_tenant_contamination;
  summary->unreconciled_references = runtime->unreconciled_references;
  if (summary->requests == 0 || !runtime->memory_recorded) {
    summary->decision = CF_RC1_INCOMPLETE;
  } else if (summary->prompt_leaks != 0 || summary->correctness_failures != 0 || summary->cuda_errors != 0 ||
             summary->cross_tenant_contamination != 0 || summary->unreconciled_references != 0 ||
             summary->ttft_p95_ms > runtime->config.max_ttft_p95_ms ||
             summary->inter_token_p95_ms > runtime->config.max_inter_token_p95_ms ||
             summary->memory_error_ratio > runtime->config.max_memory_error_ratio) {
    summary->decision = CF_RC1_FAIL;
  } else {
    summary->decision = CF_RC1_PASS;
  }
  return 0;
}

extern "C" const char *cf_rc1_decision_name(cf_rc1_decision decision) {
  switch (decision) {
    case CF_RC1_PASS: return "PASS";
    case CF_RC1_FAIL: return "FAIL";
    case CF_RC1_INCOMPLETE: return "INCOMPLETE";
  }
  return "UNKNOWN";
}
