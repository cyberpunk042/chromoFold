#include "chromofold/rc1_runtime.h"

#include <cassert>
#include <cstring>

int main() {
  cf_rc1_config config{128, 256, 800.0, 100.0, 0.05};
  cf_rc1_runtime *runtime = cf_rc1_create(&config);
  assert(runtime != nullptr);

  for (int i = 0; i < 20; ++i) {
    cf_rc1_request_event event{};
    event.trace_id = "trace";
    event.request_id = "request";
    event.tenant_id = "tenant-a";
    event.model_id = "model-a";
    event.worker_id = "decode-1";
    event.release_digest = "sha256:qualified";
    event.queue_ms = 5.0;
    event.prefill_ms = 80.0;
    event.transfer_ms = 10.0;
    event.decode_ms = 30.0;
    event.ttft_ms = 125.0 + i;
    event.inter_token_ms = 40.0 + i * 0.5;
    event.prompt_tokens = 512;
    event.completion_tokens = 128;
    assert(cf_rc1_record_request(runtime, &event) == 0);
  }

  cf_rc1_memory_sample memory{};
  memory.model_bytes = 500;
  memory.kv_bytes = 200;
  memory.compressed_page_bytes = 100;
  memory.active_tail_bytes = 50;
  memory.transfer_buffer_bytes = 50;
  memory.allocator_reserved_bytes = 100;
  memory.observed_vram_bytes = 1000;
  assert(cf_rc1_record_memory(runtime, &memory) == 0);

  cf_rc1_reference_sample refs{};
  assert(cf_rc1_record_references(runtime, &refs) == 0);

  cf_rc1_summary summary{};
  assert(cf_rc1_finalize(runtime, &summary) == 0);
  assert(summary.requests == 20);
  assert(summary.decision == CF_RC1_PASS);
  assert(std::strcmp(cf_rc1_decision_name(summary.decision), "PASS") == 0);

  cf_rc1_request_event leaked{};
  leaked.trace_id = "trace";
  leaked.request_id = "request-2";
  leaked.tenant_id = "tenant-a";
  leaked.model_id = "model-a";
  leaked.worker_id = "decode-1";
  leaked.release_digest = "sha256:qualified";
  leaked.prompt_content_present = 1;
  leaked.ttft_ms = 1.0;
  leaked.inter_token_ms = 1.0;
  assert(cf_rc1_record_request(runtime, &leaked) == 0);
  assert(cf_rc1_finalize(runtime, &summary) == 0);
  assert(summary.decision == CF_RC1_FAIL);

  cf_rc1_destroy(runtime);
  return 0;
}
