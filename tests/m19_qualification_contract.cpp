#include "chromofold/qualification.h"

#include <cassert>
#include <cstring>

static cf_request_measurement sample(const char *request, uint64_t queue, uint64_t decode) {
    cf_request_measurement m{};
    m.trace = {"trace-1", request, "tenant-a", "model-a", "seq-1", "worker-a", "1.0.0-rc1"};
    m.queue_us = queue;
    m.route_us = 10;
    m.prefill_us = 100;
    m.encode_us = 20;
    m.transfer_us = 20;
    m.decode_attach_us = 10;
    m.decode_us = decode;
    m.generated_tokens = 10;
    m.model_bytes = 100;
    m.kv_bytes = 100;
    m.compressed_page_bytes = 50;
    m.active_tail_bytes = 25;
    m.transfer_buffer_bytes = 25;
    m.allocator_reserved_bytes = 100;
    m.observed_vram_bytes = 400;
    return m;
}

int main() {
    auto *runtime = cf_qualification_create(8, 16);
    assert(runtime != nullptr);
    auto a = sample("request-1", 20, 400);
    auto b = sample("request-2", 30, 500);
    assert(cf_qualification_record(runtime, &a) == 0);
    assert(cf_qualification_record(runtime, &b) == 0);
    assert(cf_qualification_set_reference_state(runtime, 0, 0, 0, 0) == 0);
    assert(cf_qualification_compare_regression(100.0, 104.0, 5.0) == 1);
    assert(cf_qualification_compare_regression(100.0, 106.0, 5.0) == 0);

    cf_slo_policy policy{500, 80, 99.0, 99.0, 2};
    cf_qualification_result result{};
    assert(cf_qualification_evaluate(runtime, &policy, &result) == 0);
    assert(result.passed == 1);
    assert(result.content_redacted == 1);
    assert(result.cardinality_bounded == 1);
    assert(result.references_reconciled == 1);
    assert(result.memory_accounting_delta_bytes == 0);
    assert(std::strcmp(cf_error_code_name(CF_ERR_CUDA_OOM), "cuda_oom") == 0);
    cf_qualification_destroy(runtime);
    return 0;
}
