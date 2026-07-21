#include "chromofold/production_scheduler.h"
#include <cassert>

int main() {
    cf_scheduler_config config{1, 8, 8, 8, 2, 42, 0};
    cf_scheduler_runtime * runtime = cf_scheduler_create(&config);
    assert(runtime);

    cf_tenant_policy tenant1{1, 1, 1, 2, 1000, 1000, 1ull << 30, 10000000};
    cf_tenant_policy tenant2{2, 2, 1, 2, 1000, 1000, 1ull << 30, 10000000};
    assert(cf_scheduler_register_tenant(runtime, &tenant1) == 0);
    assert(cf_scheduler_register_tenant(runtime, &tenant2) == 0);

    cf_worker_capacity worker1{10, CF_WORKER_HEALTHY, 80, 12ull << 30, 16ull << 30, 1000, 200, 0, 0, 1ull << 1, CF_CIRCUIT_CLOSED};
    cf_worker_capacity worker2{11, CF_WORKER_HEALTHY, 89, 20ull << 30, 24ull << 30, 1500, 300, 0, 0, 1ull << 1, CF_CIRCUIT_CLOSED};
    assert(cf_scheduler_register_worker(runtime, &worker1) == 0);
    assert(cf_scheduler_register_worker(runtime, &worker2) == 0);

    cf_request_spec request1{100, 1, 1, CF_SLO_INTERACTIVE, 10, 128, 32, 64, 1000, 700};
    cf_schedule_decision decision1{};
    assert(cf_scheduler_admit(runtime, &request1, &decision1) == 0);
    assert(decision1.result == CF_ADMIT_NOW && decision1.worker_id == 11);

    cf_request_spec request2{101, 1, 1, CF_SLO_STANDARD, 1, 128, 32, 0, 2000, 701};
    cf_schedule_decision decision2{};
    assert(cf_scheduler_admit(runtime, &request2, &decision2) == 0);
    assert(decision2.result == CF_ADMIT_QUEUE);

    cf_request_spec request3{102, 2, 1, CF_SLO_INTERACTIVE, 5, 64, 16, 0, 1000, 702};
    cf_schedule_decision decision3{};
    assert(cf_scheduler_admit(runtime, &request3, &decision3) == 0);
    assert(decision3.result == CF_ADMIT_NOW);

    assert(cf_scheduler_cancel(runtime, 101) == 0);
    assert(cf_scheduler_complete(runtime, 100, 24, 5000) == 0);
    assert(cf_scheduler_complete(runtime, 100, 24, 5000) != 0);
    assert(cf_scheduler_complete(runtime, 102, 12, 3000) == 0);

    assert(cf_scheduler_open_circuit(runtime, 10) == 0);
    assert(cf_scheduler_recover_circuit(runtime, 10) == 0);

    cf_scale_plan plan{};
    assert(cf_scheduler_plan_scale(runtime, &plan) == 0);
    assert(plan.desired_prefill_workers >= 1 && plan.desired_decode_workers >= 2);
    assert(cf_scheduler_drain_worker(runtime, 10) == 0);
    assert(cf_scheduler_shutdown_audit(runtime) == 0);

    cf_scheduler_counters counters{};
    assert(cf_scheduler_get_counters(runtime, &counters) == 0);
    assert(counters.admissions == 2 && counters.queued == 1);
    assert(counters.cancellations == 1);
    assert(counters.circuit_opens == 1 && counters.circuit_recoveries == 1);
    assert(counters.safe_drains == 1);
    assert(counters.cross_tenant_contamination == 0 && counters.dense_fallback_launches == 0);
    cf_scheduler_destroy(runtime);
}
