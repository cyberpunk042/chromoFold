#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum cf_slo_class { CF_SLO_INTERACTIVE=0, CF_SLO_STANDARD=1, CF_SLO_BATCH=2, CF_SLO_BACKGROUND=3 } cf_slo_class;
typedef enum cf_admission_result { CF_ADMIT_NOW=0, CF_ADMIT_QUEUE=1, CF_REJECT_OVERLOAD=2, CF_REJECT_QUOTA=3, CF_REJECT_MODEL=4 } cf_admission_result;
typedef enum cf_worker_state { CF_WORKER_HEALTHY=0, CF_WORKER_DEGRADED=1, CF_WORKER_DRAINING=2, CF_WORKER_QUARANTINED=3, CF_WORKER_FAILED=4 } cf_worker_state;
typedef enum cf_circuit_state { CF_CIRCUIT_CLOSED=0, CF_CIRCUIT_OPEN=1, CF_CIRCUIT_HALF_OPEN=2 } cf_circuit_state;

typedef struct cf_scheduler_config {
    uint32_t version;
    uint32_t max_tenants;
    uint32_t max_workers;
    uint32_t max_queued_requests;
    uint32_t max_active_requests;
    uint64_t routing_seed;
    uint64_t overload_queue_threshold;
} cf_scheduler_config;

typedef struct cf_tenant_policy {
    uint64_t tenant_id;
    uint32_t weight;
    uint32_t max_concurrent_requests;
    uint32_t max_queued_requests;
    uint64_t prompt_tokens_per_minute;
    uint64_t completion_tokens_per_minute;
    uint64_t max_kv_bytes;
    uint64_t gpu_time_budget_micros;
} cf_tenant_policy;

typedef struct cf_worker_capacity {
    uint64_t worker_id;
    uint32_t state;
    uint32_t gpu_arch;
    uint64_t free_vram_bytes;
    uint64_t total_vram_bytes;
    uint64_t prefill_tokens_per_second;
    uint64_t decode_tokens_per_second;
    uint32_t active_sequences;
    uint32_t queue_depth;
    uint64_t model_mask;
    uint32_t circuit_state;
} cf_worker_capacity;

typedef struct cf_request_spec {
    uint64_t request_id;
    uint64_t tenant_id;
    uint64_t model_id;
    uint32_t slo_class;
    uint32_t priority;
    uint32_t prompt_tokens;
    uint32_t max_completion_tokens;
    uint32_t cached_prefix_tokens;
    uint64_t deadline_millis;
    uint64_t idempotency_key;
} cf_request_spec;

typedef struct cf_schedule_decision {
    uint32_t result;
    uint64_t worker_id;
    uint64_t estimated_prefill_micros;
    uint64_t estimated_decode_micros;
    uint64_t estimated_kv_bytes;
    uint64_t queue_position;
    uint64_t fairness_score;
} cf_schedule_decision;

typedef struct cf_scale_plan {
    uint32_t desired_prefill_workers;
    uint32_t desired_decode_workers;
    uint32_t desired_model_replicas;
    uint32_t drain_worker_count;
} cf_scale_plan;

typedef struct cf_scheduler_counters {
    uint64_t admissions;
    uint64_t queued;
    uint64_t overload_rejections;
    uint64_t quota_rejections;
    uint64_t model_rejections;
    uint64_t deadline_schedules;
    uint64_t fairness_schedules;
    uint64_t cross_tenant_reuse_rejections;
    uint64_t cancellations;
    uint64_t preemptions;
    uint64_t checkpoint_resumes;
    uint64_t circuit_opens;
    uint64_t circuit_recoveries;
    uint64_t scale_up_plans;
    uint64_t safe_drains;
    uint64_t duplicate_usage_records;
    uint64_t cross_tenant_contamination;
    uint64_t dense_fallback_launches;
    uint64_t active_refs_at_shutdown;
} cf_scheduler_counters;

typedef struct cf_scheduler_runtime cf_scheduler_runtime;
cf_scheduler_runtime * cf_scheduler_create(const cf_scheduler_config * config);
void cf_scheduler_destroy(cf_scheduler_runtime * runtime);
int cf_scheduler_register_tenant(cf_scheduler_runtime * runtime, const cf_tenant_policy * policy);
int cf_scheduler_register_worker(cf_scheduler_runtime * runtime, const cf_worker_capacity * worker);
int cf_scheduler_admit(cf_scheduler_runtime * runtime, const cf_request_spec * request, cf_schedule_decision * decision);
int cf_scheduler_complete(cf_scheduler_runtime * runtime, uint64_t request_id, uint32_t generated_tokens, uint64_t gpu_micros);
int cf_scheduler_cancel(cf_scheduler_runtime * runtime, uint64_t request_id);
int cf_scheduler_open_circuit(cf_scheduler_runtime * runtime, uint64_t worker_id);
int cf_scheduler_recover_circuit(cf_scheduler_runtime * runtime, uint64_t worker_id);
int cf_scheduler_plan_scale(const cf_scheduler_runtime * runtime, cf_scale_plan * plan);
int cf_scheduler_drain_worker(cf_scheduler_runtime * runtime, uint64_t worker_id);
int cf_scheduler_shutdown_audit(cf_scheduler_runtime * runtime);
int cf_scheduler_get_counters(const cf_scheduler_runtime * runtime, cf_scheduler_counters * counters);
const char * cf_scheduler_last_error(const cf_scheduler_runtime * runtime);
#ifdef __cplusplus
}
#endif
