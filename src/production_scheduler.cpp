#include "chromofold/production_scheduler.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

struct TenantState {
    cf_tenant_policy policy{};
    uint32_t active = 0;
    uint32_t queued = 0;
    uint64_t prompt_tokens = 0;
    uint64_t completion_tokens = 0;
    uint64_t gpu_micros = 0;
    uint64_t virtual_runtime = 0;
};

struct RequestState {
    cf_request_spec spec{};
    uint64_t worker_id = 0;
    bool queued = false;
};

struct cf_scheduler_runtime {
    cf_scheduler_config config{};
    mutable std::mutex mutex;
    std::unordered_map<uint64_t, TenantState> tenants;
    std::unordered_map<uint64_t, cf_worker_capacity> workers;
    std::unordered_map<uint64_t, RequestState> requests;
    std::unordered_map<uint64_t, uint64_t> usage_records;
    cf_scheduler_counters counters{};
    bool shutting_down = false;
    std::string error;
};

namespace {

bool model_supported(const cf_worker_capacity & worker, uint64_t model_id) {
    if (model_id >= 64) return false;
    return (worker.model_mask & (uint64_t{1} << model_id)) != 0;
}

uint64_t estimate_prefill(const cf_request_spec & request, const cf_worker_capacity & worker) {
    const uint64_t uncached = request.prompt_tokens > request.cached_prefix_tokens
        ? request.prompt_tokens - request.cached_prefix_tokens
        : 0;
    const uint64_t rate = std::max<uint64_t>(1, worker.prefill_tokens_per_second);
    return uncached * 1000000ull / rate;
}

uint64_t estimate_decode(const cf_request_spec & request, const cf_worker_capacity & worker) {
    const uint64_t rate = std::max<uint64_t>(1, worker.decode_tokens_per_second);
    return static_cast<uint64_t>(request.max_completion_tokens) * 1000000ull / rate;
}

uint64_t fairness_score(const TenantState & tenant, const cf_request_spec & request) {
    const uint64_t weight = std::max<uint32_t>(1, tenant.policy.weight);
    const uint64_t priority_bonus = static_cast<uint64_t>(request.priority) * 1000ull;
    return tenant.virtual_runtime / weight > priority_bonus
        ? tenant.virtual_runtime / weight - priority_bonus
        : 0;
}

} // namespace

extern "C" cf_scheduler_runtime * cf_scheduler_create(const cf_scheduler_config * config) {
    if (!config || config->version != 1 || !config->max_tenants || !config->max_workers ||
        !config->max_queued_requests || !config->max_active_requests) {
        return nullptr;
    }
    auto * runtime = new cf_scheduler_runtime;
    runtime->config = *config;
    return runtime;
}

extern "C" void cf_scheduler_destroy(cf_scheduler_runtime * runtime) {
    delete runtime;
}

extern "C" int cf_scheduler_register_tenant(cf_scheduler_runtime * runtime, const cf_tenant_policy * policy) {
    if (!runtime || !policy || !policy->tenant_id || !policy->weight ||
        !policy->max_concurrent_requests || !policy->max_queued_requests) {
        return -1;
    }
    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (runtime->shutting_down || runtime->tenants.size() >= runtime->config.max_tenants ||
        runtime->tenants.count(policy->tenant_id) != 0) {
        return -1;
    }
    TenantState state{};
    state.policy = *policy;
    runtime->tenants.emplace(policy->tenant_id, state);
    return 0;
}

extern "C" int cf_scheduler_register_worker(cf_scheduler_runtime * runtime, const cf_worker_capacity * worker) {
    if (!runtime || !worker || !worker->worker_id || !worker->total_vram_bytes ||
        !worker->prefill_tokens_per_second || !worker->decode_tokens_per_second) {
        return -1;
    }
    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (runtime->shutting_down || runtime->workers.size() >= runtime->config.max_workers ||
        runtime->workers.count(worker->worker_id) != 0) {
        return -1;
    }
    runtime->workers.emplace(worker->worker_id, *worker);
    return 0;
}

extern "C" int cf_scheduler_admit(cf_scheduler_runtime * runtime,
                                    const cf_request_spec * request,
                                    cf_schedule_decision * decision) {
    if (!runtime || !request || !decision || !request->request_id || !request->tenant_id ||
        !request->prompt_tokens || !request->max_completion_tokens) {
        return -1;
    }
    std::lock_guard<std::mutex> guard(runtime->mutex);
    if (runtime->shutting_down || runtime->requests.count(request->request_id) != 0) return -1;
    auto tenant_it = runtime->tenants.find(request->tenant_id);
    if (tenant_it == runtime->tenants.end()) return -1;
    TenantState & tenant = tenant_it->second;

    if (tenant.prompt_tokens + request->prompt_tokens > tenant.policy.prompt_tokens_per_minute ||
        tenant.completion_tokens + request->max_completion_tokens > tenant.policy.completion_tokens_per_minute ||
        tenant.gpu_micros >= tenant.policy.gpu_time_budget_micros) {
        decision->result = CF_REJECT_QUOTA;
        ++runtime->counters.quota_rejections;
        return 0;
    }

    const cf_worker_capacity * selected = nullptr;
    uint64_t best_score = std::numeric_limits<uint64_t>::max();
    for (const auto & pair : runtime->workers) {
        const auto & worker = pair.second;
        if (worker.state >= CF_WORKER_DRAINING || worker.circuit_state == CF_CIRCUIT_OPEN ||
            !model_supported(worker, request->model_id)) {
            continue;
        }
        const uint64_t prefill = estimate_prefill(*request, worker);
        const uint64_t decode = estimate_decode(*request, worker);
        const uint64_t deadline_penalty = request->deadline_millis ? (prefill + decode) : 0;
        const uint64_t load_penalty = static_cast<uint64_t>(worker.queue_depth + worker.active_sequences) * 1000ull;
        const uint64_t score = deadline_penalty + load_penalty + fairness_score(tenant, *request);
        if (!selected || score < best_score || (score == best_score && worker.worker_id < selected->worker_id)) {
            selected = &worker;
            best_score = score;
        }
    }

    if (!selected) {
        decision->result = CF_REJECT_MODEL;
        ++runtime->counters.model_rejections;
        return 0;
    }

    uint64_t active_total = 0;
    uint64_t queued_total = 0;
    for (const auto & pair : runtime->tenants) {
        active_total += pair.second.active;
        queued_total += pair.second.queued;
    }

    const bool must_queue = active_total >= runtime->config.max_active_requests ||
        tenant.active >= tenant.policy.max_concurrent_requests;
    if (must_queue) {
        if (queued_total >= runtime->config.max_queued_requests ||
            tenant.queued >= tenant.policy.max_queued_requests) {
            decision->result = CF_REJECT_OVERLOAD;
            ++runtime->counters.overload_rejections;
            return 0;
        }
        ++tenant.queued;
        ++runtime->counters.queued;
    } else {
        ++tenant.active;
        ++runtime->counters.admissions;
    }

    RequestState state{};
    state.spec = *request;
    state.worker_id = selected->worker_id;
    state.queued = must_queue;
    runtime->requests.emplace(request->request_id, state);
    tenant.prompt_tokens += request->prompt_tokens;
    tenant.virtual_runtime += static_cast<uint64_t>(request->prompt_tokens) + request->max_completion_tokens;

    decision->result = must_queue ? CF_ADMIT_QUEUE : CF_ADMIT_NOW;
    decision->worker_id = selected->worker_id;
    decision->estimated_prefill_micros = estimate_prefill(*request, *selected);
    decision->estimated_decode_micros = estimate_decode(*request, *selected);
    decision->estimated_kv_bytes = static_cast<uint64_t>(request->prompt_tokens) * 1024ull;
    decision->queue_position = must_queue ? queued_total + 1 : 0;
    decision->fairness_score = fairness_score(tenant, *request);
    ++runtime->counters.deadline_schedules;
    ++runtime->counters.fairness_schedules;
    return 0;
}

extern "C" int cf_scheduler_complete(cf_scheduler_runtime * runtime,
                                       uint64_t request_id,
                                       uint32_t generated_tokens,
                                       uint64_t gpu_micros) {
    if (!runtime || !request_id) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    auto request_it = runtime->requests.find(request_id);
    if (request_it == runtime->requests.end()) return -1;
    if (runtime->usage_records.count(request_id) != 0) {
        ++runtime->counters.duplicate_usage_records;
        return -1;
    }
    auto tenant_it = runtime->tenants.find(request_it->second.spec.tenant_id);
    if (tenant_it == runtime->tenants.end()) return -1;
    TenantState & tenant = tenant_it->second;
    if (request_it->second.queued) {
        if (tenant.queued > 0) --tenant.queued;
    } else if (tenant.active > 0) {
        --tenant.active;
    }
    tenant.completion_tokens += generated_tokens;
    tenant.gpu_micros += gpu_micros;
    runtime->usage_records.emplace(request_id, gpu_micros);
    runtime->requests.erase(request_it);
    return 0;
}

extern "C" int cf_scheduler_cancel(cf_scheduler_runtime * runtime, uint64_t request_id) {
    if (!runtime || !request_id) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    auto request_it = runtime->requests.find(request_id);
    if (request_it == runtime->requests.end()) return -1;
    auto tenant_it = runtime->tenants.find(request_it->second.spec.tenant_id);
    if (tenant_it == runtime->tenants.end()) return -1;
    if (request_it->second.queued) {
        if (tenant_it->second.queued > 0) --tenant_it->second.queued;
    } else if (tenant_it->second.active > 0) {
        --tenant_it->second.active;
    }
    runtime->requests.erase(request_it);
    ++runtime->counters.cancellations;
    return 0;
}

extern "C" int cf_scheduler_open_circuit(cf_scheduler_runtime * runtime, uint64_t worker_id) {
    if (!runtime || !worker_id) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    auto found = runtime->workers.find(worker_id);
    if (found == runtime->workers.end()) return -1;
    found->second.circuit_state = CF_CIRCUIT_OPEN;
    found->second.state = CF_WORKER_QUARANTINED;
    ++runtime->counters.circuit_opens;
    return 0;
}

extern "C" int cf_scheduler_recover_circuit(cf_scheduler_runtime * runtime, uint64_t worker_id) {
    if (!runtime || !worker_id) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    auto found = runtime->workers.find(worker_id);
    if (found == runtime->workers.end() || found->second.circuit_state != CF_CIRCUIT_OPEN) return -1;
    found->second.circuit_state = CF_CIRCUIT_CLOSED;
    found->second.state = CF_WORKER_HEALTHY;
    ++runtime->counters.circuit_recoveries;
    return 0;
}

extern "C" int cf_scheduler_plan_scale(const cf_scheduler_runtime * runtime, cf_scale_plan * plan) {
    if (!runtime || !plan) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    uint64_t queued = 0;
    uint32_t healthy = 0;
    for (const auto & tenant : runtime->tenants) queued += tenant.second.queued;
    for (const auto & worker : runtime->workers) {
        if (worker.second.state == CF_WORKER_HEALTHY && worker.second.circuit_state == CF_CIRCUIT_CLOSED) ++healthy;
    }
    *plan = {};
    plan->desired_prefill_workers = std::max<uint32_t>(1, healthy / 2);
    plan->desired_decode_workers = std::max<uint32_t>(2, healthy - healthy / 2);
    plan->desired_model_replicas = queued > runtime->config.overload_queue_threshold ? 2 : 1;
    if (queued > runtime->config.overload_queue_threshold) {
        ++plan->desired_prefill_workers;
        ++plan->desired_decode_workers;
        const_cast<cf_scheduler_runtime *>(runtime)->counters.scale_up_plans++;
    }
    return 0;
}

extern "C" int cf_scheduler_drain_worker(cf_scheduler_runtime * runtime, uint64_t worker_id) {
    if (!runtime || !worker_id) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    auto found = runtime->workers.find(worker_id);
    if (found == runtime->workers.end()) return -1;
    for (const auto & request : runtime->requests) {
        if (request.second.worker_id == worker_id) return -1;
    }
    found->second.state = CF_WORKER_DRAINING;
    ++runtime->counters.safe_drains;
    return 0;
}

extern "C" int cf_scheduler_shutdown_audit(cf_scheduler_runtime * runtime) {
    if (!runtime) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    runtime->shutting_down = true;
    runtime->counters.active_refs_at_shutdown = runtime->requests.size();
    return runtime->requests.empty() && runtime->counters.cross_tenant_contamination == 0 &&
        runtime->counters.dense_fallback_launches == 0 ? 0 : -1;
}

extern "C" int cf_scheduler_get_counters(const cf_scheduler_runtime * runtime,
                                           cf_scheduler_counters * counters) {
    if (!runtime || !counters) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    *counters = runtime->counters;
    return 0;
}

extern "C" const char * cf_scheduler_last_error(const cf_scheduler_runtime * runtime) {
    return runtime ? runtime->error.c_str() : "runtime is null";
}
