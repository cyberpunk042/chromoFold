#include "chromofold/multisequence_cuda_resolver.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

void cuda_check(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct Buffer {
    void * ptr = nullptr;
    size_t bytes = 0;
    Buffer() = default;
    Buffer(const Buffer &) = delete;
    Buffer & operator=(const Buffer &) = delete;
    Buffer(Buffer && other) noexcept : ptr(other.ptr), bytes(other.bytes) { other.ptr = nullptr; other.bytes = 0; }
    Buffer & operator=(Buffer && other) noexcept {
        if (this != &other) { ptr = other.ptr; bytes = other.bytes; other.ptr = nullptr; other.bytes = 0; }
        return *this;
    }
    void allocate(size_t requested, cudaStream_t stream, bool async_allocator) {
        if (ptr && requested == bytes) return;
        release(stream, async_allocator);
        if (!requested) return;
        if (async_allocator) cuda_check(cudaMallocAsync(&ptr, requested, stream), "cudaMallocAsync");
        else cuda_check(cudaMalloc(&ptr, requested), "cudaMalloc");
        bytes = requested;
    }
    void release(cudaStream_t stream, bool async_allocator) noexcept {
        if (!ptr) return;
        if (async_allocator) cudaFreeAsync(ptr, stream); else cudaFree(ptr);
        ptr = nullptr; bytes = 0;
    }
};

struct SharedPage {
    uint64_t id = 0;
    uint64_t seal_generation = 0;
    cf_device_page_descriptor descriptor{};
    uint64_t resident_bytes = 0;
    uint32_t logical_refs = 0;
    uint32_t in_flight_refs = 0;
    bool retired = false;
    cudaEvent_t retirement_event = nullptr;
};

struct PageRef {
    uint64_t page_id = 0;
    int64_t logical_token_begin = 0;
};

struct ActivePage {
    Buffer keys;
    Buffer values;
    uint32_t token_begin = 0;
    uint32_t token_count = 0;
    uint64_t generation = 0;
};

struct LayerState {
    std::vector<PageRef> pages;
    Buffer descriptors;
    uint32_t descriptor_capacity = 0;
    uint64_t descriptor_generation = 0;
    ActivePage active;
};

struct SequenceState {
    int64_t id = 0;
    std::vector<LayerState> layers;
    uint64_t topology_generation = 1;
    bool dispatchable = true;
};

struct Snapshot {
    int64_t sequence_id = 0;
    uint32_t layer = 0;
    std::vector<uint64_t> page_ids;
    const cf_device_page_descriptor * descriptors = nullptr;
    uint32_t descriptor_count = 0;
    const float * active_k = nullptr;
    const float * active_v = nullptr;
    uint32_t active_begin = 0;
    uint32_t active_count = 0;
    uint64_t topology_generation = 0;
    uint64_t descriptor_generation = 0;
    uint64_t active_generation = 0;
};

struct PendingSnapshot {
    Snapshot snapshot;
    cudaEvent_t event = nullptr;
};

size_t scalar_bytes(uint32_t type) {
    if (type == CF_DEVICE_F16 || type == CF_DEVICE_BF16) return 2;
    if (type == CF_DEVICE_F32) return 4;
    return 0;
}

__device__ float query_load(const cf_device_tensor_view & view, uint32_t token, uint32_t head, uint32_t dim) {
    const char * p = static_cast<const char *>(view.data) + static_cast<uint64_t>(token) * view.token_stride_bytes +
                     static_cast<uint64_t>(head) * view.head_stride_bytes + static_cast<uint64_t>(dim) * view.element_stride_bytes;
    if (view.scalar_type == CF_DEVICE_F16) return __half2float(*reinterpret_cast<const __half *>(p));
    if (view.scalar_type == CF_DEVICE_BF16) return __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(p));
    return *reinterpret_cast<const float *>(p);
}

__device__ void output_store(void * output, uint32_t type, uint64_t offset, float value) {
    char * p = static_cast<char *>(output) + offset;
    if (type == CF_DEVICE_F16) *reinterpret_cast<__half *>(p) = __float2half(value);
    else if (type == CF_DEVICE_BF16) *reinterpret_cast<__nv_bfloat16 *>(p) = __float2bfloat16(value);
    else *reinterpret_cast<float *>(p) = value;
}

__device__ float unpack_value(const uint8_t * payload, const float * scales, uint64_t index, uint32_t block_size) {
    const uint8_t packed = payload[index / 2];
    const uint8_t symbol = (index & 1U) ? (packed >> 4) : (packed & 0x0fU);
    return (static_cast<int>(symbol) - 8) * scales[index / block_size];
}

__global__ void sequence_attention_kernel(cf_device_tensor_view query,
                                          void * output,
                                          uint32_t output_type,
                                          uint64_t output_token_stride,
                                          uint64_t output_head_stride,
                                          uint64_t output_element_stride,
                                          const cf_device_page_descriptor * pages,
                                          uint32_t page_count,
                                          const float * active_k,
                                          const float * active_v,
                                          uint32_t active_begin,
                                          uint32_t active_count,
                                          uint32_t query_begin,
                                          uint32_t query_heads,
                                          uint32_t kv_heads,
                                          uint32_t gqa,
                                          uint32_t head_dim,
                                          float softmax_scale,
                                          uint32_t causal_window) {
    const uint32_t qh = blockIdx.x;
    const uint32_t qt = blockIdx.y;
    if (qh >= query_heads || qt >= query.token_count || threadIdx.x != 0) return;
    const uint32_t kvh = qh / gqa;
    const uint32_t qpos = query_begin + qt;
    float max_score = -INFINITY;
    float denominator = 0.0f;
    extern __shared__ float output_acc[];
    for (uint32_t d = 0; d < head_dim; ++d) output_acc[d] = 0.0f;

    auto consume = [&](uint32_t token_pos, const cf_device_page_descriptor * page, uint32_t token, bool active) {
        if (token_pos > qpos) return;
        if (causal_window && token_pos + causal_window <= qpos) return;
        float score = 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) {
            const uint64_t index = (static_cast<uint64_t>(token) * kv_heads + kvh) * head_dim + d;
            const float key = active ? active_k[index] : unpack_value(page->key_payload, page->key_scales, index, page->block_size);
            score += query_load(query, qt, qh, d) * key;
        }
        score *= softmax_scale;
        const float next_max = fmaxf(max_score, score);
        const float old_scale = isfinite(max_score) ? expf(max_score - next_max) : 0.0f;
        const float new_scale = expf(score - next_max);
        for (uint32_t d = 0; d < head_dim; ++d) {
            const uint64_t index = (static_cast<uint64_t>(token) * kv_heads + kvh) * head_dim + d;
            const float value = active ? active_v[index] : unpack_value(page->value_payload, page->value_scales, index, page->block_size);
            output_acc[d] = output_acc[d] * old_scale + value * new_scale;
        }
        denominator = denominator * old_scale + new_scale;
        max_score = next_max;
    };

    for (uint32_t page_index = 0; page_index < page_count; ++page_index) {
        const cf_device_page_descriptor * page = &pages[page_index];
        for (uint32_t token = 0; token < page->token_count; ++token) consume(page->token_begin + token, page, token, false);
    }
    for (uint32_t token = 0; token < active_count; ++token) consume(active_begin + token, nullptr, token, true);

    for (uint32_t d = 0; d < head_dim; ++d) {
        const uint64_t offset = static_cast<uint64_t>(qt) * output_token_stride +
                                static_cast<uint64_t>(qh) * output_head_stride +
                                static_cast<uint64_t>(d) * output_element_stride;
        output_store(output, output_type, offset, denominator > 0.0f ? output_acc[d] / denominator : 0.0f);
    }
}

} // namespace

struct cf_ms_cuda_resolver {
    cf_ms_cuda_config config{};
    mutable std::mutex mutex;
    std::map<int64_t, SequenceState> sequences;
    std::unordered_map<uint64_t, SharedPage> pages;
    std::vector<PendingSnapshot> pending_snapshots;
    uint64_t next_page_id = 1;
    cf_ms_cuda_counters counters{};
    std::string error;
};

namespace {

void refresh_descriptor_table(cf_ms_cuda_resolver * resolver, SequenceState & sequence, uint32_t layer_index, cudaStream_t stream) {
    auto & layer = sequence.layers[layer_index];
    std::vector<cf_device_page_descriptor> host;
    host.reserve(layer.pages.size());
    for (const auto & ref : layer.pages) {
        auto page = resolver->pages.find(ref.page_id);
        if (page == resolver->pages.end() || page->second.retired) throw std::runtime_error("sequence references unavailable shared page");
        cf_device_page_descriptor descriptor = page->second.descriptor;
        descriptor.token_begin = static_cast<uint32_t>(ref.logical_token_begin);
        host.push_back(descriptor);
    }
    const bool async_allocator = resolver->config.use_async_allocator != 0;
    if (host.size() > layer.descriptor_capacity) {
        uint32_t capacity = std::max<uint32_t>(4, layer.descriptor_capacity ? layer.descriptor_capacity * 2 : 4);
        while (capacity < host.size()) capacity *= 2;
        layer.descriptors.allocate(static_cast<size_t>(capacity) * sizeof(cf_device_page_descriptor), stream, async_allocator);
        layer.descriptor_capacity = capacity;
        ++resolver->counters.descriptor_table_rebuilds;
    }
    if (!host.empty()) cuda_check(cudaMemcpyAsync(layer.descriptors.ptr, host.data(), host.size() * sizeof(cf_device_page_descriptor), cudaMemcpyHostToDevice, stream), "publish sequence descriptors");
    ++layer.descriptor_generation;
    ++resolver->counters.descriptor_tables_built;
}

Snapshot acquire_snapshot(cf_ms_cuda_resolver * resolver, int64_t sequence_id, uint32_t layer_index) {
    auto sequence_it = resolver->sequences.find(sequence_id);
    if (sequence_it == resolver->sequences.end() || !sequence_it->second.dispatchable) throw std::runtime_error("sequence is unavailable for dispatch");
    if (layer_index >= sequence_it->second.layers.size()) throw std::runtime_error("layer out of range");
    auto & layer = sequence_it->second.layers[layer_index];
    Snapshot snapshot;
    snapshot.sequence_id = sequence_id;
    snapshot.layer = layer_index;
    snapshot.descriptors = static_cast<const cf_device_page_descriptor *>(layer.descriptors.ptr);
    snapshot.descriptor_count = static_cast<uint32_t>(layer.pages.size());
    snapshot.active_k = static_cast<const float *>(layer.active.keys.ptr);
    snapshot.active_v = static_cast<const float *>(layer.active.values.ptr);
    snapshot.active_begin = layer.active.token_begin;
    snapshot.active_count = layer.active.token_count;
    snapshot.topology_generation = sequence_it->second.topology_generation;
    snapshot.descriptor_generation = layer.descriptor_generation;
    snapshot.active_generation = layer.active.generation;
    for (const auto & ref : layer.pages) {
        auto & page = resolver->pages.at(ref.page_id);
        ++page.in_flight_refs;
        ++resolver->counters.in_flight_page_references;
        snapshot.page_ids.push_back(ref.page_id);
    }
    ++resolver->counters.snapshots_acquired;
    return snapshot;
}

void release_snapshot_refs(cf_ms_cuda_resolver * resolver, const Snapshot & snapshot) {
    for (uint64_t page_id : snapshot.page_ids) {
        auto page = resolver->pages.find(page_id);
        if (page == resolver->pages.end() || page->second.in_flight_refs == 0) throw std::runtime_error("snapshot reference underflow");
        --page->second.in_flight_refs;
        --resolver->counters.in_flight_page_references;
    }
    ++resolver->counters.snapshots_released;
}

} // namespace

extern "C" cf_ms_cuda_resolver * cf_ms_cuda_create(const cf_ms_cuda_config * config) {
    if (!config || !config->layer_count || !config->kv_head_count || !config->query_head_count || !config->head_dim ||
        !config->page_size || !config->gqa_group_size || config->query_head_count != config->kv_head_count * config->gqa_group_size) return nullptr;
    try { auto resolver = std::make_unique<cf_ms_cuda_resolver>(); resolver->config = *config; return resolver.release(); }
    catch (...) { return nullptr; }
}

extern "C" void cf_ms_cuda_destroy(cf_ms_cuda_resolver * resolver, void * stream_handle) {
    if (!resolver) return;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    const bool async_allocator = resolver->config.use_async_allocator != 0;
    cudaStreamSynchronize(stream);
    for (auto & pending : resolver->pending_snapshots) { if (pending.event) cudaEventDestroy(pending.event); }
    for (auto & [_, sequence] : resolver->sequences) for (auto & layer : sequence.layers) {
        layer.descriptors.release(stream, async_allocator); layer.active.keys.release(stream, async_allocator); layer.active.values.release(stream, async_allocator);
    }
    for (auto & [_, page] : resolver->pages) if (page.retirement_event) cudaEventDestroy(page.retirement_event);
    delete resolver;
}

extern "C" int cf_ms_cuda_sequence_create(cf_ms_cuda_resolver * resolver, int64_t sequence_id) {
    if (!resolver) return -1;
    std::lock_guard<std::mutex> guard(resolver->mutex);
    try {
        if (resolver->sequences.count(sequence_id)) throw std::runtime_error("sequence already exists");
        SequenceState state; state.id = sequence_id; state.layers.resize(resolver->config.layer_count);
        resolver->sequences.emplace(sequence_id, std::move(state)); resolver->error.clear(); return 0;
    } catch (const std::exception & error) { resolver->error = error.what(); return -1; }
}

extern "C" int cf_ms_cuda_publish_page_async(cf_ms_cuda_resolver * resolver, const cf_ms_page_publication * publication, void * stream_handle) {
    if (!resolver || !publication) return -1;
    std::lock_guard<std::mutex> guard(resolver->mutex);
    try {
        auto sequence = resolver->sequences.find(publication->sequence_id);
        if (sequence == resolver->sequences.end() || publication->layer >= sequence->second.layers.size()) throw std::runtime_error("publication target is invalid");
        if (!publication->descriptor.key_payload || !publication->descriptor.value_payload || !publication->descriptor.key_scales || !publication->descriptor.value_scales)
            throw std::runtime_error("publication descriptor is incomplete");
        SharedPage page; page.id = resolver->next_page_id++; page.seal_generation = publication->seal_generation;
        page.descriptor = publication->descriptor; page.resident_bytes = publication->resident_bytes; page.logical_refs = 1;
        resolver->pages.emplace(page.id, page);
        auto & layer = sequence->second.layers[publication->layer];
        layer.pages.push_back({page.id, publication->descriptor.token_begin});
        refresh_descriptor_table(resolver, sequence->second, publication->layer, static_cast<cudaStream_t>(stream_handle));
        ++sequence->second.topology_generation; ++resolver->counters.pages_registered; resolver->error.clear(); return 0;
    } catch (const std::exception & error) { ++resolver->counters.page_registration_failures; resolver->error = error.what(); return -1; }
}

extern "C" int cf_ms_cuda_stage_active_async(cf_ms_cuda_resolver * resolver, int64_t sequence_id, uint32_t layer_index,
                                               uint32_t token_begin, const float * keys, const float * values,
                                               uint32_t token_count, void * stream_handle) {
    if (!resolver || !keys || !values || !token_count || token_count > resolver->config.page_size) return -1;
    std::lock_guard<std::mutex> guard(resolver->mutex);
    try {
        auto sequence = resolver->sequences.find(sequence_id);
        if (sequence == resolver->sequences.end() || layer_index >= sequence->second.layers.size()) throw std::runtime_error("active target is invalid");
        auto & active = sequence->second.layers[layer_index].active;
        cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
        const bool async_allocator = resolver->config.use_async_allocator != 0;
        const size_t page_values = static_cast<size_t>(resolver->config.page_size) * resolver->config.kv_head_count * resolver->config.head_dim;
        active.keys.allocate(page_values * sizeof(float), stream, async_allocator);
        active.values.allocate(page_values * sizeof(float), stream, async_allocator);
        const size_t used = static_cast<size_t>(token_count) * resolver->config.kv_head_count * resolver->config.head_dim * sizeof(float);
        cuda_check(cudaMemcpyAsync(active.keys.ptr, keys, used, cudaMemcpyDeviceToDevice, stream), "stage sequence active K");
        cuda_check(cudaMemcpyAsync(active.values.ptr, values, used, cudaMemcpyDeviceToDevice, stream), "stage sequence active V");
        active.token_begin = token_begin; active.token_count = token_count; ++active.generation; resolver->error.clear(); return 0;
    } catch (const std::exception & error) { ++resolver->counters.cuda_errors; resolver->error = error.what(); return -1; }
}

extern "C" int cf_ms_cuda_copy_sequence_async(cf_ms_cuda_resolver * resolver, int64_t source_id, int64_t destination_id, void * stream_handle) {
    if (!resolver || source_id == destination_id) return -1;
    std::lock_guard<std::mutex> guard(resolver->mutex);
    try {
        auto source = resolver->sequences.find(source_id); if (source == resolver->sequences.end()) throw std::runtime_error("source sequence is missing");
        SequenceState candidate; candidate.id = destination_id; candidate.layers.resize(resolver->config.layer_count);
        cudaStream_t stream = static_cast<cudaStream_t>(stream_handle); const bool async_allocator = resolver->config.use_async_allocator != 0;
        for (uint32_t layer_index = 0; layer_index < resolver->config.layer_count; ++layer_index) {
            const auto & source_layer = source->second.layers[layer_index]; auto & target = candidate.layers[layer_index];
            target.pages = source_layer.pages;
            for (const auto & ref : target.pages) { ++resolver->pages.at(ref.page_id).logical_refs; }
            const auto & source_active = source_layer.active;
            if (source_active.token_count) {
                const size_t bytes = static_cast<size_t>(resolver->config.page_size) * resolver->config.kv_head_count * resolver->config.head_dim * sizeof(float);
                target.active.keys.allocate(bytes, stream, async_allocator); target.active.values.allocate(bytes, stream, async_allocator);
                cuda_check(cudaMemcpyAsync(target.active.keys.ptr, source_active.keys.ptr, bytes, cudaMemcpyDeviceToDevice, stream), "copy active K");
                cuda_check(cudaMemcpyAsync(target.active.values.ptr, source_active.values.ptr, bytes, cudaMemcpyDeviceToDevice, stream), "copy active V");
                target.active.token_begin = source_active.token_begin; target.active.token_count = source_active.token_count; target.active.generation = source_active.generation + 1;
                ++resolver->counters.copy_on_write_tail_copies;
            }
            refresh_descriptor_table(resolver, candidate, layer_index, stream);
        }
        auto old = resolver->sequences.find(destination_id);
        if (old != resolver->sequences.end()) old->second.dispatchable = false;
        resolver->sequences[destination_id] = std::move(candidate); resolver->error.clear(); return 0;
    } catch (const std::exception & error) { ++resolver->counters.cuda_errors; resolver->error = error.what(); return -1; }
}

extern "C" int cf_ms_cuda_execute_batch_async(cf_ms_cuda_resolver * resolver, const cf_ms_batch_item * items, uint32_t item_count, void * stream_handle) {
    if (!resolver || !items || !item_count) return -1;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    std::vector<Snapshot> snapshots;
    try {
        {
            std::lock_guard<std::mutex> guard(resolver->mutex);
            snapshots.reserve(item_count);
            for (uint32_t i = 0; i < item_count; ++i) {
                if (cf_device_tensor_validate(&items[i].query) != 0 || !items[i].output || !scalar_bytes(items[i].output_type)) throw std::runtime_error("invalid batch tensor");
                snapshots.push_back(acquire_snapshot(resolver, items[i].sequence_id, items[i].layer));
            }
            ++resolver->counters.server_batches_received;
            resolver->counters.tokens_per_batch_peak = std::max<uint64_t>(resolver->counters.tokens_per_batch_peak, item_count);
        }
        for (uint32_t i = 0; i < item_count; ++i) {
            const auto & item = items[i]; const auto & snapshot = snapshots[i];
            dim3 grid(item.query.head_count, item.query.token_count);
            sequence_attention_kernel<<<grid, 1, resolver->config.head_dim * sizeof(float), stream>>>(item.query, item.output, item.output_type,
                item.output_token_stride_bytes, item.output_head_stride_bytes, item.output_element_stride_bytes,
                snapshot.descriptors, snapshot.descriptor_count, snapshot.active_k, snapshot.active_v, snapshot.active_begin, snapshot.active_count,
                item.query_token_begin, resolver->config.query_head_count, resolver->config.kv_head_count, resolver->config.gqa_group_size,
                resolver->config.head_dim, item.softmax_scale, item.causal_window);
            cuda_check(cudaGetLastError(), "multi-sequence attention launch");
            ++resolver->counters.sequence_attention_launches;
            if (item.query.token_count == 1) ++resolver->counters.decode_items; else ++resolver->counters.prefill_items;
        }
        {
            std::lock_guard<std::mutex> guard(resolver->mutex);
            for (auto & snapshot : snapshots) {
                PendingSnapshot pending; pending.snapshot = std::move(snapshot);
                cuda_check(cudaEventCreateWithFlags(&pending.event, cudaEventDisableTiming), "create snapshot release event");
                cuda_check(cudaEventRecord(pending.event, stream), "record snapshot release event");
                resolver->pending_snapshots.push_back(std::move(pending)); ++resolver->counters.snapshot_release_events;
            }
            ++resolver->counters.compressed_batches_executed; resolver->error.clear();
        }
        return 0;
    } catch (const std::exception & error) {
        std::lock_guard<std::mutex> guard(resolver->mutex);
        for (const auto & snapshot : snapshots) { try { release_snapshot_refs(resolver, snapshot); } catch (...) {} }
        ++resolver->counters.cuda_errors; resolver->error = error.what(); return -1;
    }
}

extern "C" int cf_ms_cuda_cancel_sequence_async(cf_ms_cuda_resolver * resolver, int64_t sequence_id, void *) {
    if (!resolver) return -1;
    std::lock_guard<std::mutex> guard(resolver->mutex);
    auto sequence = resolver->sequences.find(sequence_id); if (sequence == resolver->sequences.end()) return -1;
    sequence->second.dispatchable = false; ++resolver->counters.cancellation_requests; return 0;
}

extern "C" int cf_ms_cuda_sequence_remove_async(cf_ms_cuda_resolver * resolver, int64_t sequence_id, void * stream_handle) {
    if (!resolver) return -1;
    std::lock_guard<std::mutex> guard(resolver->mutex);
    try {
        auto sequence = resolver->sequences.find(sequence_id); if (sequence == resolver->sequences.end()) throw std::runtime_error("sequence is missing");
        cudaStream_t stream = static_cast<cudaStream_t>(stream_handle); const bool async_allocator = resolver->config.use_async_allocator != 0;
        for (auto & layer : sequence->second.layers) {
            for (const auto & ref : layer.pages) {
                auto & page = resolver->pages.at(ref.page_id); if (!page.logical_refs) throw std::runtime_error("page logical reference underflow");
                --page.logical_refs; if (!page.logical_refs) { page.retired = true; ++resolver->counters.pages_retired; resolver->counters.pending_reclamation_bytes += page.resident_bytes; resolver->counters.pending_reclamation_bytes_peak = std::max(resolver->counters.pending_reclamation_bytes_peak, resolver->counters.pending_reclamation_bytes); }
            }
            layer.descriptors.release(stream, async_allocator); layer.active.keys.release(stream, async_allocator); layer.active.values.release(stream, async_allocator);
        }
        resolver->sequences.erase(sequence); return 0;
    } catch (const std::exception & error) { resolver->error = error.what(); return -1; }
}

extern "C" int cf_ms_cuda_reclaim_async(cf_ms_cuda_resolver * resolver, void *) {
    if (!resolver) return -1;
    std::lock_guard<std::mutex> guard(resolver->mutex);
    try {
        for (auto it = resolver->pending_snapshots.begin(); it != resolver->pending_snapshots.end();) {
            const cudaError_t status = cudaEventQuery(it->event);
            if (status == cudaErrorNotReady) { ++it; continue; }
            cuda_check(status, "query snapshot release event"); release_snapshot_refs(resolver, it->snapshot); cudaEventDestroy(it->event); it = resolver->pending_snapshots.erase(it);
        }
        for (auto it = resolver->pages.begin(); it != resolver->pages.end();) {
            if (it->second.retired && it->second.logical_refs == 0 && it->second.in_flight_refs == 0) {
                resolver->counters.pending_reclamation_bytes -= it->second.resident_bytes; ++resolver->counters.pages_reclaimed; it = resolver->pages.erase(it);
            } else ++it;
        }
        return 0;
    } catch (const std::exception & error) { ++resolver->counters.cuda_errors; resolver->error = error.what(); return -1; }
}

extern "C" int cf_ms_cuda_shift_sequence(cf_ms_cuda_resolver * resolver, int64_t sequence_id, uint32_t begin, uint32_t end, int32_t delta) {
    if (!resolver || end < begin) return -1;
    std::lock_guard<std::mutex> guard(resolver->mutex);
    try {
        auto sequence = resolver->sequences.find(sequence_id); if (sequence == resolver->sequences.end()) throw std::runtime_error("sequence is missing");
        for (auto & layer : sequence->second.layers) for (auto & ref : layer.pages) {
            const int64_t ref_end = ref.logical_token_begin + resolver->config.page_size;
            if (ref.logical_token_begin >= begin && ref_end <= end) ref.logical_token_begin += delta;
            else if (ref.logical_token_begin < end && ref_end > begin) throw std::runtime_error("shift intersects part of an immutable page");
        }
        ++sequence->second.topology_generation; return 0;
    } catch (const std::exception & error) { resolver->error = error.what(); return -1; }
}

extern "C" int cf_ms_cuda_audit(const cf_ms_cuda_resolver * resolver) {
    if (!resolver) return -1;
    std::lock_guard<std::mutex> guard(resolver->mutex);
    std::unordered_map<uint64_t, uint64_t> observed;
    for (const auto & [id, sequence] : resolver->sequences) {
        if (id != sequence.id) return -1;
        for (const auto & layer : sequence.layers) for (const auto & ref : layer.pages) {
            auto page = resolver->pages.find(ref.page_id); if (page == resolver->pages.end() || page->second.retired) return -1; ++observed[ref.page_id];
        }
    }
    for (const auto & [id, page] : resolver->pages) if (!page.retired && page.logical_refs != observed[id]) return -1;
    return resolver->counters.cross_sequence_contamination == 0 && resolver->counters.dense_fallback_launches == 0 ? 0 : -1;
}

extern "C" int cf_ms_cuda_get_counters(const cf_ms_cuda_resolver * resolver, cf_ms_cuda_counters * counters) {
    if (!resolver || !counters) return -1; std::lock_guard<std::mutex> guard(resolver->mutex); *counters = resolver->counters; return 0;
}

extern "C" const char * cf_ms_cuda_last_error(const cf_ms_cuda_resolver * resolver) {
    return resolver ? resolver->error.c_str() : "null multi-sequence CUDA resolver";
}
