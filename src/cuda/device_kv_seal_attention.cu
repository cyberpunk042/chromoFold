#include "chromofold/device_kv_seal_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
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
        if (requested == bytes && ptr) return;
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

struct Page {
    Buffer key_payload;
    Buffer value_payload;
    Buffer key_scales;
    Buffer value_scales;
    cf_device_page_descriptor descriptor{};
};

struct Layer {
    Buffer active_k;
    Buffer active_v;
    uint32_t active_begin = 0;
    uint32_t active_count = 0;
    uint32_t state = CF_SEAL_ACTIVE_FILLING;
    std::vector<Page> pages;
    Buffer device_descriptors;
    uint32_t descriptor_capacity = 0;
    uint64_t generation = 0;
};

__global__ void finite_scan(const float * values, uint64_t count, unsigned long long * nan_count, unsigned long long * inf_count) {
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float value = values[i];
    if (isnan(value)) atomicAdd(nan_count, 1ULL);
    if (isinf(value)) atomicAdd(inf_count, 1ULL);
}

__global__ void quantize_int4(const float * source, uint8_t * symbols, float * scales, uint64_t values, uint32_t block_size,
                              unsigned long long * saturation_count) {
    const uint64_t block = static_cast<uint64_t>(blockIdx.x);
    const uint64_t begin = block * block_size;
    if (begin >= values) return;
    __shared__ float shared_max[256];
    float local = 0.0f;
    for (uint64_t i = begin + threadIdx.x; i < min(begin + block_size, values); i += blockDim.x)
        local = fmaxf(local, fabsf(source[i]));
    shared_max[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        __syncthreads();
    }
    const float scale = shared_max[0] > 0.0f ? shared_max[0] / 7.0f : 1.0f;
    if (threadIdx.x == 0) scales[block] = scale;
    __syncthreads();
    for (uint64_t i = begin + threadIdx.x; i < min(begin + block_size, values); i += blockDim.x) {
        int q = __float2int_rn(source[i] / scale);
        if (q < -8) { q = -8; atomicAdd(saturation_count, 1ULL); }
        if (q > 7) { q = 7; atomicAdd(saturation_count, 1ULL); }
        symbols[i] = static_cast<uint8_t>(q + 8);
    }
}

__global__ void pack_nibbles(const uint8_t * symbols, uint8_t * payload, uint64_t values) {
    const uint64_t byte = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t first = byte * 2;
    if (first >= values) return;
    const uint8_t lo = symbols[first] & 0x0f;
    const uint8_t hi = first + 1 < values ? static_cast<uint8_t>((symbols[first + 1] & 0x0f) << 4) : 0;
    payload[byte] = lo | hi;
}

__device__ float unpack_value(const uint8_t * payload, const float * scales, uint64_t index, uint32_t block_size) {
    const uint8_t packed = payload[index / 2];
    const uint8_t symbol = (index & 1) ? (packed >> 4) : (packed & 0x0f);
    return (static_cast<int>(symbol) - 8) * scales[index / block_size];
}

__device__ float query_value(const cf_device_tensor_view view, uint32_t token, uint32_t head, uint32_t dim) {
    const char * p = static_cast<const char *>(view.data) + static_cast<uint64_t>(token) * view.token_stride_bytes +
                     static_cast<uint64_t>(head) * view.head_stride_bytes + static_cast<uint64_t>(dim) * view.element_stride_bytes;
    if (view.scalar_type == CF_DEVICE_F16) return __half2float(*reinterpret_cast<const __half *>(p));
    if (view.scalar_type == CF_DEVICE_BF16) return __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(p));
    return *reinterpret_cast<const float *>(p);
}

__device__ void store_output(void * output, uint32_t type, uint64_t offset, float value) {
    char * p = static_cast<char *>(output) + offset;
    if (type == CF_DEVICE_F16) *reinterpret_cast<__half *>(p) = __float2half(value);
    else if (type == CF_DEVICE_BF16) *reinterpret_cast<__nv_bfloat16 *>(p) = __float2bfloat16(value);
    else *reinterpret_cast<float *>(p) = value;
}

__global__ void reference_attention(cf_device_tensor_view query, void * output, uint32_t output_type,
                                    uint64_t out_token_stride, uint64_t out_head_stride, uint64_t out_element_stride,
                                    const cf_device_page_descriptor * pages, uint32_t page_count,
                                    const float * active_k, const float * active_v, uint32_t active_begin, uint32_t active_count,
                                    uint32_t query_begin, uint32_t query_heads, uint32_t kv_heads, uint32_t gqa,
                                    uint32_t head_dim, float softmax_scale, uint32_t causal_window) {
    const uint32_t qh = blockIdx.x;
    const uint32_t qt = blockIdx.y;
    if (qh >= query_heads || qt >= query.token_count) return;
    const uint32_t kvh = qh / gqa;
    const uint32_t qpos = query_begin + qt;
    float max_score = -CUDART_INF_F;
    float denom = 0.0f;
    extern __shared__ float acc[];
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) acc[d] = 0.0f;
    __syncthreads();

    auto consume = [&](uint32_t token_pos, auto key_at, auto value_at) {
        if (token_pos > qpos) return;
        if (causal_window && token_pos + causal_window <= qpos) return;
        float partial = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) partial += query_value(query, qt, qh, d) * key_at(d);
        __shared__ float reduce[256];
        reduce[threadIdx.x] = partial;
        __syncthreads();
        for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
            if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
            __syncthreads();
        }
        const float score = reduce[0] * softmax_scale;
        const float next_max = fmaxf(max_score, score);
        const float old_scale = isfinite(max_score) ? expf(max_score - next_max) : 0.0f;
        const float new_scale = expf(score - next_max);
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x)
            acc[d] = acc[d] * old_scale + value_at(d) * new_scale;
        if (threadIdx.x == 0) { denom = denom * old_scale + new_scale; max_score = next_max; }
        __syncthreads();
    };

    for (uint32_t p = 0; p < page_count; ++p) {
        const auto page = pages[p];
        for (uint32_t token = 0; token < page.token_count; ++token) {
            const uint64_t base = (static_cast<uint64_t>(token) * kv_heads + kvh) * head_dim;
            consume(page.token_begin + token,
                    [&](uint32_t d) { return unpack_value(page.key_payload, page.key_scales, base + d, page.block_size); },
                    [&](uint32_t d) { return unpack_value(page.value_payload, page.value_scales, base + d, page.block_size); });
        }
    }
    for (uint32_t token = 0; token < active_count; ++token) {
        const uint64_t base = (static_cast<uint64_t>(token) * kv_heads + kvh) * head_dim;
        consume(active_begin + token,
                [&](uint32_t d) { return active_k[base + d]; },
                [&](uint32_t d) { return active_v[base + d]; });
    }
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const uint64_t offset = static_cast<uint64_t>(qt) * out_token_stride + static_cast<uint64_t>(qh) * out_head_stride + static_cast<uint64_t>(d) * out_element_stride;
        store_output(output, output_type, offset, denom > 0.0f ? acc[d] / denom : 0.0f);
    }
}

size_t scalar_bytes(uint32_t type) {
    if (type == CF_DEVICE_F16 || type == CF_DEVICE_BF16) return 2;
    if (type == CF_DEVICE_F32) return 4;
    return 0;
}

} // namespace

struct cf_device_seal_engine {
    cf_device_seal_config config{};
    std::vector<Layer> layers;
    cf_device_seal_counters counters{};
    uint32_t fault = CF_SEAL_FAULT_NONE;
    std::string error;
};

extern "C" cf_device_seal_engine * cf_device_seal_create(const cf_device_seal_config * config) {
    if (!config || !config->layer_count || !config->kv_head_count || !config->query_head_count || !config->head_dim ||
        !config->page_size || !config->block_size || config->query_head_count != config->kv_head_count * config->gqa_group_size) return nullptr;
    try { auto e = std::make_unique<cf_device_seal_engine>(); e->config = *config; e->layers.resize(config->layer_count); return e.release(); }
    catch (...) { return nullptr; }
}

extern "C" void cf_device_seal_destroy(cf_device_seal_engine * e, void * stream_handle) {
    if (!e) return; cudaStream_t stream = static_cast<cudaStream_t>(stream_handle); const bool async = e->config.use_async_allocator != 0;
    for (auto & layer : e->layers) {
        layer.active_k.release(stream, async); layer.active_v.release(stream, async); layer.device_descriptors.release(stream, async);
        for (auto & page : layer.pages) { page.key_payload.release(stream, async); page.value_payload.release(stream, async); page.key_scales.release(stream, async); page.value_scales.release(stream, async); }
    }
    delete e;
}

extern "C" int cf_device_seal_stage_active_page(cf_device_seal_engine * e, uint32_t layer_index, uint32_t token_begin,
                                                  const float * k, const float * v, uint32_t token_count, void * stream_handle) {
    if (!e || layer_index >= e->layers.size() || !k || !v || !token_count || token_count > e->config.page_size) return -1;
    try {
        auto & layer = e->layers[layer_index]; cudaStream_t stream = static_cast<cudaStream_t>(stream_handle); const bool async = e->config.use_async_allocator != 0;
        const size_t values = static_cast<size_t>(e->config.page_size) * e->config.kv_head_count * e->config.head_dim;
        layer.active_k.allocate(values * sizeof(float), stream, async); layer.active_v.allocate(values * sizeof(float), stream, async);
        const size_t used = static_cast<size_t>(token_count) * e->config.kv_head_count * e->config.head_dim * sizeof(float);
        cuda_check(cudaMemcpyAsync(layer.active_k.ptr, k, used, cudaMemcpyDeviceToDevice, stream), "stage active K");
        cuda_check(cudaMemcpyAsync(layer.active_v.ptr, v, used, cudaMemcpyDeviceToDevice, stream), "stage active V");
        layer.active_begin = token_begin; layer.active_count = token_count;
        layer.state = token_count == e->config.page_size ? CF_SEAL_ACTIVE_FULL : CF_SEAL_ACTIVE_FILLING;
        e->error.clear(); return 0;
    } catch (const std::exception & ex) { e->error = ex.what(); e->counters.cuda_errors++; return -1; }
}

extern "C" int cf_device_seal_full_page_async(cf_device_seal_engine * e, uint32_t layer_index, void * stream_handle) {
    if (!e || layer_index >= e->layers.size()) return -1;
    auto & layer = e->layers[layer_index]; e->counters.seal_attempts++;
    if (layer.active_count != e->config.page_size || layer.state != CF_SEAL_ACTIVE_FULL) { e->error = "layer does not contain a full active page"; e->counters.seal_failures++; return -1; }
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle); const bool async = e->config.use_async_allocator != 0;
    Page candidate; Buffer symbols_k, symbols_v, invalid_counts, saturation;
    try {
        auto inject = [&](uint32_t point) { if (e->fault == point) throw std::runtime_error("injected transactional seal failure"); };
        inject(CF_SEAL_FAULT_CANDIDATE_ALLOC);
        const uint64_t values = static_cast<uint64_t>(e->config.page_size) * e->config.kv_head_count * e->config.head_dim;
        const uint64_t blocks = (values + e->config.block_size - 1) / e->config.block_size;
        layer.state = CF_SEAL_VALIDATING;
        invalid_counts.allocate(2 * sizeof(unsigned long long), stream, async); cuda_check(cudaMemsetAsync(invalid_counts.ptr, 0, invalid_counts.bytes, stream), "clear finite counters");
        const int threads = 256; const int scan_blocks = static_cast<int>((values + threads - 1) / threads);
        finite_scan<<<scan_blocks, threads, 0, stream>>>(static_cast<const float *>(layer.active_k.ptr), values,
            static_cast<unsigned long long *>(invalid_counts.ptr), static_cast<unsigned long long *>(invalid_counts.ptr) + 1);
        finite_scan<<<scan_blocks, threads, 0, stream>>>(static_cast<const float *>(layer.active_v.ptr), values,
            static_cast<unsigned long long *>(invalid_counts.ptr), static_cast<unsigned long long *>(invalid_counts.ptr) + 1);
        cuda_check(cudaGetLastError(), "finite validation launch"); e->counters.finite_validation_launches += 2; e->counters.values_scanned += values * 2;
        inject(CF_SEAL_FAULT_FINITE_VALIDATION);
        unsigned long long host_invalid[2]{}; cuda_check(cudaMemcpyAsync(host_invalid, invalid_counts.ptr, sizeof(host_invalid), cudaMemcpyDeviceToHost, stream), "read finite counters");
        cuda_check(cudaStreamSynchronize(stream), "finite validation sync");
        e->counters.nan_values += host_invalid[0]; e->counters.inf_values += host_invalid[1];
        if (host_invalid[0] || host_invalid[1]) { e->counters.invalid_pages_rejected++; throw std::runtime_error("active page contains NaN or Inf"); }

        layer.state = CF_SEAL_QUANTIZING; inject(CF_SEAL_FAULT_SCALE_ALLOC);
        symbols_k.allocate(values, stream, async); symbols_v.allocate(values, stream, async);
        candidate.key_scales.allocate(blocks * sizeof(float), stream, async); candidate.value_scales.allocate(blocks * sizeof(float), stream, async);
        saturation.allocate(sizeof(unsigned long long), stream, async); cuda_check(cudaMemsetAsync(saturation.ptr, 0, saturation.bytes, stream), "clear saturation");
        quantize_int4<<<static_cast<int>(blocks), 256, 0, stream>>>(static_cast<const float *>(layer.active_k.ptr), static_cast<uint8_t *>(symbols_k.ptr), static_cast<float *>(candidate.key_scales.ptr), values, e->config.block_size, static_cast<unsigned long long *>(saturation.ptr));
        quantize_int4<<<static_cast<int>(blocks), 256, 0, stream>>>(static_cast<const float *>(layer.active_v.ptr), static_cast<uint8_t *>(symbols_v.ptr), static_cast<float *>(candidate.value_scales.ptr), values, e->config.block_size, static_cast<unsigned long long *>(saturation.ptr));
        cuda_check(cudaGetLastError(), "int4 quantization launch"); e->counters.quantization_launches += 2; e->counters.values_quantized += values * 2;
        inject(CF_SEAL_FAULT_QUANTIZE_LAUNCH);

        layer.state = CF_SEAL_PACKING; inject(CF_SEAL_FAULT_PAYLOAD_ALLOC);
        const uint64_t payload_bytes = (values + 1) / 2;
        candidate.key_payload.allocate(payload_bytes, stream, async); candidate.value_payload.allocate(payload_bytes, stream, async);
        const int pack_blocks = static_cast<int>((payload_bytes + threads - 1) / threads);
        pack_nibbles<<<pack_blocks, threads, 0, stream>>>(static_cast<const uint8_t *>(symbols_k.ptr), static_cast<uint8_t *>(candidate.key_payload.ptr), values);
        pack_nibbles<<<pack_blocks, threads, 0, stream>>>(static_cast<const uint8_t *>(symbols_v.ptr), static_cast<uint8_t *>(candidate.value_payload.ptr), values);
        cuda_check(cudaGetLastError(), "nibble packing launch"); e->counters.packing_launches += 2; e->counters.symbols_packed += values * 2; e->counters.payload_bytes_written += payload_bytes * 2;
        inject(CF_SEAL_FAULT_PACK_LAUNCH);

        candidate.descriptor.key_payload = static_cast<const uint8_t *>(candidate.key_payload.ptr);
        candidate.descriptor.value_payload = static_cast<const uint8_t *>(candidate.value_payload.ptr);
        candidate.descriptor.key_scales = static_cast<const float *>(candidate.key_scales.ptr);
        candidate.descriptor.value_scales = static_cast<const float *>(candidate.value_scales.ptr);
        candidate.descriptor.token_begin = layer.active_begin; candidate.descriptor.token_count = layer.active_count;
        candidate.descriptor.kv_head_count = e->config.kv_head_count; candidate.descriptor.head_dim = e->config.head_dim;
        candidate.descriptor.block_size = e->config.block_size; candidate.descriptor.blocks_per_page = static_cast<uint32_t>(blocks);
        candidate.descriptor.key_payload_bytes = payload_bytes; candidate.descriptor.value_payload_bytes = payload_bytes;
        candidate.descriptor.resident_bytes = candidate.key_payload.bytes + candidate.value_payload.bytes + candidate.key_scales.bytes + candidate.value_scales.bytes;
        candidate.descriptor.generation = layer.generation + 1;
        e->counters.descriptor_validations++; inject(CF_SEAL_FAULT_DESCRIPTOR_VALIDATE);
        if (!candidate.descriptor.key_payload || !candidate.descriptor.value_payload || !candidate.descriptor.key_scales || !candidate.descriptor.value_scales)
            throw std::runtime_error("candidate descriptor contains null storage");

        layer.state = CF_SEAL_PUBLISHING;
        const uint32_t needed = static_cast<uint32_t>(layer.pages.size() + 1);
        if (needed > layer.descriptor_capacity) {
            inject(CF_SEAL_FAULT_PAGE_TABLE_GROW);
            uint32_t next = std::max<uint32_t>(4, layer.descriptor_capacity ? layer.descriptor_capacity * 2 : 4);
            while (next < needed) next *= 2;
            Buffer replacement; replacement.allocate(static_cast<size_t>(next) * sizeof(cf_device_page_descriptor), stream, async);
            layer.device_descriptors.release(stream, async); layer.device_descriptors = std::move(replacement); layer.descriptor_capacity = next;
            e->counters.page_table_reallocations++;
        }
        inject(CF_SEAL_FAULT_ACTIVE_REPLACEMENT);
        Buffer next_k, next_v;
        const size_t active_bytes = static_cast<size_t>(e->config.page_size) * e->config.kv_head_count * e->config.head_dim * sizeof(float);
        next_k.allocate(active_bytes, stream, async); next_v.allocate(active_bytes, stream, async);
        inject(CF_SEAL_FAULT_DESCRIPTOR_PUBLISH);
        layer.pages.push_back(std::move(candidate));
        std::vector<cf_device_page_descriptor> host_descriptors; host_descriptors.reserve(layer.pages.size());
        for (const auto & page : layer.pages) host_descriptors.push_back(page.descriptor);
        cuda_check(cudaMemcpyAsync(layer.device_descriptors.ptr, host_descriptors.data(), host_descriptors.size() * sizeof(cf_device_page_descriptor), cudaMemcpyHostToDevice, stream), "publish descriptor table");
        layer.active_k.release(stream, async); layer.active_v.release(stream, async); layer.active_k = std::move(next_k); layer.active_v = std::move(next_v);
        layer.active_begin += layer.active_count; layer.active_count = 0; layer.generation++; layer.state = CF_SEAL_ACTIVE_FILLING;
        e->counters.descriptor_publications++; e->counters.page_table_generation = layer.generation; e->counters.seal_successes++;
        e->counters.compressed_bytes += layer.pages.back().descriptor.resident_bytes; e->counters.descriptor_bytes = 0;
        for (const auto & l : e->layers) e->counters.descriptor_bytes += l.device_descriptors.bytes;
        e->counters.peak_owned_bytes = std::max(e->counters.peak_owned_bytes, e->counters.compressed_bytes + e->counters.descriptor_bytes + e->counters.workspace_bytes);
        symbols_k.release(stream, async); symbols_v.release(stream, async); invalid_counts.release(stream, async); saturation.release(stream, async);
        e->error.clear(); return 0;
    } catch (const std::exception & ex) {
        layer.state = CF_SEAL_ACTIVE_FULL; e->error = ex.what(); e->counters.seal_failures++;
        candidate.key_payload.release(stream, async); candidate.value_payload.release(stream, async); candidate.key_scales.release(stream, async); candidate.value_scales.release(stream, async);
        symbols_k.release(stream, async); symbols_v.release(stream, async); invalid_counts.release(stream, async); saturation.release(stream, async);
        return -1;
    }
}

extern "C" int cf_device_seal_attention_async(cf_device_seal_engine * e, uint32_t layer_index, const cf_device_tensor_view * query,
                                                void * output, uint32_t output_type, uint64_t ots, uint64_t ohs, uint64_t oes,
                                                uint32_t query_begin, float softmax_scale, uint32_t window, void * stream_handle) {
    if (!e || layer_index >= e->layers.size() || cf_device_tensor_validate(query) != 0 || !output || !scalar_bytes(output_type)) return -1;
    try {
        auto & layer = e->layers[layer_index]; cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
        if (query->head_count != e->config.query_head_count || query->head_dim != e->config.head_dim) throw std::runtime_error("query topology mismatch");
        if (ots < static_cast<uint64_t>(query->head_count - 1) * ohs + static_cast<uint64_t>(query->head_dim - 1) * oes + scalar_bytes(output_type))
            throw std::runtime_error("output stride is too small");
        dim3 grid(query->head_count, query->token_count);
        reference_attention<<<grid, 256, e->config.head_dim * sizeof(float), stream>>>(*query, output, output_type, ots, ohs, oes,
            static_cast<const cf_device_page_descriptor *>(layer.device_descriptors.ptr), static_cast<uint32_t>(layer.pages.size()),
            static_cast<const float *>(layer.active_k.ptr), static_cast<const float *>(layer.active_v.ptr), layer.active_begin, layer.active_count,
            query_begin, e->config.query_head_count, e->config.kv_head_count, e->config.gqa_group_size, e->config.head_dim, softmax_scale, window);
        cuda_check(cudaGetLastError(), "compressed attention launch");
        e->counters.compressed_attention_launches++; e->counters.sealed_pages_consumed += layer.pages.size();
        for (const auto & page : layer.pages) e->counters.sealed_values_consumed += static_cast<uint64_t>(page.descriptor.token_count) * e->config.kv_head_count * e->config.head_dim * 2;
        e->counters.active_values_consumed += static_cast<uint64_t>(layer.active_count) * e->config.kv_head_count * e->config.head_dim * 2;
        if (query->scalar_type != CF_DEVICE_F32) e->counters.query_conversion_launches++;
        if (output_type != CF_DEVICE_F32) e->counters.output_conversion_launches++;
        e->error.clear(); return 0;
    } catch (const std::exception & ex) { e->error = ex.what(); e->counters.cuda_errors++; return -1; }
}

extern "C" int cf_device_seal_clear_async(cf_device_seal_engine * e, void * stream_handle) {
    if (!e) return -1; cudaStream_t stream = static_cast<cudaStream_t>(stream_handle); const bool async = e->config.use_async_allocator != 0;
    for (auto & layer : e->layers) {
        layer.active_k.release(stream, async); layer.active_v.release(stream, async); layer.device_descriptors.release(stream, async);
        for (auto & page : layer.pages) { page.key_payload.release(stream, async); page.value_payload.release(stream, async); page.key_scales.release(stream, async); page.value_scales.release(stream, async); }
        layer.pages.clear(); layer.active_begin = layer.active_count = 0; layer.state = CF_SEAL_ACTIVE_FILLING; layer.generation = 0; layer.descriptor_capacity = 0;
    }
    e->counters.compressed_bytes = e->counters.descriptor_bytes = 0; return 0;
}

extern "C" int cf_device_seal_set_fault(cf_device_seal_engine * e, uint32_t fault) { if (!e || fault > CF_SEAL_FAULT_ACTIVE_REPLACEMENT) return -1; e->fault = fault; return 0; }
extern "C" int cf_device_seal_get_state(const cf_device_seal_engine * e, uint32_t layer, uint32_t * state) { if (!e || layer >= e->layers.size() || !state) return -1; *state = e->layers[layer].state; return 0; }
extern "C" int cf_device_seal_get_counters(const cf_device_seal_engine * e, cf_device_seal_counters * counters) { if (!e || !counters) return -1; *counters = e->counters; return 0; }
extern "C" const char * cf_device_seal_last_error(const cf_device_seal_engine * e) { return e ? e->error.c_str() : "null seal engine"; }
