#include "chromofold/production_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

struct GraphEntry {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
};

__device__ inline float load_scalar(const void * base, uint32_t type, uint64_t offset) {
    const char * p = static_cast<const char *>(base) + offset;
    if (type == CF_DEVICE_F16) return __half2float(*reinterpret_cast<const __half *>(p));
    if (type == CF_DEVICE_BF16) return __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(p));
    return *reinterpret_cast<const float *>(p);
}

__device__ inline void store_scalar(void * base, uint32_t type, uint64_t offset, float value) {
    char * p = static_cast<char *>(base) + offset;
    if (type == CF_DEVICE_F16) *reinterpret_cast<__half *>(p) = __float2half(value);
    else if (type == CF_DEVICE_BF16) *reinterpret_cast<__nv_bfloat16 *>(p) = __float2bfloat16(value);
    else *reinterpret_cast<float *>(p) = value;
}

__device__ inline float warp_sum(float value) {
    for (int delta = 16; delta > 0; delta >>= 1) value += __shfl_down_sync(0xffffffffu, value, delta);
    return value;
}

__device__ inline float unpack_int4(const uint8_t * payload, uint64_t index, float scale) {
    const uint8_t packed = payload[index >> 1];
    int value = (index & 1u) ? (packed >> 4) : (packed & 0x0fu);
    value = value >= 8 ? value - 16 : value;
    return static_cast<float>(value) * scale;
}

__device__ inline void update_online(float score,
                                     float value,
                                     float & running_max,
                                     float & running_denom,
                                     float & accumulator) {
    const float next_max = fmaxf(running_max, score);
    const float old_scale = isfinite(running_max) ? expf(running_max - next_max) : 0.0f;
    const float weight = expf(score - next_max);
    accumulator = accumulator * old_scale + weight * value;
    running_denom = running_denom * old_scale + weight;
    running_max = next_max;
}

__global__ void optimized_decode_kernel(const cf_attention_batch_item * items,
                                        uint32_t item_count,
                                        uint32_t query_heads,
                                        uint32_t kv_heads,
                                        uint32_t head_dim,
                                        uint32_t gqa_group_size) {
    const uint32_t item_index = blockIdx.x;
    const uint32_t query_head = blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (item_index >= item_count || query_head >= query_heads) return;

    const cf_attention_batch_item item = items[item_index];
    const uint32_t kv_head = query_head / gqa_group_size;
    float running_max = -CUDART_INF_F;
    float running_denom = 0.0f;
    extern __shared__ float output_accumulator[];
    for (uint32_t d = lane; d < head_dim; d += 32u) output_accumulator[d] = 0.0f;
    __syncwarp();

    for (uint32_t page_index = 0; page_index < item.sealed_page_count; ++page_index) {
        const cf_device_page_descriptor page = item.sealed_pages[page_index];
        if (!page.key_payload || !page.value_payload || !page.key_scales || !page.value_scales) continue;
        const uint32_t blocks_per_head = (page.head_dim + page.block_size - 1u) / page.block_size;
        for (uint32_t token = 0; token < page.token_count; ++token) {
            const uint32_t absolute = page.token_begin + token;
            if (item.causal_window && absolute + item.causal_window < item.query_token_begin) continue;
            float dot = 0.0f;
            for (uint32_t d = lane; d < head_dim; d += 32u) {
                const uint64_t qoff = item.query.head_stride_bytes * query_head + item.query.element_stride_bytes * d;
                const float query_value = load_scalar(item.query.data, item.query.scalar_type, qoff);
                const uint64_t index = (static_cast<uint64_t>(token) * kv_heads + kv_head) * head_dim + d;
                const uint64_t scale_index = (static_cast<uint64_t>(token) * kv_heads + kv_head) * blocks_per_head + d / page.block_size;
                dot += query_value * unpack_int4(page.key_payload, index, page.key_scales[scale_index]);
            }
            dot = __shfl_sync(0xffffffffu, warp_sum(dot), 0) * item.softmax_scale;
            const float previous_max = running_max;
            const float next_max = fmaxf(previous_max, dot);
            const float old_scale = isfinite(previous_max) ? expf(previous_max - next_max) : 0.0f;
            const float weight = expf(dot - next_max);
            const float next_denom = running_denom * old_scale + weight;
            for (uint32_t d = lane; d < head_dim; d += 32u) {
                const uint64_t index = (static_cast<uint64_t>(token) * kv_heads + kv_head) * head_dim + d;
                const uint64_t scale_index = (static_cast<uint64_t>(token) * kv_heads + kv_head) * blocks_per_head + d / page.block_size;
                const float value = unpack_int4(page.value_payload, index, page.value_scales[scale_index]);
                output_accumulator[d] = output_accumulator[d] * old_scale + weight * value;
            }
            running_max = next_max;
            running_denom = next_denom;
        }
    }

    for (uint32_t token = 0; token < item.active_token_count; ++token) {
        const uint32_t absolute = item.active_token_begin + token;
        if (item.causal_window && absolute + item.causal_window < item.query_token_begin) continue;
        float dot = 0.0f;
        for (uint32_t d = lane; d < head_dim; d += 32u) {
            const uint64_t qoff = item.query.head_stride_bytes * query_head + item.query.element_stride_bytes * d;
            const float query_value = load_scalar(item.query.data, item.query.scalar_type, qoff);
            const uint64_t index = (static_cast<uint64_t>(token) * kv_heads + kv_head) * head_dim + d;
            dot += query_value * item.active_keys[index];
        }
        dot = __shfl_sync(0xffffffffu, warp_sum(dot), 0) * item.softmax_scale;
        const float previous_max = running_max;
        const float next_max = fmaxf(previous_max, dot);
        const float old_scale = isfinite(previous_max) ? expf(previous_max - next_max) : 0.0f;
        const float weight = expf(dot - next_max);
        const float next_denom = running_denom * old_scale + weight;
        for (uint32_t d = lane; d < head_dim; d += 32u) {
            const uint64_t index = (static_cast<uint64_t>(token) * kv_heads + kv_head) * head_dim + d;
            output_accumulator[d] = output_accumulator[d] * old_scale + weight * item.active_values[index];
        }
        running_max = next_max;
        running_denom = next_denom;
    }

    const float inverse = running_denom > 0.0f ? 1.0f / running_denom : 0.0f;
    for (uint32_t d = lane; d < head_dim; d += 32u) {
        const uint64_t offset = item.output_head_stride_bytes * query_head + item.output_element_stride_bytes * d;
        store_scalar(item.output, item.output_type, offset, output_accumulator[d] * inverse);
    }
}

} // namespace

struct cf_attention_runtime {
    cf_attention_runtime_config config{};
    mutable std::mutex mutex;
    cf_attention_counters counters{};
    std::unordered_map<uint64_t, GraphEntry> graphs;
    std::string error;
};

extern "C" cf_attention_runtime * cf_attention_runtime_create(const cf_attention_runtime_config * config) {
    if (!config || !config->layer_count || !config->query_head_count || !config->kv_head_count || !config->head_dim) return nullptr;
    if (config->query_head_count != config->kv_head_count * config->gqa_group_size) return nullptr;
    try {
        auto runtime = std::make_unique<cf_attention_runtime>();
        runtime->config = *config;
        return runtime.release();
    } catch (...) { return nullptr; }
}

extern "C" void cf_attention_runtime_destroy(cf_attention_runtime * runtime, void *) {
    if (!runtime) return;
    for (auto & entry : runtime->graphs) {
        if (entry.second.exec) cudaGraphExecDestroy(entry.second.exec);
        if (entry.second.graph) cudaGraphDestroy(entry.second.graph);
    }
    delete runtime;
}

extern "C" int cf_attention_execute_async(cf_attention_runtime * runtime,
                                             const cf_attention_batch_item * items,
                                             uint32_t item_count,
                                             void * stream_ptr) {
    if (!runtime || !items || !item_count) return -1;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    std::lock_guard<std::mutex> guard(runtime->mutex);
    try {
        if (runtime->config.mode == CF_ATTENTION_REFERENCE) {
            runtime->error = "reference execution is delegated to the M9 correctness kernel";
            ++runtime->counters.reference_launches;
            return -1;
        }
        cf_attention_batch_item * device_items = nullptr;
        const size_t bytes = sizeof(cf_attention_batch_item) * item_count;
        cudaError_t error = cudaMallocAsync(&device_items, bytes, stream);
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        ++runtime->counters.allocation_count;
        error = cudaMemcpyAsync(device_items, items, bytes, cudaMemcpyHostToDevice, stream);
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        optimized_decode_kernel<<<dim3(item_count, runtime->config.query_head_count, 1), 32,
                                  runtime->config.head_dim * sizeof(float), stream>>>(
            device_items, item_count, runtime->config.query_head_count, runtime->config.kv_head_count,
            runtime->config.head_dim, runtime->config.gqa_group_size);
        error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        cudaFreeAsync(device_items, stream);
        ++runtime->counters.optimized_launches;
        ++runtime->counters.decode_launches;
        if (item_count > 1) {
            ++runtime->counters.grouped_batches;
            runtime->counters.grouped_items += item_count;
            runtime->counters.kernel_launches_avoided += item_count - 1;
        }
        runtime->error.clear();
        return 0;
    } catch (const std::exception & exception) {
        ++runtime->counters.cuda_errors;
        runtime->error = exception.what();
        return -1;
    }
}

extern "C" int cf_attention_compare_async(cf_attention_runtime * runtime,
                                             const cf_attention_batch_item *,
                                             uint32_t item_count,
                                             float max_abs_error,
                                             float mean_abs_error,
                                             void *) {
    if (!runtime || !item_count || max_abs_error <= 0.0f || mean_abs_error <= 0.0f) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    runtime->error = "comparison requires reference and optimized buffers from the hardware harness";
    return -1;
}

extern "C" int cf_attention_invalidate_graphs(cf_attention_runtime * runtime) {
    if (!runtime) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    for (auto & entry : runtime->graphs) {
        if (entry.second.exec) cudaGraphExecDestroy(entry.second.exec);
        if (entry.second.graph) cudaGraphDestroy(entry.second.graph);
    }
    runtime->graphs.clear();
    ++runtime->counters.graph_invalidations;
    return 0;
}

extern "C" int cf_attention_get_counters(const cf_attention_runtime * runtime, cf_attention_counters * counters) {
    if (!runtime || !counters) return -1;
    std::lock_guard<std::mutex> guard(runtime->mutex);
    *counters = runtime->counters;
    return 0;
}

extern "C" const char * cf_attention_last_error(const cf_attention_runtime * runtime) {
    return runtime ? runtime->error.c_str() : "runtime is null";
}
