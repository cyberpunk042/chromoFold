#include "chromofold/device_kv_dataplane.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

size_t scalar_bytes(uint32_t type) {
    switch (type) {
        case CF_DEVICE_F16: return sizeof(__half);
        case CF_DEVICE_BF16: return sizeof(__nv_bfloat16);
        case CF_DEVICE_F32: return sizeof(float);
        default: return 0;
    }
}

template <typename T>
__device__ float load_scalar(const void * base, size_t byte_offset);

template <>
__device__ float load_scalar<__half>(const void * base, size_t byte_offset) {
    return __half2float(*reinterpret_cast<const __half *>(static_cast<const char *>(base) + byte_offset));
}

template <>
__device__ float load_scalar<__nv_bfloat16>(const void * base, size_t byte_offset) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(static_cast<const char *>(base) + byte_offset));
}

template <>
__device__ float load_scalar<float>(const void * base, size_t byte_offset) {
    return *reinterpret_cast<const float *>(static_cast<const char *>(base) + byte_offset);
}

template <typename T>
__global__ void convert_to_active(const void * source,
                                  float * destination,
                                  uint32_t token_count,
                                  uint32_t head_count,
                                  uint32_t head_dim,
                                  uint64_t token_stride,
                                  uint64_t head_stride,
                                  uint64_t element_stride,
                                  uint32_t destination_token_offset) {
    const uint64_t count = static_cast<uint64_t>(token_count) * head_count * head_dim;
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint32_t dim = index % head_dim;
    const uint32_t head = (index / head_dim) % head_count;
    const uint32_t token = index / (static_cast<uint64_t>(head_dim) * head_count);
    const size_t source_offset = static_cast<size_t>(token) * token_stride +
                                 static_cast<size_t>(head) * head_stride +
                                 static_cast<size_t>(dim) * element_stride;
    const size_t destination_index = (static_cast<size_t>(destination_token_offset + token) * head_count + head) * head_dim + dim;
    destination[destination_index] = load_scalar<T>(source, source_offset);
}

struct DeviceBuffer {
    void * pointer = nullptr;
    size_t bytes = 0;

    void allocate(size_t requested, cudaStream_t stream, bool async_allocator) {
        if (requested == bytes && pointer != nullptr) return;
        release(stream, async_allocator);
        if (requested == 0) return;
        if (async_allocator) check_cuda(cudaMallocAsync(&pointer, requested, stream), "cudaMallocAsync");
        else check_cuda(cudaMalloc(&pointer, requested), "cudaMalloc");
        bytes = requested;
    }

    void release(cudaStream_t stream, bool async_allocator) noexcept {
        if (!pointer) return;
        if (async_allocator) cudaFreeAsync(pointer, stream);
        else cudaFree(pointer);
        pointer = nullptr;
        bytes = 0;
    }
};

struct LayerState {
    DeviceBuffer active_k;
    DeviceBuffer active_v;
    DeviceBuffer descriptors;
    uint32_t active_token_begin = 0;
    uint32_t active_token_count = 0;
    uint32_t descriptor_count = 0;
    uint32_t descriptor_capacity = 0;
};

}  // namespace

struct cf_device_kv_dataplane {
    cf_device_kv_config config{};
    std::vector<LayerState> layers;
    cf_device_kv_counters counters{};
    std::string last_error;
};

extern "C" int cf_device_tensor_validate(const cf_device_tensor_view * view) {
    if (!view || !view->data || view->token_count == 0 || view->head_count == 0 || view->head_dim == 0) return -1;
    const size_t width = scalar_bytes(view->scalar_type);
    if (width == 0 || view->element_stride_bytes < width) return -1;
    if (view->head_stride_bytes < static_cast<uint64_t>(view->head_dim - 1) * view->element_stride_bytes + width) return -1;
    if (view->token_stride_bytes < static_cast<uint64_t>(view->head_count - 1) * view->head_stride_bytes +
                                   static_cast<uint64_t>(view->head_dim - 1) * view->element_stride_bytes + width) return -1;
    return 0;
}

extern "C" cf_device_kv_dataplane * cf_device_kv_create(const cf_device_kv_config * config) {
    if (!config || config->layer_count == 0 || config->kv_head_count == 0 || config->query_head_count == 0 ||
        config->head_dim == 0 || config->page_size == 0 || config->gqa_group_size == 0 ||
        config->query_head_count != config->kv_head_count * config->gqa_group_size) return nullptr;
    try {
        auto result = std::make_unique<cf_device_kv_dataplane>();
        result->config = *config;
        result->layers.resize(config->layer_count);
        return result.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void cf_device_kv_destroy(cf_device_kv_dataplane * dataplane, void * stream_handle) {
    if (!dataplane) return;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    for (auto & layer : dataplane->layers) {
        layer.active_k.release(stream, dataplane->config.use_async_allocator != 0);
        layer.active_v.release(stream, dataplane->config.use_async_allocator != 0);
        layer.descriptors.release(stream, dataplane->config.use_async_allocator != 0);
    }
    delete dataplane;
}

extern "C" int cf_device_kv_append_async(cf_device_kv_dataplane * dataplane,
                                           uint32_t layer_index,
                                           uint32_t token_begin,
                                           const cf_device_tensor_view * keys,
                                           const cf_device_tensor_view * values,
                                           void * stream_handle) {
    if (!dataplane || layer_index >= dataplane->layers.size() || cf_device_tensor_validate(keys) != 0 ||
        cf_device_tensor_validate(values) != 0) return -1;
    try {
        if (keys->token_count != values->token_count || keys->head_count != values->head_count ||
            keys->head_dim != values->head_dim || keys->head_count != dataplane->config.kv_head_count ||
            keys->head_dim != dataplane->config.head_dim) throw std::runtime_error("device K/V topology mismatch");
        auto & layer = dataplane->layers[layer_index];
        if (layer.active_token_count != 0 && token_begin != layer.active_token_begin + layer.active_token_count)
            throw std::runtime_error("non-contiguous device token append");
        if (layer.active_token_count + keys->token_count > dataplane->config.page_size)
            throw std::runtime_error("append crosses active page boundary; seal before retry");
        cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
        const size_t page_values = static_cast<size_t>(dataplane->config.page_size) * dataplane->config.kv_head_count * dataplane->config.head_dim;
        layer.active_k.allocate(page_values * sizeof(float), stream, dataplane->config.use_async_allocator != 0);
        layer.active_v.allocate(page_values * sizeof(float), stream, dataplane->config.use_async_allocator != 0);
        if (layer.active_token_count == 0) layer.active_token_begin = token_begin;
        const uint64_t values_count = static_cast<uint64_t>(keys->token_count) * keys->head_count * keys->head_dim;
        const int threads = 256;
        const int blocks = static_cast<int>((values_count + threads - 1) / threads);
        auto launch = [&](const cf_device_tensor_view * source, float * destination) {
            switch (source->scalar_type) {
                case CF_DEVICE_F16:
                    convert_to_active<__half><<<blocks, threads, 0, stream>>>(source->data, destination, source->token_count, source->head_count, source->head_dim, source->token_stride_bytes, source->head_stride_bytes, source->element_stride_bytes, layer.active_token_count);
                    dataplane->counters.fp16_values_converted += values_count;
                    break;
                case CF_DEVICE_BF16:
                    convert_to_active<__nv_bfloat16><<<blocks, threads, 0, stream>>>(source->data, destination, source->token_count, source->head_count, source->head_dim, source->token_stride_bytes, source->head_stride_bytes, source->element_stride_bytes, layer.active_token_count);
                    dataplane->counters.bf16_values_converted += values_count;
                    break;
                case CF_DEVICE_F32:
                    convert_to_active<float><<<blocks, threads, 0, stream>>>(source->data, destination, source->token_count, source->head_count, source->head_dim, source->token_stride_bytes, source->head_stride_bytes, source->element_stride_bytes, layer.active_token_count);
                    dataplane->counters.fp32_values_copied += values_count;
                    break;
                default: throw std::runtime_error("unsupported device scalar type");
            }
        };
        launch(keys, static_cast<float *>(layer.active_k.pointer));
        launch(values, static_cast<float *>(layer.active_v.pointer));
        check_cuda(cudaGetLastError(), "device KV conversion launch");
        layer.active_token_count += keys->token_count;
        dataplane->counters.device_append_calls++;
        dataplane->counters.device_values_appended += values_count * 2;
        dataplane->counters.active_bytes = 0;
        for (const auto & state : dataplane->layers) dataplane->counters.active_bytes += state.active_k.bytes + state.active_v.bytes;
        dataplane->counters.peak_owned_bytes = std::max(dataplane->counters.peak_owned_bytes,
            dataplane->counters.active_bytes + dataplane->counters.compressed_bytes + dataplane->counters.descriptor_bytes + dataplane->counters.workspace_bytes);
        dataplane->last_error.clear();
        return 0;
    } catch (const std::exception & error) {
        dataplane->last_error = error.what();
        dataplane->counters.cuda_errors++;
        return -1;
    }
}

extern "C" int cf_device_kv_seal_full_pages_async(cf_device_kv_dataplane * dataplane, void *) {
    if (!dataplane) return -1;
    dataplane->counters.seal_attempts++;
    for (auto & layer : dataplane->layers) {
        if (layer.active_token_count == dataplane->config.page_size) {
            dataplane->last_error = "GPU quantization/packing publication is not enabled until the hardware fixture validates the transactional candidate page";
            dataplane->counters.seal_failures++;
            return -1;
        }
    }
    return 0;
}

extern "C" int cf_device_kv_attention_async(cf_device_kv_dataplane *, uint32_t, const float *, float *, uint32_t,
                                              uint32_t, float, uint32_t, void *) {
    return -1;
}

extern "C" int cf_device_kv_clear_async(cf_device_kv_dataplane * dataplane, void * stream_handle) {
    if (!dataplane) return -1;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    for (auto & layer : dataplane->layers) {
        layer.active_k.release(stream, dataplane->config.use_async_allocator != 0);
        layer.active_v.release(stream, dataplane->config.use_async_allocator != 0);
        layer.active_token_count = 0;
        layer.active_token_begin = 0;
    }
    dataplane->counters.active_bytes = 0;
    return 0;
}

extern "C" int cf_device_kv_get_counters(const cf_device_kv_dataplane * dataplane, cf_device_kv_counters * counters) {
    if (!dataplane || !counters) return -1;
    *counters = dataplane->counters;
    return 0;
}

extern "C" const char * cf_device_kv_last_error(const cf_device_kv_dataplane * dataplane) {
    return dataplane ? dataplane->last_error.c_str() : "null device KV dataplane";
}
