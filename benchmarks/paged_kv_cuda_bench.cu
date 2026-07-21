#include "chromofold/kv_cuda.h"
#include "chromofold/kv_cuda_owner.hpp"
#include "chromofold/kv_gpu_fixture.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
}

int main(int argc, char** argv) {
    const std::uint32_t tokens = argc > 1 ? static_cast<std::uint32_t>(std::stoul(argv[1])) : 4096u;
    const std::uint32_t dim = argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 64u;
    const std::uint32_t iterations = argc > 3 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 100u;
    if (tokens == 0 || dim == 0 || dim > CF_KV_MAX_HEAD_DIM || iterations == 0) return 2;

    int device = 0;
    cudaDeviceProp properties{};
    check(cudaGetDevice(&device), "cudaGetDevice");
    check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");

    std::vector<float> keys(static_cast<std::size_t>(tokens) * dim);
    std::vector<float> values(keys.size());
    std::vector<float> query(dim);
    for (std::uint32_t t = 0; t < tokens; ++t) {
        for (std::uint32_t d = 0; d < dim; ++d) {
            const std::size_t i = static_cast<std::size_t>(t) * dim + d;
            keys[i] = std::sin(static_cast<float>((t + 1) * (d + 5)) * 0.0017f);
            values[i] = std::cos(static_cast<float>((t + 7) * (d + 1)) * 0.0021f);
        }
    }
    for (std::uint32_t d = 0; d < dim; ++d) query[d] = std::sin(static_cast<float>(d + 2) * 0.019f);

    const auto host_page = chromofold::gpu_fixture::encode_int4_page(keys, values, 0, tokens, dim, 0, 64);
    auto page = chromofold::cuda_runtime::DeviceKvPage::upload(host_page);
    auto pages = chromofold::cuda_runtime::DeviceKvPageArray::upload({page.descriptor()});

    float* device_query = nullptr;
    float* device_output = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device_query), dim * sizeof(float)), "cudaMalloc query");
    check(cudaMalloc(reinterpret_cast<void**>(&device_output), dim * sizeof(float)), "cudaMalloc output");
    check(cudaMemcpy(device_query, query.data(), dim * sizeof(float), cudaMemcpyHostToDevice), "copy query");

    cf_kv_paged_attention_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = CF_KV_CUDA_ABI_VERSION;
    desc.pages = pages.device_data();
    desc.page_count = pages.size();
    desc.kv_head_count = 1;
    desc.query_head_count = 1;
    desc.gqa_group_size = 1;
    desc.queries = device_query;
    desc.output = device_output;
    desc.query_token_begin = tokens - 1;
    desc.query_count = 1;
    desc.head_dim = dim;
    desc.causal_window = 0;
    desc.softmax_scale = 1.0f / std::sqrt(static_cast<float>(dim));

    for (int warmup = 0; warmup < 10; ++warmup) check(static_cast<cudaError_t>(cf_kv_paged_attention_async(&desc, nullptr) == CF_OK ? cudaSuccess : cudaErrorUnknown), "warmup launch");
    check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "create start event");
    check(cudaEventCreate(&stop), "create stop event");
    check(cudaEventRecord(start), "record start");
    for (std::uint32_t i = 0; i < iterations; ++i) {
        if (cf_kv_paged_attention_async(&desc, nullptr) != CF_OK) throw std::runtime_error("kernel launch rejected");
    }
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "event sync");
    float milliseconds = 0.0f;
    check(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed time");

    const std::uint64_t dense_bytes = static_cast<std::uint64_t>(tokens) * dim * sizeof(float) * 2u;
    const std::uint64_t compressed_bytes = page.resident_bytes();
    const double latency_us = static_cast<double>(milliseconds) * 1000.0 / iterations;
    const double compression_ratio = static_cast<double>(dense_bytes) / compressed_bytes;

    std::cout << "{\n"
              << "  \"gpu\":{\"name\":\"" << properties.name << "\",\"compute_capability\":\""
              << properties.major << '.' << properties.minor << "\"},\n"
              << "  \"shape\":{\"tokens\":" << tokens << ",\"head_dim\":" << dim << ",\"iterations\":" << iterations << "},\n"
              << "  \"memory\":{\"dense_kv_bytes\":" << dense_bytes << ",\"compressed_page_bytes\":" << compressed_bytes << "},\n"
              << "  \"compression\":{\"ratio\":" << compression_ratio << "},\n"
              << "  \"latency\":{\"mean_us\":" << latency_us << "}\n"
              << "}\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_query);
    cudaFree(device_output);
    return 0;
}
