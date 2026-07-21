#include "chromofold/kv_cuda.h"
#include "chromofold/kv_cuda_owner.hpp"
#include "chromofold/kv_gpu_fixture.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

std::vector<float> cpu_attention(const std::vector<float>& query,
                                 const chromofold::gpu_fixture::EncodedKvPage& page,
                                 float scale) {
    std::vector<float> scores(page.token_count);
    float maximum = -INFINITY;
    for (std::uint32_t t = 0; t < page.token_count; ++t) {
        float score = 0.0f;
        for (std::uint32_t d = 0; d < page.head_dim; ++d) {
            score += query[d] * page.dequantized_k[static_cast<std::size_t>(t) * page.head_dim + d];
        }
        scores[t] = score * scale;
        maximum = std::max(maximum, scores[t]);
    }
    float sum = 0.0f;
    for (float& score : scores) {
        score = std::exp(score - maximum);
        sum += score;
    }
    std::vector<float> output(page.head_dim, 0.0f);
    for (std::uint32_t t = 0; t < page.token_count; ++t) {
        const float weight = scores[t] / sum;
        for (std::uint32_t d = 0; d < page.head_dim; ++d) {
            output[d] += weight * page.dequantized_v[static_cast<std::size_t>(t) * page.head_dim + d];
        }
    }
    return output;
}

}  // namespace

int main() {
    int device_count = 0;
    check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        std::cerr << "No CUDA device available\n";
        return 77;
    }

    constexpr std::uint32_t tokens = 128;
    constexpr std::uint32_t dim = 64;
    std::vector<float> keys(tokens * dim);
    std::vector<float> values(tokens * dim);
    std::vector<float> query(dim);
    for (std::uint32_t t = 0; t < tokens; ++t) {
        for (std::uint32_t d = 0; d < dim; ++d) {
            const std::size_t i = static_cast<std::size_t>(t) * dim + d;
            keys[i] = std::sin(static_cast<float>((t + 1) * (d + 1)) * 0.004f);
            values[i] = std::cos(static_cast<float>((t + 3) * (d + 2)) * 0.006f);
        }
    }
    for (std::uint32_t d = 0; d < dim; ++d) query[d] = std::sin(static_cast<float>(d + 1) * 0.017f);

    const auto host_page = chromofold::gpu_fixture::encode_int4_page(keys, values, 0, tokens, dim, 0, 64);
    auto device_page = chromofold::cuda_runtime::DeviceKvPage::upload(host_page);
    std::vector<cf_kv_device_page> descriptors{device_page.descriptor()};
    auto device_pages = chromofold::cuda_runtime::DeviceKvPageArray::upload(descriptors);

    float* device_query = nullptr;
    float* device_output = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device_query), query.size() * sizeof(float)), "cudaMalloc query");
    check(cudaMalloc(reinterpret_cast<void**>(&device_output), dim * sizeof(float)), "cudaMalloc output");
    check(cudaMemcpy(device_query, query.data(), query.size() * sizeof(float), cudaMemcpyHostToDevice), "copy query");

    cf_kv_paged_attention_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = CF_KV_CUDA_ABI_VERSION;
    desc.pages = device_pages.device_data();
    desc.page_count = device_pages.size();
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

    assert(cf_kv_validate_paged_attention_desc(&desc) == CF_OK);
    assert(cf_kv_paged_attention_async(&desc, nullptr) == CF_OK);
    check(cudaDeviceSynchronize(), "paged attention");

    std::vector<float> actual(dim);
    check(cudaMemcpy(actual.data(), device_output, dim * sizeof(float), cudaMemcpyDeviceToHost), "copy output");
    const auto expected = cpu_attention(query, host_page, desc.softmax_scale);

    float max_abs_error = 0.0f;
    double mse = 0.0;
    for (std::uint32_t d = 0; d < dim; ++d) {
        const float error = std::abs(actual[d] - expected[d]);
        max_abs_error = std::max(max_abs_error, error);
        mse += static_cast<double>(error) * error;
    }
    mse /= dim;

    cudaFree(device_query);
    cudaFree(device_output);

    std::cout << "{\"max_abs_error\":" << max_abs_error << ",\"mse\":" << mse
              << ",\"tokens\":" << tokens << ",\"head_dim\":" << dim << "}\n";
    return max_abs_error <= 2e-4f ? 0 : 1;
}
