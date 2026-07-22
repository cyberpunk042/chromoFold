#include "chromofold/kv_cuda_owner.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace chromofold::cuda_runtime {
namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename T>
T* upload_vector(const std::vector<T>& source, cudaStream_t stream, std::vector<void*>& allocations) {
    if (source.empty()) return nullptr;
    T* destination = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&destination), source.size() * sizeof(T)), "cudaMalloc");
    try {
        check(cudaMemcpyAsync(destination, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync");
    } catch (...) {
        cudaFree(destination);
        throw;
    }
    allocations.push_back(destination);
    return destination;
}

}  // namespace

DeviceKvPage::DeviceKvPage(DeviceKvPage&& other) noexcept
    : descriptor_(other.descriptor_), allocations_(std::move(other.allocations_)), resident_bytes_(other.resident_bytes_) {
    other.descriptor_ = {};
    other.resident_bytes_ = 0;
}

DeviceKvPage& DeviceKvPage::operator=(DeviceKvPage&& other) noexcept {
    if (this != &other) {
        release();
        descriptor_ = other.descriptor_;
        allocations_ = std::move(other.allocations_);
        resident_bytes_ = other.resident_bytes_;
        other.descriptor_ = {};
        other.resident_bytes_ = 0;
    }
    return *this;
}

DeviceKvPage::~DeviceKvPage() { release(); }

void DeviceKvPage::release() noexcept {
    for (void* pointer : allocations_) cudaFree(pointer);
    allocations_.clear();
    descriptor_ = {};
    resident_bytes_ = 0;
}

DeviceKvPage DeviceKvPage::upload(const gpu_fixture::EncodedKvPage& page, void* raw_stream) {
    gpu_fixture::validate_page(page);
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(raw_stream);
    DeviceKvPage result;
    result.descriptor_.k_words = upload_vector(page.k.words, stream, result.allocations_);
    result.descriptor_.k_block_offsets = upload_vector(page.k.block_offsets, stream, result.allocations_);
    result.descriptor_.k_lut = upload_vector(page.k.lut, stream, result.allocations_);
    result.descriptor_.k_scales = upload_vector(page.k_scales, stream, result.allocations_);
    result.descriptor_.v_words = upload_vector(page.v.words, stream, result.allocations_);
    result.descriptor_.v_block_offsets = upload_vector(page.v.block_offsets, stream, result.allocations_);
    result.descriptor_.v_lut = upload_vector(page.v.lut, stream, result.allocations_);
    result.descriptor_.v_scales = upload_vector(page.v_scales, stream, result.allocations_);
    result.descriptor_.token_begin = page.token_begin;
    result.descriptor_.token_count = page.token_count;
    result.descriptor_.head_dim = page.head_dim;
    result.descriptor_.block_size = page.k.block_size;
    result.descriptor_.k_max_code_length = page.k.max_code_length;
    result.descriptor_.v_max_code_length = page.v.max_code_length;
    result.descriptor_.zero_point = page.zero_point;
    result.descriptor_.kv_head = page.kv_head;
    result.descriptor_.fixed_code_length = page.k.fixed_width;  // K and V share the codec width in this fixture
    result.resident_bytes_ = page.resident_bytes();
    return result;
}

DeviceKvPageArray::DeviceKvPageArray(DeviceKvPageArray&& other) noexcept
    : device_data_(other.device_data_), size_(other.size_) {
    other.device_data_ = nullptr;
    other.size_ = 0;
}

DeviceKvPageArray& DeviceKvPageArray::operator=(DeviceKvPageArray&& other) noexcept {
    if (this != &other) {
        if (device_data_ != nullptr) cudaFree(device_data_);
        device_data_ = other.device_data_;
        size_ = other.size_;
        other.device_data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

DeviceKvPageArray::~DeviceKvPageArray() {
    if (device_data_ != nullptr) cudaFree(device_data_);
}

DeviceKvPageArray DeviceKvPageArray::upload(const std::vector<cf_kv_device_page>& descriptors, void* raw_stream) {
    if (descriptors.empty() || descriptors.size() > CF_KV_CUDA_MAX_PAGES_PER_LAUNCH) {
        throw std::invalid_argument("invalid descriptor array size");
    }
    DeviceKvPageArray result;
    result.size_ = static_cast<std::uint32_t>(descriptors.size());
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(raw_stream);
    check(cudaMalloc(reinterpret_cast<void**>(&result.device_data_), descriptors.size() * sizeof(cf_kv_device_page)),
          "cudaMalloc descriptors");
    try {
        check(cudaMemcpyAsync(result.device_data_, descriptors.data(), descriptors.size() * sizeof(cf_kv_device_page),
                              cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync descriptors");
    } catch (...) {
        cudaFree(result.device_data_);
        result.device_data_ = nullptr;
        result.size_ = 0;
        throw;
    }
    return result;
}

}  // namespace chromofold::cuda_runtime
