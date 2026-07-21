#ifndef CHROMOFOLD_KV_CUDA_OWNER_HPP
#define CHROMOFOLD_KV_CUDA_OWNER_HPP

#include <cstdint>
#include <vector>

#include "chromofold/kv_cuda.h"
#include "chromofold/kv_gpu_fixture.hpp"

namespace chromofold::cuda_runtime {

class DeviceKvPage {
public:
    DeviceKvPage() = default;
    DeviceKvPage(const DeviceKvPage&) = delete;
    DeviceKvPage& operator=(const DeviceKvPage&) = delete;
    DeviceKvPage(DeviceKvPage&& other) noexcept;
    DeviceKvPage& operator=(DeviceKvPage&& other) noexcept;
    ~DeviceKvPage();

    static DeviceKvPage upload(const gpu_fixture::EncodedKvPage& page, void* stream = nullptr);

    const cf_kv_device_page& descriptor() const noexcept { return descriptor_; }
    std::uint64_t resident_bytes() const noexcept { return resident_bytes_; }

private:
    void release() noexcept;

    cf_kv_device_page descriptor_{};
    std::vector<void*> allocations_;
    std::uint64_t resident_bytes_ = 0;
};

class DeviceKvPageArray {
public:
    DeviceKvPageArray() = default;
    DeviceKvPageArray(const DeviceKvPageArray&) = delete;
    DeviceKvPageArray& operator=(const DeviceKvPageArray&) = delete;
    DeviceKvPageArray(DeviceKvPageArray&& other) noexcept;
    DeviceKvPageArray& operator=(DeviceKvPageArray&& other) noexcept;
    ~DeviceKvPageArray();

    static DeviceKvPageArray upload(const std::vector<cf_kv_device_page>& descriptors, void* stream = nullptr);
    const cf_kv_device_page* device_data() const noexcept { return device_data_; }
    std::uint32_t size() const noexcept { return size_; }

private:
    cf_kv_device_page* device_data_ = nullptr;
    std::uint32_t size_ = 0;
};

}  // namespace chromofold::cuda_runtime

#endif
