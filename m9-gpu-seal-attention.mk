NVCC ?= nvcc
CXX ?= g++
BUILD ?= build/m9-gpu-seal-attention
CUDA_ARCH ?= 80

INCLUDES := -Iinclude
NVCCFLAGS := -std=c++17 -O2 -lineinfo -arch=sm_$(CUDA_ARCH) -include math_constants.h $(INCLUDES)

.PHONY: gpu-seal-compile gpu-seal-contract gpu-seal-sanitize gpu-seal-clean

gpu-seal-compile:
	mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c src/cuda/device_kv_seal_attention.cu -o $(BUILD)/device_kv_seal_attention.o
	$(NVCC) $(NVCCFLAGS) -c src/cuda/device_kv_dataplane.cu -o $(BUILD)/device_kv_dataplane.o
	$(NVCC) $(NVCCFLAGS) tests/m9_device_seal_contract.cpp $(BUILD)/device_kv_seal_attention.o $(BUILD)/device_kv_dataplane.o -o $(BUILD)/m9_device_seal_contract

gpu-seal-contract: gpu-seal-compile
	$(BUILD)/m9_device_seal_contract

gpu-seal-sanitize: gpu-seal-compile
	compute-sanitizer --tool memcheck $(BUILD)/m9_device_seal_contract
	compute-sanitizer --tool racecheck $(BUILD)/m9_device_seal_contract
	compute-sanitizer --tool initcheck $(BUILD)/m9_device_seal_contract
	compute-sanitizer --tool synccheck $(BUILD)/m9_device_seal_contract

gpu-seal-clean:
	rm -rf $(BUILD)
