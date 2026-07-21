NVCC ?= nvcc
BUILD ?= build/m9-device-kv
CUDA_ARCH ?= sm_75

.PHONY: device-kv-compile device-kv-contract device-kv-clean

device-kv-compile:
	mkdir -p $(BUILD)
	$(NVCC) -O2 -std=c++17 -arch=$(CUDA_ARCH) -Iinclude -Xcompiler=-Wall,-Wextra,-Werror -c src/cuda/device_kv_dataplane.cu -o $(BUILD)/device_kv_dataplane.o

device-kv-contract: device-kv-compile
	$(NVCC) -O2 -std=c++17 -arch=$(CUDA_ARCH) -Iinclude -Xcompiler=-Wall,-Wextra,-Werror tests/device_kv_dataplane_contract.cpp $(BUILD)/device_kv_dataplane.o -o $(BUILD)/device_kv_dataplane_contract
	$(BUILD)/device_kv_dataplane_contract

device-kv-clean:
	rm -rf $(BUILD)
