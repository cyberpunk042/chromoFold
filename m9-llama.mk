ARCH ?= sm_75
NVCC ?= nvcc
CXX ?= g++
BUILD ?= build
COMMON = -O2 -std=c++17 -Iinclude -Iintegrations/llama.cpp -Wall -Wextra -Werror
CUDA_SOURCES = src/runtime/kv_gpu_fixture.cpp src/cuda/kv_cuda_owner.cu src/cuda/paged_kv_attention.cu src/runtime/compressed_kv_cache.cu

.PHONY: llama-adapter-seam llama-adapter-compile llama-integration-clean

llama-adapter-seam:
	@mkdir -p $(BUILD)
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) -Iinclude -Iintegrations/llama.cpp \
		-Xcompiler=-Wall,-Wextra,-Werror \
		tests/llama_kv_adapter_seam.cpp integrations/llama.cpp/chromofold_kv_adapter.cpp \
		$(CUDA_SOURCES) -o $(BUILD)/llama_kv_adapter_seam
	$(BUILD)/llama_kv_adapter_seam

llama-adapter-compile:
	@mkdir -p $(BUILD)
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) -Iinclude -Iintegrations/llama.cpp \
		-Xcompiler=-Wall,-Wextra,-Werror -c integrations/llama.cpp/chromofold_kv_adapter.cpp \
		-o $(BUILD)/chromofold_kv_adapter.o
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) -Iinclude -Iintegrations/llama.cpp \
		-Xcompiler=-Wall,-Wextra,-Werror -c src/runtime/compressed_kv_cache.cu \
		-o $(BUILD)/compressed_kv_cache.o

llama-integration-clean:
	rm -f $(BUILD)/llama_kv_adapter_seam $(BUILD)/chromofold_kv_adapter.o $(BUILD)/compressed_kv_cache.o
