NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O2 -Xcompiler "-Wall -Wextra -Werror" -Iinclude

.PHONY: m11-contract m11-clean

m11-contract:
	mkdir -p build/m11
	$(NVCC) $(NVCCFLAGS) tests/m11_pinned_server_contract.cpp src/llama_server_runtime.cpp src/multisequence_cache.cpp src/cuda/multisequence_cuda_resolver.cu src/cuda/device_kv_dataplane.cu -o build/m11/m11_contract
	build/m11/m11_contract

m11-clean:
	rm -rf build/m11
