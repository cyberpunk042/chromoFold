ARCH ?= sm_75
NVCC ?= nvcc
CXX ?= g++
BUILD ?= build
COMMON_NVCC = -O2 -std=c++17 -arch=$(ARCH) -Iinclude -lineinfo -Xcompiler=-Wall,-Wextra,-Werror

.PHONY: gpu-fixture-test gpu-correctness gpu-cache-roundtrip gpu-adapter-attention gpu-sanitize gpu-benchmark gpu-validation-clean

gpu-fixture-test:
	@mkdir -p $(BUILD)
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Wpedantic -Werror -Iinclude \
		tests/kv_gpu_fixture_test.cpp src/runtime/kv_gpu_fixture.cpp -o $(BUILD)/kv_gpu_fixture_test
	$(BUILD)/kv_gpu_fixture_test

$(BUILD)/paged_kv_cuda_correctness:
	@mkdir -p $(BUILD)
	$(NVCC) $(COMMON_NVCC) tests/paged_kv_cuda_correctness.cu \
		src/runtime/kv_gpu_fixture.cpp src/cuda/kv_cuda_owner.cu src/cuda/paged_kv_attention.cu -o $@

gpu-correctness: $(BUILD)/paged_kv_cuda_correctness
	$(BUILD)/paged_kv_cuda_correctness

$(BUILD)/cache_roundtrip_attention:
	@mkdir -p $(BUILD)
	$(NVCC) $(COMMON_NVCC) tests/cache_roundtrip_attention.cu \
		src/runtime/compressed_kv_cache.cu src/runtime/kv_gpu_fixture.cpp \
		src/cuda/kv_cuda_owner.cu src/cuda/paged_kv_attention.cu -o $@

# Round-trip: append into CompressedKvCache -> attention_view (sealed pages + active tail) -> paged attention,
# vs a dense CPU reference over the same stored values. Proves the appended KV round-trips (append-step gate).
gpu-cache-roundtrip: $(BUILD)/cache_roundtrip_attention
	$(BUILD)/cache_roundtrip_attention

$(BUILD)/adapter_attention_test:
	@mkdir -p $(BUILD)
	$(NVCC) $(COMMON_NVCC) -Iintegrations/llama.cpp tests/adapter_attention_test.cu \
		integrations/llama.cpp/chromofold_kv_adapter.cpp \
		src/runtime/compressed_kv_cache.cu src/runtime/kv_gpu_fixture.cpp \
		src/cuda/kv_cuda_owner.cu src/cuda/paged_kv_attention.cu -o $@

# The adapter-level replace primitive with the live model's GQA shape: create -> append -> attention,
# vs a dense CPU reference. This is exactly the cf_llama_kv_attention call the replace callback makes.
gpu-adapter-attention: $(BUILD)/adapter_attention_test
	$(BUILD)/adapter_attention_test

gpu-sanitize: $(BUILD)/paged_kv_cuda_correctness
	compute-sanitizer --tool memcheck --error-exitcode 1 $(BUILD)/paged_kv_cuda_correctness

$(BUILD)/paged_kv_cuda_bench:
	@mkdir -p $(BUILD)
	$(NVCC) $(COMMON_NVCC) benchmarks/paged_kv_cuda_bench.cu \
		src/runtime/kv_gpu_fixture.cpp src/cuda/kv_cuda_owner.cu src/cuda/paged_kv_attention.cu -o $@

gpu-benchmark: $(BUILD)/paged_kv_cuda_bench
	@mkdir -p $(BUILD)/results
	$(BUILD)/paged_kv_cuda_bench 4096 64 100 | tee $(BUILD)/results/paged-kv-4096.json
	$(BUILD)/paged_kv_cuda_bench 16384 128 25 | tee $(BUILD)/results/paged-kv-16384.json

gpu-validation-clean:
	rm -f $(BUILD)/kv_gpu_fixture_test $(BUILD)/paged_kv_cuda_correctness $(BUILD)/paged_kv_cuda_bench
	rm -rf $(BUILD)/results
