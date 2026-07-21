ARCH ?= sm_75
NVCC ?= nvcc
CXX ?= g++
BUILD ?= build
COMMON_NVCC = -O2 -std=c++17 -arch=$(ARCH) -Iinclude -lineinfo -Xcompiler=-Wall,-Wextra,-Werror

.PHONY: gpu-fixture-test gpu-correctness gpu-sanitize gpu-benchmark gpu-validation-clean

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
