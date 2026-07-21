NVCC ?= nvcc
BUILD ?= build/paged-kv-seam
NVCCFLAGS ?= -std=c++17 -O2 -Iinclude

.PHONY: paged-kv-seam paged-kv-seam-clean

paged-kv-seam:
	mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) tests/paged_kv_cuda_seam.cu src/cuda/paged_kv_attention.cu -o $(BUILD)/paged_kv_cuda_seam
	$(BUILD)/paged_kv_cuda_seam

paged-kv-seam-clean:
	rm -rf $(BUILD)
