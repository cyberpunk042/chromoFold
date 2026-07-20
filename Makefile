# ChromoFold native engine — build (milestone M1).
# sm_75 = RTX 2080 / 2080 Ti (the reference box). Override ARCH for other GPUs.

ARCH   ?= sm_75
NVCC   ?= nvcc
CXXSTD ?= c++17
NVFLAGS = -O3 -std=$(CXXSTD) -arch=$(ARCH) -Iinclude --expt-relaxed-constexpr -lineinfo

BUILD = build

.PHONY: all clean bench reference

all: $(BUILD)/gpu_access

$(BUILD)/gpu_access: benchmarks/gpu_access.cu src/cuda/access.cu include/chromofold/chromofold.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/gpu_access.cu src/cuda/access.cu -o $@

# regenerate the frozen reference vector from the Warp prototype
reference: tools/export_reference.py
	python tools/export_reference.py benchmarks/reference.cfwv

# verify + benchmark against the frozen reference
bench: $(BUILD)/gpu_access
	cd benchmarks && ../$(BUILD)/gpu_access reference.cfwv

clean:
	rm -rf $(BUILD)
