# ChromoFold native engine — build (milestones M1, M2).
# sm_75 = RTX 2080 / 2080 Ti (the reference box). Override ARCH for other GPUs.

ARCH   ?= sm_75
NVCC   ?= nvcc
PYTHON ?= python
CXXSTD ?= c++17
NVFLAGS = -O3 -std=$(CXXSTD) -arch=$(ARCH) -Iinclude -Ibenchmarks --expt-relaxed-constexpr -lineinfo

BUILD = build
REFS  = benchmarks/refs
VOCABS = 4 16 256 32768 65536 131072

.PHONY: all clean bench frontier reference experiment-a fused

all: $(BUILD)/gpu_access $(BUILD)/frontier $(BUILD)/fused_embedding

$(BUILD)/gpu_access: benchmarks/gpu_access.cu benchmarks/reference_io.h src/cuda/access.cu include/chromofold/chromofold.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/gpu_access.cu src/cuda/access.cu -o $@

$(BUILD)/frontier: benchmarks/frontier.cu benchmarks/reference_io.h src/cuda/access.cu include/chromofold/chromofold.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/frontier.cu src/cuda/access.cu -o $@

$(BUILD)/fused_embedding: benchmarks/fused_embedding.cu benchmarks/reference_io.h src/cuda/access.cu src/cuda/fused_embedding.cu include/chromofold/chromofold.h include/chromofold/detail/access_device.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/fused_embedding.cu src/cuda/access.cu src/cuda/fused_embedding.cu -o $@

# M1: freeze the default reference vector and verify the CUDA access kernel against it
reference: tools/export_reference.py
	$(PYTHON) tools/export_reference.py benchmarks/reference.cfwv

bench: $(BUILD)/gpu_access reference
	cd benchmarks && ../$(BUILD)/gpu_access reference.cfwv

# M2: freeze one reference per vocabulary width, then run the price-of-addressability frontier
experiment-a: $(BUILD)/frontier
	@mkdir -p $(REFS)
	@for v in $(VOCABS); do \
	  $(PYTHON) tools/export_reference.py $(REFS)/ref_V$$v.cfwv --vocab $$v >/dev/null && echo "froze V=$$v"; \
	done
	$(BUILD)/frontier $(REFS)/ref_V4.cfwv $(REFS)/ref_V16.cfwv $(REFS)/ref_V256.cfwv \
	                  $(REFS)/ref_V32768.cfwv $(REFS)/ref_V65536.cfwv $(REFS)/ref_V131072.cfwv

frontier: experiment-a

# M6: fused decode+embedding-gather vs unfused (Experiment D). Reuses the V=32768 reference.
fused: $(BUILD)/fused_embedding
	@mkdir -p $(REFS)
	@test -f $(REFS)/ref_V32768.cfwv || $(PYTHON) tools/export_reference.py $(REFS)/ref_V32768.cfwv --vocab 32768 >/dev/null
	$(BUILD)/fused_embedding $(REFS)/ref_V32768.cfwv

clean:
	rm -rf $(BUILD) $(REFS)
