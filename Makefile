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

.PHONY: all clean bench frontier reference experiment-a fused rank rrr rrr-wavelet fm-search

all: $(BUILD)/gpu_access $(BUILD)/frontier $(BUILD)/fused_embedding $(BUILD)/rank_bench $(BUILD)/rrr_bench $(BUILD)/rrr_wavelet $(BUILD)/fm_search

$(BUILD)/gpu_access: benchmarks/gpu_access.cu benchmarks/reference_io.h src/cuda/access.cu include/chromofold/chromofold.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/gpu_access.cu src/cuda/access.cu -o $@

$(BUILD)/frontier: benchmarks/frontier.cu benchmarks/reference_io.h src/cuda/access.cu include/chromofold/chromofold.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/frontier.cu src/cuda/access.cu -o $@

$(BUILD)/fused_embedding: benchmarks/fused_embedding.cu benchmarks/reference_io.h src/cuda/access.cu src/cuda/fused_embedding.cu include/chromofold/chromofold.h include/chromofold/detail/access_device.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/fused_embedding.cu src/cuda/access.cu src/cuda/fused_embedding.cu -o $@

$(BUILD)/rank_bench: benchmarks/rank_bench.cu benchmarks/reference_io.h src/cuda/rank.cu include/chromofold/chromofold.h include/chromofold/detail/access_device.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/rank_bench.cu src/cuda/rank.cu -o $@

$(BUILD)/rrr_bench: benchmarks/rrr_bench.cu src/cuda/rrr.cu include/chromofold/chromofold.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/rrr_bench.cu src/cuda/rrr.cu -o $@

$(BUILD)/rrr_wavelet: benchmarks/rrr_wavelet.cu src/cuda/rrr_wavelet.cu include/chromofold/chromofold.h include/chromofold/detail/rrr_wavelet_device.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/rrr_wavelet.cu src/cuda/rrr_wavelet.cu -o $@

$(BUILD)/fm_search: benchmarks/fm_search.cu src/cuda/fm_search.cu include/chromofold/detail/fm_search_device.cuh include/chromofold/detail/rrr_wavelet_device.cuh include/chromofold/detail/access_device.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/fm_search.cu src/cuda/fm_search.cu -o $@

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

# M3: native rank + two-level rank directory experiment (verify vs golden, latency vs memory)
rank: $(BUILD)/rank_bench
	@mkdir -p $(REFS)
	@for v in 256 32768; do \
	  test -f $(REFS)/ref_V$$v.cfwv || $(PYTHON) tools/export_reference.py $(REFS)/ref_V$$v.cfwv --vocab $$v >/dev/null; \
	done
	$(BUILD)/rank_bench $(REFS)/ref_V256.cfwv $(REFS)/ref_V32768.cfwv

# M4: RRR-coded bitvector rank1 — verify vs golden + the entropy memory-latency frontier across densities
RRR_DENS = 0.5 0.1 0.03 0.005
rrr: $(BUILD)/rrr_bench
	@mkdir -p $(REFS)
	@for d in $(RRR_DENS); do \
	  test -f $(REFS)/rrr_$$d.cfrr || $(PYTHON) tools/export_rrr.py $(REFS)/rrr_$$d.cfrr --density $$d >/dev/null; \
	done
	$(BUILD)/rrr_bench $(REFS)/rrr_0.5.cfrr $(REFS)/rrr_0.1.cfrr $(REFS)/rrr_0.03.cfrr $(REFS)/rrr_0.005.cfrr

# M4 (wavelet wiring): RRR-backed wavelet access+rank on a BWT'd stream — verify vs golden + entropy-size win
RRW_VOCABS = 64 256
rrr-wavelet: $(BUILD)/rrr_wavelet
	@mkdir -p $(REFS)
	@for v in $(RRW_VOCABS); do \
	  test -f $(REFS)/rrw_V$$v.cfrw || $(PYTHON) tools/export_rrr_wavelet.py $(REFS)/rrw_V$$v.cfrw --vocab $$v >/dev/null; \
	done
	$(BUILD)/rrr_wavelet $(REFS)/rrw_V64.cfrw $(REFS)/rrw_V256.cfrw

# M7: FM-index count + locate over the RRR-backed BWT wavelet — verify vs ground truth + batched throughput
FM_VOCABS = 64 256
fm-search: $(BUILD)/fm_search
	@mkdir -p $(REFS)
	@for v in $(FM_VOCABS); do \
	  test -f $(REFS)/fm_V$$v.cffm || $(PYTHON) tools/export_fm_index.py $(REFS)/fm_V$$v.cffm --vocab $$v >/dev/null; \
	done
	$(BUILD)/fm_search $(REFS)/fm_V64.cffm $(REFS)/fm_V256.cffm

# M6: fused decode+embedding-gather vs unfused (Experiment D). Reuses the V=32768 reference.
fused: $(BUILD)/fused_embedding
	@mkdir -p $(REFS)
	@test -f $(REFS)/ref_V32768.cfwv || $(PYTHON) tools/export_reference.py $(REFS)/ref_V32768.cfwv --vocab 32768 >/dev/null
	$(BUILD)/fused_embedding $(REFS)/ref_V32768.cfwv

clean:
	rm -rf $(BUILD) $(REFS)
