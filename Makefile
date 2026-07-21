# ChromoFold native engine — build (milestones M1, M2).
# sm_75 = RTX 2080 / 2080 Ti (the reference box). Override ARCH for other GPUs.

ARCH   ?= sm_75
NVCC   ?= nvcc
CXX    ?= g++
PYTHON ?= python3
CXXSTD ?= c++17
NVFLAGS = -O3 -std=$(CXXSTD) -arch=$(ARCH) -Iinclude -Ibenchmarks --expt-relaxed-constexpr -lineinfo

BUILD = build
REFS  = benchmarks/refs
VOCABS = 4 16 256 32768 65536 131072

.PHONY: all clean bench frontier reference experiment-a fused rank rrr rrr-wavelet rans fm-search fused-matmul kv-attention sparse-gather delta suffix-array build-index test test-quick package conformance m16-contract bench-smoke

all: $(BUILD)/gpu_access $(BUILD)/frontier $(BUILD)/fused_embedding $(BUILD)/rank_bench $(BUILD)/rrr_bench $(BUILD)/rrr_wavelet $(BUILD)/rans_bench $(BUILD)/fm_search $(BUILD)/fused_matmul $(BUILD)/fused_kv_attention $(BUILD)/sparse_gather $(BUILD)/delta_bench $(BUILD)/suffix_array

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

$(BUILD)/rans_bench: benchmarks/rans_bench.cu src/cuda/rans.cu include/chromofold/chromofold.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/rans_bench.cu src/cuda/rans.cu -o $@

$(BUILD)/fm_search: benchmarks/fm_search.cu src/cuda/fm_search.cu include/chromofold/detail/fm_search_device.cuh include/chromofold/detail/rrr_wavelet_device.cuh include/chromofold/detail/access_device.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/fm_search.cu src/cuda/fm_search.cu -o $@

$(BUILD)/fused_matmul: benchmarks/fused_matmul.cu src/cuda/fused_matmul.cu include/chromofold/chromofold.h include/chromofold/detail/block_huffman_device.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/fused_matmul.cu src/cuda/fused_matmul.cu -o $@

$(BUILD)/fused_kv_attention: benchmarks/fused_kv_attention.cu src/cuda/fused_kv_attention.cu include/chromofold/chromofold.h include/chromofold/detail/block_huffman_device.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/fused_kv_attention.cu src/cuda/fused_kv_attention.cu -o $@

$(BUILD)/sparse_gather: benchmarks/sparse_gather.cu src/cuda/sparse_gather.cu include/chromofold/detail/rrr_wavelet_device.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/sparse_gather.cu src/cuda/sparse_gather.cu -o $@

$(BUILD)/delta_bench: benchmarks/delta_bench.cu src/cuda/delta_apply.cu include/chromofold/chromofold.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/delta_bench.cu src/cuda/delta_apply.cu -o $@

$(BUILD)/suffix_array: benchmarks/suffix_array.cu src/cuda/suffix_array.cu include/chromofold/detail/suffix_cpu.hpp
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) benchmarks/suffix_array.cu src/cuda/suffix_array.cu -o $@

# M5: native C++ offline builder (no Warp) — pure g++, no CUDA
$(BUILD)/build_index: tools/build_index.cpp
	@mkdir -p $(BUILD)
	$(CXX) -O3 -std=$(CXXSTD) -march=native tools/build_index.cpp -o $@

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

# M5: build the RRR-wavelet index natively in C++ (no Warp), then VERIFY the GPU query kernel is bit-identical to
# the CPU oracle golden written by the builder — the build≠query split, self-hosted.
# M5: GPU suffix-array build verified bit-identical to the CPU SA + speedup (no Python)
suffix-array: $(BUILD)/suffix_array
	$(BUILD)/suffix_array

build-index: $(BUILD)/build_index $(BUILD)/rrr_wavelet $(BUILD)/fm_search
	@mkdir -p $(REFS)
	$(BUILD)/build_index $(REFS)/cpp_V64.cfrw --vocab 64 --fm $(REFS)/cpp_V64.cffm
	@echo "--- GPU access/rank kernel vs the C++ builder's CPU oracle golden ---"
	$(BUILD)/rrr_wavelet $(REFS)/cpp_V64.cfrw
	@echo "--- GPU FM count/locate vs the C++ builder's CPU oracle golden ---"
	$(BUILD)/fm_search $(REFS)/cpp_V64.cffm

# M6 (large-intermediate): fused int4 decode-in-GEMM vs decode-then-dense — the memory the fusion buys
fused-matmul: $(BUILD)/fused_matmul
	@mkdir -p $(REFS)
	@test -f $(REFS)/fw_2048.cffw || $(PYTHON) tools/export_fused_matmul.py $(REFS)/fw_2048.cffw --m 2048 --k 2048 >/dev/null
	@test -f $(REFS)/fw_4096.cffw || $(PYTHON) tools/export_fused_matmul.py $(REFS)/fw_4096.cffw --m 4096 --k 4096 >/dev/null
	$(BUILD)/fused_matmul $(REFS)/fw_2048.cffw $(REFS)/fw_4096.cffw

# M4: block-rANS decode vs Huffman — the near-entropy coder, honest crossover across entropy × block size
RANS_CFG = peaky:64 peaky:1024 skewed:64 skewed:1024
rans: $(BUILD)/rans_bench
	@mkdir -p $(REFS)
	@for c in $(RANS_CFG); do s=$${c%%:*}; b=$${c##*:}; \
	  test -f $(REFS)/rans_$${s}_$${b}.cfrs || $(PYTHON) tools/export_rans.py $(REFS)/rans_$${s}_$${b}.cfrs --stream $$s --block $$b >/dev/null; \
	done
	$(BUILD)/rans_bench $(REFS)/rans_peaky_64.cfrs $(REFS)/rans_peaky_1024.cfrs $(REFS)/rans_skewed_64.cfrs $(REFS)/rans_skewed_1024.cfrs

# M8: reference/delta cluster decode — cross-sequence dedup (shared prefix once + per-member sparse deltas)
delta: $(BUILD)/delta_bench
	@mkdir -p $(REFS)
	@test -f $(REFS)/delta_256.cfdc || $(PYTHON) tools/export_delta.py $(REFS)/delta_256.cfdc --members 256 --base 8000 >/dev/null
	$(BUILD)/delta_bench $(REFS)/delta_256.cfdc

# M6 (sparse-consumer / P2): fused decode+gather (touch K positions) vs decompress-all over the RRR-wavelet
sparse-gather: $(BUILD)/sparse_gather
	@mkdir -p $(REFS)
	@test -f $(REFS)/sg_n1M.cfsg || $(PYTHON) tools/export_sparse_gather.py $(REFS)/sg_n1M.cfsg --n 1000000 --vocab 256 --dim 64 >/dev/null
	$(BUILD)/sparse_gather $(REFS)/sg_n1M.cfsg

# M6/M9 (KV-path fusion): decode-in-attention over an entropy-coded KV cache vs decode-then-dense
KV_CFG = --seq 4096 --dim 64 --window 256
kv-attention: $(BUILD)/fused_kv_attention
	@mkdir -p $(REFS)
	@test -f $(REFS)/kv_s4096.cfkv || $(PYTHON) tools/export_kv_attention.py $(REFS)/kv_s4096.cfkv $(KV_CFG) >/dev/null
	@test -f $(REFS)/kv_s8192.cfkv || $(PYTHON) tools/export_kv_attention.py $(REFS)/kv_s8192.cfkv --seq 8192 --dim 128 --window 512 >/dev/null
	$(BUILD)/fused_kv_attention $(REFS)/kv_s4096.cfkv $(REFS)/kv_s8192.cfkv

# M6: fused decode+embedding-gather vs unfused (Experiment D). Reuses the V=32768 reference.
fused: $(BUILD)/fused_embedding
	@mkdir -p $(REFS)
	@test -f $(REFS)/ref_V32768.cfwv || $(PYTHON) tools/export_reference.py $(REFS)/ref_V32768.cfwv --vocab 32768 >/dev/null
	$(BUILD)/fused_embedding $(REFS)/ref_V32768.cfwv

# Unified contract test runner: builds and runs all M9–M16 CPU contracts,
# validates every evidence schema, and rejects fabricated evidence.
test:
	$(PYTHON) tests/run_all_contracts.py

test-quick:
	$(PYTHON) tests/run_all_contracts.py --quick

# Packaging: build shared library and run ABI conformance seams (no GPU needed)
package:
	$(MAKE) -C packaging lib

conformance:
	$(MAKE) -C packaging seams

# M11 pinned server contract (needs nvcc for CUDA resolver linkage)
m11-contract:
	$(MAKE) -f m11-pinned-server.mk m11-contract

# M16 disaggregated serving contract
m16-contract:
	$(MAKE) -f m16-disaggregated.mk m16-contract

# Benchmark smoke: CPU generation is mandatory; GPU verification runs only with a usable NVIDIA driver.
# M17 production scheduler contract
m17-contract:
	$(MAKE) -f m17-production-scheduler.mk m17-contract

# Paged KV CUDA seam contract
paged-kv-seam:
	$(MAKE) -f paged-kv-seam.mk paged-kv-seam

# Benchmark smoke: core benchmarks that don't need the Warp prototype.
# Skips gracefully if reference files are missing (they need the prototype Python).
bench-smoke: $(BUILD)/gpu_access $(BUILD)/rank_bench $(BUILD)/rrr_wavelet $(BUILD)/fm_search $(BUILD)/suffix_array $(BUILD)/build_index
	@echo "=== Access benchmark ==="
	@test -f $(REFS)/reference.cfwv \
	  && $(BUILD)/gpu_access $(REFS)/reference.cfwv \
	  || echo "  [SKIP] access: reference.cfwv missing (needs Warp prototype)"
	@echo "=== Rank benchmark ==="
	@test -f $(REFS)/ref_V256.cfwv \
	  && $(BUILD)/rank_bench $(REFS)/ref_V256.cfwv \
	  || echo "  [SKIP] rank: ref_V256.cfwv missing (needs Warp prototype)"
	@echo "=== RRR-wavelet benchmark ==="
	@test -f $(REFS)/rrw_V64.cfrw \
	  && $(BUILD)/rrr_wavelet $(REFS)/rrw_V64.cfrw \
	  || echo "  [SKIP] rrr-wavelet: rrw_V64.cfrw missing (needs Warp prototype)"
	@echo "=== FM-search benchmark ==="
	@test -f $(REFS)/fm_V64.cffm \
	  && $(BUILD)/fm_search $(REFS)/fm_V64.cffm \
	  || echo "  [SKIP] fm-search: fm_V64.cffm missing (needs Warp prototype)"
	@echo "=== Suffix-array build ==="
	$(BUILD)/suffix_array
	@echo "=== Native C++ build + verify ==="
	@mkdir -p $(REFS)
	$(BUILD)/build_index $(REFS)/cpp_V64.cfrw --vocab 64 --fm $(REFS)/cpp_V64.cffm
	@if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then \
	  $(BUILD)/rrr_wavelet $(REFS)/cpp_V64.cfrw; \
	  $(BUILD)/fm_search $(REFS)/cpp_V64.cffm; \
	else \
	  echo "  [SKIP] native GPU verification: no usable NVIDIA driver/device"; \
	fi

clean:
	rm -rf $(BUILD) $(REFS)
	$(MAKE) -C packaging clean
