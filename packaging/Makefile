# packaging/Makefile — build libchromofold.so (the native query engine as a linkable shared library) and the
# pure-seam ABI conformance test. Self-contained: does not touch the top-level build. For sovereign-os SDD-500
# (the `sovereign-chromofold-sys` FFI crate links this .so; CI runs `make conformance` with no GPU).
#
#   make lib          -> build/libchromofold.so
#   make conformance  -> build the .so + link the C seam test + run it (no GPU needed)

ARCH     ?= sm_75
NVCC     ?= nvcc
CC       ?= cc
CUDA_LIB ?= /usr/local/cuda/lib64

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
INC  := -I$(ROOT)/include
NVFLAGS := -O3 -std=c++17 -arch=$(ARCH) $(INC) --expt-relaxed-constexpr -Xcompiler -fPIC

# query-side translation units (the extern "C" cf_* entry points). Build-side / in-progress files excluded.
SRC := access.cu rank.cu rrr.cu rrr_wavelet.cu fm_search.cu fused_embedding.cu fused_kv_attention.cu fused_matmul.cu
CU  := $(addprefix $(ROOT)/src/cuda/,$(SRC))
BUILD := build

FIXTURES := fixtures/tiny.cfwv fixtures/tiny.cfrr fixtures/tiny.cfrw

.PHONY: all lib conformance seam-check seams functional clean
all: seams

lib: $(BUILD)/libchromofold.so
$(BUILD)/libchromofold.so: $(CU)
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -shared $(CU) -o $@

# FFI-signature seam: links the .so, asserts entry points export + the NULL-arg contract (no GPU)
conformance: $(BUILD)/libchromofold.so conformance.c
	$(CC) -std=c11 $(INC) conformance.c -L$(BUILD) -lchromofold -L$(CUDA_LIB) -lcudart \
	  -Wl,-rpath,$(BUILD) -Wl,-rpath,$(CUDA_LIB) -o $(BUILD)/conformance
	$(BUILD)/conformance

# .cfold-header seam: validates fixture magic + version (pure CPU, no CUDA at all)
seam-check: $(BUILD)/seam_check
$(BUILD)/seam_check: seam_check.c
	@mkdir -p $(BUILD)
	$(CC) -std=c11 seam_check.c -o $@

# the full Q-500-C pure-seam suite (both run with no GPU); the hardware/GPU functional run is gated elsewhere
seams: conformance $(BUILD)/seam_check
	@echo "-- .cfold header seam --"
	@for fx in $(FIXTURES); do $(BUILD)/seam_check $$fx || exit 1; done
	@echo "seams: PASS"

# GPU FUNCTIONAL run (hardware-gated): FM-index search through the .so, verified vs the fixture's golden
functional: $(BUILD)/libchromofold.so functional.cu fixtures/tiny.cffm
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) $(INC) functional.cu -L$(BUILD) -lchromofold \
	  -o $(BUILD)/functional -Xlinker -rpath -Xlinker $(BUILD) -Xlinker -rpath -Xlinker $(CUDA_LIB)
	$(BUILD)/functional fixtures/tiny.cffm

clean:
	rm -rf $(BUILD)
