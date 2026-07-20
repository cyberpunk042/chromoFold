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

.PHONY: all lib conformance clean
all: conformance

lib: $(BUILD)/libchromofold.so
$(BUILD)/libchromofold.so: $(CU)
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -shared $(CU) -o $@

conformance: $(BUILD)/libchromofold.so conformance.c
	$(CC) -std=c11 $(INC) conformance.c -L$(BUILD) -lchromofold -L$(CUDA_LIB) -lcudart \
	  -Wl,-rpath,$(BUILD) -Wl,-rpath,$(CUDA_LIB) -o $(BUILD)/conformance
	$(BUILD)/conformance

clean:
	rm -rf $(BUILD)
