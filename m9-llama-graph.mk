LLAMA_DIR ?= build/llama.cpp-pinned
PYTHON ?= python3
NVCC ?= nvcc
BUILD ?= build

.PHONY: llama-graph-check llama-graph-apply llama-graph-verify llama-graph-contract-test

llama-graph-check:
	$(PYTHON) integrations/llama.cpp/graph/apply_graph_wiring.py $(LLAMA_DIR) --check

llama-graph-apply:
	$(PYTHON) integrations/llama.cpp/graph/apply_graph_wiring.py $(LLAMA_DIR)

llama-graph-verify:
	$(PYTHON) integrations/llama.cpp/graph/verify_graph_wiring.py $(LLAMA_DIR)

llama-graph-contract-test:
	mkdir -p $(BUILD)
	$(NVCC) -O2 -std=c++17 -I. -Iinclude -Iintegrations/llama.cpp \
		tests/llama_graph_contract_test.cpp \
		integrations/llama.cpp/graph/chromofold_graph_contract.cpp \
		integrations/llama.cpp/chromofold_runtime_bridge.cpp \
		integrations/llama.cpp/chromofold_kv_adapter.cpp \
		src/runtime/compressed_kv_cache.cu src/runtime/kv_gpu_fixture.cpp \
		src/cuda/kv_cuda_owner.cu src/cuda/paged_kv_attention.cu \
		-o $(BUILD)/llama_graph_contract_test
	$(BUILD)/llama_graph_contract_test
