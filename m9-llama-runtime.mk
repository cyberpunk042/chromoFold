LLAMA_DIR ?= build/llama.cpp-pinned
PYTHON ?= python3
NVCC ?= nvcc
CXX ?= g++
ARCH ?= sm_75

.PHONY: llama-runtime-check llama-runtime-apply llama-runtime-verify llama-runtime-compile

llama-runtime-check:
	$(PYTHON) integrations/llama.cpp/runtime/apply_runtime_patch.py $(LLAMA_DIR) --check

llama-runtime-apply:
	$(PYTHON) integrations/llama.cpp/runtime/apply_runtime_patch.py $(LLAMA_DIR)

llama-runtime-verify:
	$(PYTHON) integrations/llama.cpp/runtime/verify_runtime_patch.py $(LLAMA_DIR) --require-applied

llama-runtime-compile:
	mkdir -p build/runtime-objects
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Wpedantic -Werror -Iintegrations/llama.cpp -Iinclude \
		-c integrations/llama.cpp/chromofold_runtime_bridge.cpp -o build/runtime-objects/chromofold_runtime_bridge.o
	$(PYTHON) -m py_compile integrations/llama.cpp/runtime/apply_runtime_patch.py integrations/llama.cpp/runtime/verify_runtime_patch.py
