NVCC ?= nvcc
CXX ?= g++
PYTHON ?= python3
CUDA_ARCH ?= 80
BUILD ?= build/m15

.PHONY: m15-compile m15-contract m15-schema m15-anchor-check m15-false-evidence m15-sanitize m15-clean

m15-compile:
	mkdir -p $(BUILD)
	$(NVCC) -std=c++17 -O3 -lineinfo -arch=sm_$(CUDA_ARCH) -Iinclude -c src/cuda/multigpu_cuda_runtime.cu -o $(BUILD)/multigpu_cuda_runtime.o
	$(NVCC) -std=c++17 -O2 -Iinclude tests/m15_multigpu_cuda_contract.cpp $(BUILD)/multigpu_cuda_runtime.o -o $(BUILD)/m15_contract

m15-contract: m15-compile
	$(BUILD)/m15_contract

m15-schema:
	$(PYTHON) -m json.tool integrations/llama.cpp/multigpu/m15-evidence.schema.json >/dev/null
	$(PYTHON) -m py_compile integrations/llama.cpp/multigpu/validate_m15_evidence.py

m15-anchor-check:
	grep -F 'cudaDeviceCanAccessPeer' src/cuda/multigpu_cuda_runtime.cu
	grep -F 'p2p_copy_calls' include/chromofold/multigpu_cuda_runtime.h
	grep -F 'tensor_parallel_batches' integrations/llama.cpp/multigpu/m15-evidence.schema.json
	grep -F 'device_failure_injected' integrations/llama.cpp/multigpu/validate_m15_evidence.py

m15-false-evidence:
	mkdir -p $(BUILD)
	printf '%s\n' '{"upstream":{"patched_server_executed":true},"topology":{"devices_used":2,"peer_topology_verified":true}}' > $(BUILD)/false-evidence.json
	! $(PYTHON) integrations/llama.cpp/multigpu/validate_m15_evidence.py $(BUILD)/false-evidence.json

m15-sanitize: m15-compile
	compute-sanitizer --tool memcheck $(BUILD)/m15_contract
	compute-sanitizer --tool racecheck $(BUILD)/m15_contract
	compute-sanitizer --tool initcheck $(BUILD)/m15_contract
	compute-sanitizer --tool synccheck $(BUILD)/m15_contract

m15-clean:
	rm -rf $(BUILD)
