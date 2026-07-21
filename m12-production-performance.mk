NVCC ?= nvcc
CXX ?= g++
PYTHON ?= python3
BUILD ?= build/m12-production-performance
CUDA_ARCH ?= 80
INCLUDES := -Iinclude
NVCCFLAGS := -std=c++17 -O3 -lineinfo -arch=sm_$(CUDA_ARCH) --extended-lambda -include math_constants.h $(INCLUDES)

.PHONY: m12-compile m12-contract m12-anchor-check m12-schema m12-false-evidence m12-sanitize m12-clean

m12-compile:
	mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c src/cuda/production_attention.cu -o $(BUILD)/production_attention.o
	$(NVCC) $(NVCCFLAGS) tests/m12_production_attention_contract.cpp $(BUILD)/production_attention.o -o $(BUILD)/m12_production_attention_contract

m12-contract: m12-compile
	$(BUILD)/m12_production_attention_contract

m12-anchor-check:
	grep -F 'optimized_decode_kernel' src/cuda/production_attention.cu
	grep -F 'running_max' src/cuda/production_attention.cu
	grep -F 'unpack_int4' src/cuda/production_attention.cu
	grep -F 'dense_fallback_launches' include/chromofold/production_attention.h
	$(PYTHON) -c 'import json; p="integrations/llama.cpp/performance/performance-evidence.schema.json"; s=json.load(open(p)); assert s["properties"]["numerics"]["properties"]["optimized_reference_passed"]["const"] is True; assert s["properties"]["soak"]["properties"]["requests"]["minimum"] == 1000'

m12-schema:
	$(PYTHON) -m json.tool integrations/llama.cpp/performance/performance-evidence.schema.json >/dev/null
	$(PYTHON) -m py_compile integrations/llama.cpp/performance/validate_performance_evidence.py
	$(PYTHON) -m py_compile integrations/llama.cpp/performance/run_benchmark_matrix.py

m12-false-evidence:
	mkdir -p $(BUILD)
	printf '%s\n' '{"upstream":{"patched_server_executed":true},"model":{"gguf_sha256":"fake"},"kernels":{"reference_launches":1,"optimized_launches":1,"grouped_batches":1,"dense_fallback_launches":0,"cuda_errors":0}}' > $(BUILD)/false-evidence.json
	! $(PYTHON) integrations/llama.cpp/performance/validate_performance_evidence.py $(BUILD)/false-evidence.json

m12-sanitize: m12-compile
	compute-sanitizer --tool memcheck $(BUILD)/m12_production_attention_contract
	compute-sanitizer --tool racecheck $(BUILD)/m12_production_attention_contract
	compute-sanitizer --tool initcheck $(BUILD)/m12_production_attention_contract
	compute-sanitizer --tool synccheck $(BUILD)/m12_production_attention_contract

m12-clean:
	rm -rf $(BUILD)
