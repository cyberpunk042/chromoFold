NVCC ?= nvcc
BUILD ?= build/m10-cuda-sequence-resolver
CUDA_ARCH ?= 80
NVCCFLAGS := -std=c++17 -O2 -lineinfo -arch=sm_$(CUDA_ARCH) -Iinclude -Iintegrations/llama.cpp/multisequence

.PHONY: cuda-resolver-compile cuda-resolver-contract cuda-resolver-anchor-check cuda-resolver-sanitize cuda-resolver-clean

cuda-resolver-compile:
	mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c src/cuda/multisequence_cuda_resolver.cu -o $(BUILD)/resolver.o
	$(NVCC) $(NVCCFLAGS) -c src/cuda/device_kv_dataplane.cu -o $(BUILD)/dataplane.o
	$(NVCC) $(NVCCFLAGS) -c integrations/llama.cpp/multisequence/llama_server_cuda_resolver_bridge.cpp -o $(BUILD)/bridge.o
	$(NVCC) $(NVCCFLAGS) tests/m10_cuda_sequence_resolver_contract.cpp $(BUILD)/resolver.o $(BUILD)/dataplane.o $(BUILD)/bridge.o -o $(BUILD)/contract

cuda-resolver-contract: cuda-resolver-compile
	$(BUILD)/contract

cuda-resolver-anchor-check:
	grep -F 'cf_ms_cuda_execute_batch_async' include/chromofold/multisequence_cuda_resolver.h
	grep -F 'sequence_attention_kernel' src/cuda/multisequence_cuda_resolver.cu
	grep -F 'in_flight_page_references' src/cuda/multisequence_cuda_resolver.cu
	grep -F 'dense_fallback_launches == 0' src/cuda/multisequence_cuda_resolver.cu
	grep -F 'cf_llama_server_cuda_dispatch_batch_async' integrations/llama.cpp/multisequence/llama_server_cuda_resolver_bridge.cpp

cuda-resolver-sanitize: cuda-resolver-compile
	compute-sanitizer --tool memcheck $(BUILD)/contract
	compute-sanitizer --tool racecheck $(BUILD)/contract
	compute-sanitizer --tool initcheck $(BUILD)/contract
	compute-sanitizer --tool synccheck $(BUILD)/contract

cuda-resolver-clean:
	rm -rf $(BUILD)
