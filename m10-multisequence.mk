CXX ?= g++
BUILD ?= build/m10-multisequence
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude

.PHONY: multisequence-compile multisequence-contract multisequence-anchor-check multisequence-clean

multisequence-compile:
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -c src/multisequence_cache.cpp -o $(BUILD)/multisequence_cache.o
	$(CXX) $(CXXFLAGS) -Iintegrations/llama.cpp/multisequence -c integrations/llama.cpp/multisequence/llama_server_multisequence_bridge.cpp -o $(BUILD)/llama_server_multisequence_bridge.o
	$(CXX) $(CXXFLAGS) tests/m10_multisequence_host_contract.cpp $(BUILD)/multisequence_cache.o -o $(BUILD)/m10_multisequence_contract

multisequence-contract: multisequence-compile
	$(BUILD)/m10_multisequence_contract

multisequence-anchor-check:
	grep -F 'cf_sequence_copy_async' include/chromofold/multisequence_cache.h
	grep -F 'cf_sequence_speculation_rollback' include/chromofold/multisequence_cache.h
	grep -F 'dense fallback is forbidden' src/multisequence_cache.cpp
	grep -F 'cf_llama_server_slot_reset_async' integrations/llama.cpp/multisequence/llama_server_multisequence_bridge.cpp

multisequence-clean:
	rm -rf $(BUILD)
