CXX ?= g++
PYTHON ?= python3
BUILD ?= build/m13
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude
SRC := src/adaptive_compression.cpp src/persistent_page_store.cpp

.PHONY: m13-contract m13-schema m13-anchor-check m13-false-evidence m13-clean
m13-contract:
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) tests/m13_adaptive_compression_contract.cpp $(SRC) -o $(BUILD)/m13_contract
	$(BUILD)/m13_contract
	$(CXX) $(CXXFLAGS) tools/chromofold_cache.cpp $(SRC) -o $(BUILD)/chromofold-cache
	$(BUILD)/chromofold-cache verify $(BUILD)/cache.cfp model
m13-schema:
	$(PYTHON) -m json.tool integrations/llama.cpp/adaptive/adaptive-evidence.schema.json >/dev/null
	$(PYTHON) -m py_compile integrations/llama.cpp/adaptive/validate_adaptive_evidence.py
m13-anchor-check:
	grep -F 'CF_PAGE_INT2_BLOCKWISE' include/chromofold/adaptive_compression.h
	grep -F 'CF_PAGE_FP16_RAW' include/chromofold/adaptive_compression.h
	grep -F 'corrupted_records_rejected' include/chromofold/persistent_page_store.h
	$(PYTHON) -c 'import json;d=json.load(open("integrations/llama.cpp/adaptive/adaptive-evidence.schema.json"));assert d["properties"]["runtime"]["properties"]["dense_fallback_launches"]["const"]==0'
m13-false-evidence:
	mkdir -p $(BUILD)
	printf '%s\n' '{"codec_distribution":{"int2_pages":1}}' > $(BUILD)/false.json
	! $(PYTHON) integrations/llama.cpp/adaptive/validate_adaptive_evidence.py $(BUILD)/false.json
m13-clean:
	rm -rf $(BUILD)
