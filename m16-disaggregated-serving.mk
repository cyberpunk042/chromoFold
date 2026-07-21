CXX ?= g++
PYTHON ?= python3
BUILD ?= build/m16
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude
M16 := integrations/llama.cpp/disaggregated

.PHONY: m16-contract m16-schema m16-anchor-check m16-false-evidence m16-clean

m16-contract:
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) tests/m16_disaggregated_serving_contract.cpp src/disaggregated_serving.cpp -o $(BUILD)/m16_contract
	$(BUILD)/m16_contract

m16-schema:
	$(PYTHON) -m json.tool $(M16)/m16-evidence.schema.json >/dev/null
	$(PYTHON) -m py_compile $(M16)/validate_m16_evidence.py $(M16)/run_m16_cluster.py

m16-anchor-check:
	grep -F 'CF_WORKER_PREFILL' include/chromofold/disaggregated_serving.h
	grep -F 'CF_MSG_LEASE_TRANSFER' include/chromofold/disaggregated_serving.h
	grep -F 'stale_generation_rejections' src/disaggregated_serving.cpp
	grep -F 'partial_transfers_rejected' src/disaggregated_serving.cpp
	$(PYTHON) -c 'import json; s=json.load(open("$(M16)/m16-evidence.schema.json")); assert s["properties"]["cluster"]["properties"]["decode_workers"]["minimum"] >= 2'

m16-false-evidence:
	mkdir -p $(BUILD)
	printf '%s\n' '{"cluster":{"router":1,"prefill_workers":1,"decode_workers":1}}' > $(BUILD)/false.json
	! $(PYTHON) $(M16)/validate_m16_evidence.py $(BUILD)/false.json

m16-clean:
	rm -rf $(BUILD)
