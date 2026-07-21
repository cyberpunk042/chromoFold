CXX ?= g++
PYTHON ?= python3
BUILD ?= build/m14
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude

.PHONY: m14-contract m14-schema m14-anchor-check m14-false-evidence m14-clean

m14-contract:
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) tests/m14_distributed_runtime_contract.cpp src/distributed_runtime.cpp -o $(BUILD)/m14_contract
	$(BUILD)/m14_contract

m14-schema:
	$(PYTHON) -m json.tool integrations/llama.cpp/distributed/distributed-evidence.schema.json >/dev/null
	$(PYTHON) -m py_compile integrations/llama.cpp/distributed/validate_distributed_evidence.py

m14-anchor-check:
	grep -F 'CF_DISTRIBUTION_TENSOR_PARALLEL' include/chromofold/distributed_runtime.h
	grep -F 'peer_links_enabled' src/distributed_runtime.cpp
	grep -F 'cross_device_contamination' integrations/llama.cpp/distributed/distributed-evidence.schema.json
	$(PYTHON) -c 'import json; d=json.load(open("integrations/llama.cpp/distributed/distributed-evidence.schema.json")); assert d["properties"]["topology"]["properties"]["device_count"]["minimum"] == 2'

m14-false-evidence:
	mkdir -p $(BUILD)
	printf '%s\n' '{"topology":{"device_count":1}}' > $(BUILD)/false-evidence.json
	! $(PYTHON) integrations/llama.cpp/distributed/validate_distributed_evidence.py $(BUILD)/false-evidence.json

m14-clean:
	rm -rf $(BUILD)
