CXX ?= g++
PYTHON ?= python3
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror
BUILD ?= build/m19

.PHONY: m19-contract m19-schema m19-anchor-check m19-false-evidence m19-all

m19-contract:
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -Iinclude tests/m19_qualification_contract.cpp src/qualification.cpp -o $(BUILD)/m19_contract
	$(BUILD)/m19_contract

m19-schema:
	$(PYTHON) -m json.tool integrations/llama.cpp/qualification/m19-evidence.schema.json >/dev/null

m19-anchor-check:
	grep -q 'CF_ERR_CUDA_OOM' include/chromofold/qualification.h
	grep -q 'cardinality_bounded' src/qualification.cpp
	grep -q 'steady_state_growth_bytes' integrations/llama.cpp/qualification/m19-evidence.schema.json
	grep -q 'qualified_digest' integrations/llama.cpp/qualification/validate_m19_evidence.py

m19-false-evidence:
	mkdir -p $(BUILD)
	printf '%s\n' '{"release":{"qualified_digest":"a","promoted_digest":"b"}}' > $(BUILD)/false-evidence.json
	! $(PYTHON) integrations/llama.cpp/qualification/validate_m19_evidence.py $(BUILD)/false-evidence.json

m19-all: m19-contract m19-schema m19-anchor-check m19-false-evidence
