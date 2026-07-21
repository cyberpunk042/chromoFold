CXX ?= g++
PYTHON ?= python3
BUILD ?= build
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Werror -Iinclude

.PHONY: rc1-contract rc1-schema rc1-cli rc1-false-evidence rc1-all

$(BUILD)/rc1_runtime_contract: tests/rc1_runtime_contract.cpp src/rc1_runtime.cpp include/chromofold/rc1_runtime.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) tests/rc1_runtime_contract.cpp src/rc1_runtime.cpp -o $@

rc1-contract: $(BUILD)/rc1_runtime_contract
	$(BUILD)/rc1_runtime_contract

rc1-schema:
	$(PYTHON) -c 'import json; json.load(open("integrations/llama.cpp/qualification/rc1-evidence.schema.json"))'

rc1-cli:
	$(PYTHON) -m py_compile tools/chromofold_qualify.py

rc1-false-evidence:
	@tmp=$$(mktemp); printf '%s\n' '{"runtime":{"requests":1,"prompt_leaks":1,"correctness_failures":0,"cuda_errors":0,"cross_tenant_contamination":0,"unreconciled_references":0},"security":{"mtls_observed":true,"audit_chain_verified":true},"operations":{"worker_failure_recovered":true,"rolling_upgrade_observed":true,"rollback_observed":true,"incident_bundle_redacted":true},"release":{"qualified_digest":"sha256:a","promoted_digest":"sha256:a","sbom":true,"provenance":true}}' > $$tmp; ! $(PYTHON) tools/chromofold_qualify.py evaluate $$tmp; rm -f $$tmp

rc1-all: rc1-contract rc1-schema rc1-cli rc1-false-evidence
