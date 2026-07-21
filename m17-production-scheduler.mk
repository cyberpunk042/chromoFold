CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude

.PHONY: m17-contract m17-schema m17-anchor-check m17-false-evidence

m17-contract:
	mkdir -p build/m17
	$(CXX) $(CXXFLAGS) tests/m17_production_scheduler_contract.cpp src/production_scheduler.cpp -o build/m17/m17_contract
	build/m17/m17_contract

m17-schema:
	python3 -m json.tool integrations/llama.cpp/scheduler/m17-evidence.schema.json >/dev/null
	python3 -m py_compile integrations/llama.cpp/scheduler/validate_m17_evidence.py

m17-anchor-check:
	grep -F 'CF_REJECT_QUOTA' include/chromofold/production_scheduler.h
	grep -F 'fairness_score' src/production_scheduler.cpp
	grep -F 'CF_CIRCUIT_OPEN' src/production_scheduler.cpp
	grep -F 'active_refs_at_shutdown' include/chromofold/production_scheduler.h
	python3 -c "import json; s=json.load(open('integrations/llama.cpp/scheduler/m17-evidence.schema.json')); assert s['properties']['isolation']['properties']['cross_tenant_contamination']['const'] == 0"

m17-false-evidence:
	! python3 integrations/llama.cpp/scheduler/validate_m17_evidence.py integrations/llama.cpp/scheduler/false-evidence.json
