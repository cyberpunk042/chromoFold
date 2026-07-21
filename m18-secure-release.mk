CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror
M18_BUILD := build/m18
M18_EVIDENCE := integrations/llama.cpp/security/m18-evidence.json

.PHONY: m18-contract m18-schema m18-anchor-check m18-false-evidence m18-sbom m18-all

m18-contract:
	mkdir -p $(M18_BUILD)
	$(CXX) $(CXXFLAGS) -Iinclude tests/m18_security_release_contract.cpp src/security_release.cpp -o $(M18_BUILD)/m18_contract
	$(M18_BUILD)/m18_contract

m18-schema:
	python3 -m py_compile tools/chromofold_release.py integrations/llama.cpp/security/validate_m18_evidence.py
	python3 -c 'import json; json.load(open("integrations/llama.cpp/security/m18-evidence.schema.json"))'

m18-anchor-check:
	grep -q 'CF_OP_UPDATE_POLICY' include/chromofold/security_release.h
	grep -q 'tenant_substitution_rejections' src/security_release.cpp
	grep -q 'replay_rejections' integrations/llama.cpp/security/m18-evidence.schema.json
	grep -q 'readOnlyRootFilesystem: true' deploy/m18/chromofold-secure.yaml
	grep -q '^USER 65532:65532' containers/m18/Dockerfile

m18-false-evidence:
	mkdir -p $(M18_BUILD)
	printf '%s\n' '{"transport":{"mtls_connections":0}}' > $(M18_BUILD)/false-evidence.json
	! python3 integrations/llama.cpp/security/validate_m18_evidence.py $(M18_BUILD)/false-evidence.json

m18-sbom:
	mkdir -p $(M18_BUILD)/bundle
	printf 'chromofold-m18\n' > $(M18_BUILD)/bundle/chromofold.txt
	python3 tools/chromofold_release.py sbom $(M18_BUILD)/bundle $(M18_BUILD)/bundle/sbom.cdx.json

test: m18-contract m18-schema m18-anchor-check m18-false-evidence
m18-all: test m18-sbom
