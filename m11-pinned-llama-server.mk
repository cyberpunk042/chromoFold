PYTHON ?= python3
LLAMA_DIR ?= build/llama.cpp-m11
LLAMA_REPO ?= https://github.com/ggml-org/llama.cpp.git
LLAMA_PIN := 76f46ad29d61fd8c1401e8221842934bf62a6064
M11 := integrations/llama.cpp/server

.PHONY: m11-fetch m11-verify m11-apply m11-idempotency m11-schema m11-false-evidence m11-anchor-check m11-clean

m11-fetch:
	@if [ ! -d "$(LLAMA_DIR)/.git" ]; then git clone --filter=blob:none $(LLAMA_REPO) $(LLAMA_DIR); fi
	git -C $(LLAMA_DIR) fetch origin $(LLAMA_PIN)
	git -C $(LLAMA_DIR) checkout --detach $(LLAMA_PIN)

m11-verify: m11-fetch
	$(PYTHON) $(M11)/apply_pinned_server_patch.py $(LLAMA_DIR) --verify-only

m11-apply: m11-fetch
	$(PYTHON) $(M11)/apply_pinned_server_patch.py $(LLAMA_DIR) --report build/m11-patch-report.json

m11-idempotency: m11-apply
	$(PYTHON) $(M11)/apply_pinned_server_patch.py $(LLAMA_DIR) --report build/m11-patch-report-second.json
	$(PYTHON) -c 'import json; d=json.load(open("build/m11-patch-report-second.json")); assert d["idempotent"] and not d["changed"]'

m11-schema:
	$(PYTHON) -m json.tool $(M11)/production-evidence.schema.json >/dev/null
	$(PYTHON) -m json.tool $(M11)/pinned_server_manifest.json >/dev/null

m11-false-evidence:
	@mkdir -p build
	@printf '%s\n' '{"upstream":{"llama_commit":"$(LLAMA_PIN)","chromofold_commit":"deadbee","patch_idempotent":true}}' > build/m11-false-evidence.json
	@if $(PYTHON) $(M11)/validate_production_evidence.py build/m11-false-evidence.json >/dev/null 2>&1; then echo 'false production evidence was accepted' >&2; exit 1; fi

m11-anchor-check:
	grep -F 'exact_pin_required' $(M11)/pinned_server_manifest.json
	grep -F 'cf_llama_server_execute_batch_async' include/chromofold/llama_server_runtime.h
	grep -F 'dense_nodes_scheduled' src/llama_server_runtime.cpp
	grep -F 'snapshots_acquired' $(M11)/validate_production_evidence.py
	grep -F 'real_http_requests' $(M11)/production-evidence.schema.json

m11-clean:
	rm -rf $(LLAMA_DIR) build/m11-*.json
