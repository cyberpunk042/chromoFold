PYTHON ?= python3

.PHONY: rc1-harness-test rc1-harness-schema rc1-harness-anchor-check rc1-harness-false-evidence rc1-harness-all

rc1-harness-test:
	$(PYTHON) tests/test_chromofold_runtime_harness.py

rc1-harness-schema:
	$(PYTHON) -m json.tool integrations/llama.cpp/qualification/rc1-runtime-harness.schema.json >/dev/null
	$(PYTHON) -m py_compile tools/chromofold_runtime_harness.py integrations/llama.cpp/qualification/validate_rc1_runtime_evidence.py

rc1-harness-anchor-check:
	grep -q 'decode-worker-failure' tools/chromofold_runtime_harness.py
	grep -q 'rolling-upgrade' tools/chromofold_runtime_harness.py
	grep -q 'INCOMPLETE' tools/chromofold_runtime_harness.py
	grep -q 'qualified_digest' integrations/llama.cpp/qualification/validate_rc1_runtime_evidence.py
	grep -q 'incident_bundle_redacted' integrations/llama.cpp/qualification/rc1-runtime-harness.schema.json

rc1-harness-false-evidence:
	@tmp=$$(mktemp); \
	printf '%s\n' '{"schema":"chromofold.rc1.runtime-evidence.v1","mode":"smoke","release_digest":"deadbeef","qualified_digest":"a","promoted_digest":"b","decision":"PASS","environment":{},"gpu":{"source":"none","devices":[]},"telemetry":{"end_to_end_correlated":false,"prompt_content_leaks":1,"metrics_scraped":false,"mtls_observed":false,"audit_chain_verified":false},"quality":{"correctness_failures":1,"cuda_errors":1,"cross_tenant_contamination":1,"dense_fallback_launches":1,"ttft_p95_ms":0,"inter_token_p95_ms":0},"memory":{},"references":{},"scenarios":{},"scenario_execution_complete":false,"scenario_success":false,"incident_bundle_redacted":false}' > $$tmp; \
	! $(PYTHON) integrations/llama.cpp/qualification/validate_rc1_runtime_evidence.py $$tmp; \
	rm -f $$tmp

rc1-harness-all: rc1-harness-test rc1-harness-schema rc1-harness-anchor-check rc1-harness-false-evidence
