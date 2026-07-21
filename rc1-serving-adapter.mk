PYTHON ?= python3

.PHONY: rc1-serving-test rc1-serving-schema rc1-serving-false-evidence rc1-serving-all

rc1-serving-test:
	$(PYTHON) tests/test_chromofold_qualification_adapter.py

rc1-serving-schema:
	$(PYTHON) -m json.tool integrations/llama.cpp/qualification/rc1-serving-adapter.schema.json >/dev/null
	$(PYTHON) integrations/llama.cpp/qualification/validate_rc1_serving_adapter.py tests/fixtures/rc1-serving-adapter-valid.json --strict-ready

rc1-serving-false-evidence:
	@! $(PYTHON) integrations/llama.cpp/qualification/validate_rc1_serving_adapter.py tests/fixtures/rc1-serving-adapter-false.json --strict-ready

rc1-serving-all: rc1-serving-test rc1-serving-schema rc1-serving-false-evidence
	@echo "RC1 serving runtime adapter contracts: PASS"
