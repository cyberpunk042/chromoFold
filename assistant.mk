.PHONY: assistant-test assistant-smoke assistant-all

PYTHON ?= python3

assistant-test:
	$(PYTHON) tests/test_chromofold_assistant.py

assistant-smoke:
	printf '%s' '{"intent":"explain"}' | $(PYTHON) tools/chromofold_assistant.py >/dev/null
	printf '%s' '{"intent":"longer-context","model":"model.gguf","context":65536,"concurrency":4}' | $(PYTHON) tools/chromofold_assistant.py >/dev/null

assistant-all: assistant-test assistant-smoke
