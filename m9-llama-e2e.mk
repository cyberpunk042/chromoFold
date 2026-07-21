LLAMA_DIR ?= build/llama.cpp-pinned
MODEL ?=
RESULTS ?= build/m9-llama-e2e
PYTHON ?= python3
PIN_SHA := 76f46ad29d61fd8c1401e8221842934bf62a6064

.PHONY: llama-fetch llama-verify llama-build-dense llama-build-chromofold llama-patch-check llama-dense-evidence llama-chromofold-evidence llama-capacity llama-e2e-clean

llama-fetch:
	$(PYTHON) integrations/llama.cpp/e2e/fetch_and_prepare.py $(LLAMA_DIR)

llama-verify:
	$(PYTHON) integrations/llama.cpp/e2e/verify_upstream.py $(LLAMA_DIR)

llama-patch-check: llama-verify
	test -f $(LLAMA_DIR)/chromofold-overlay/PINNED_COMMIT
	test "$$(cat $(LLAMA_DIR)/chromofold-overlay/PINNED_COMMIT)" = "$(PIN_SHA)"

llama-build-dense: llama-fetch
	cmake -S $(LLAMA_DIR) -B $(LLAMA_DIR)/build-dense -DGGML_CUDA=ON -DLLAMA_CURL=OFF
	cmake --build $(LLAMA_DIR)/build-dense --target llama-cli -j2

llama-build-chromofold: llama-fetch
	cmake -S $(LLAMA_DIR) -B $(LLAMA_DIR)/build-chromofold -DGGML_CUDA=ON -DGGML_CHROMOFOLD=ON -DLLAMA_CURL=OFF
	cmake --build $(LLAMA_DIR)/build-chromofold --target llama-cli -j2

llama-dense-evidence:
	test -n "$(MODEL)"
	mkdir -p $(RESULTS)
	$(PYTHON) integrations/llama.cpp/e2e/run_pair.py --llama-cli $(LLAMA_DIR)/build-dense/bin/llama-cli --model $(MODEL) --output $(RESULTS)/dense.json --backend dense --context 8192 --llama-commit $(PIN_SHA)
	$(PYTHON) integrations/llama.cpp/e2e/validate_evidence.py $(RESULTS)/dense.json

llama-chromofold-evidence:
	test -n "$(MODEL)"
	mkdir -p $(RESULTS)
	$(PYTHON) integrations/llama.cpp/e2e/run_pair.py --llama-cli $(LLAMA_DIR)/build-chromofold/bin/llama-cli --model $(MODEL) --output $(RESULTS)/chromofold.json --backend chromofold --context 8192 --llama-commit $(PIN_SHA)
	$(PYTHON) integrations/llama.cpp/e2e/validate_evidence.py $(RESULTS)/chromofold.json --require-claim

llama-capacity:
	test -n "$(MODEL)"
	$(PYTHON) integrations/llama.cpp/e2e/capacity_sweep.py --llama-cli $(LLAMA_DIR)/build-chromofold/bin/llama-cli --model $(MODEL) --llama-commit $(PIN_SHA) --output-dir $(RESULTS)/capacity

llama-e2e-clean:
	rm -rf $(LLAMA_DIR) $(RESULTS)
