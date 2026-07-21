PYTHON ?= python3
BUILD ?= build/m9-hardware-e2e
LLAMA_ROOT ?= build/llama.cpp-pinned
LLAMA_BIN ?= $(LLAMA_ROOT)/build/bin/llama-cli
MODEL ?=
PROMPT ?= integrations/llama.cpp/e2e/prompts/long-context.txt
CUDA_ARCH ?= 80
CONTEXTS ?= 4096,8192,16384,32768,65536

.PHONY: hardware-proof-check hardware-proof-build hardware-proof-pair hardware-proof-capacity hardware-proof-validate hardware-proof-false-claim hardware-proof-clean

hardware-proof-check:
	$(PYTHON) -m py_compile integrations/llama.cpp/e2e/run_hardware_pair.py
	$(PYTHON) -m py_compile integrations/llama.cpp/e2e/compare_pair_v2.py
	$(PYTHON) -m py_compile integrations/llama.cpp/e2e/capacity_sweep_v2.py
	$(PYTHON) -m py_compile integrations/llama.cpp/e2e/validate_evidence_v2.py
	$(PYTHON) -c 'import json; json.load(open("integrations/llama.cpp/e2e/evidence-v2.schema.json"))'

hardware-proof-build:
	@test -d $(LLAMA_ROOT) || (echo "missing pinned llama.cpp checkout: $(LLAMA_ROOT)" && exit 2)
	cmake -S $(LLAMA_ROOT) -B $(LLAMA_ROOT)/build -DGGML_CUDA=ON -DGGML_CHROMOFOLD=ON -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(LLAMA_ROOT)/build --target llama-cli -j$${JOBS:-4}
	$(LLAMA_BIN) --help | grep -F -- '--kv-cache-backend'
	$(LLAMA_BIN) --help | grep -F -- '--chromofold-page-size'
	$(LLAMA_BIN) --help | grep -F -- '--chromofold-evidence'

hardware-proof-pair: hardware-proof-build
	@test -n "$(MODEL)" || (echo "MODEL is required" && exit 2)
	mkdir -p $(BUILD)/pair
	$(PYTHON) integrations/llama.cpp/e2e/run_hardware_pair.py --binary $(LLAMA_BIN) --model $(MODEL) --prompt $(PROMPT) --output-dir $(BUILD)/pair --llama-commit $$(cat integrations/llama.cpp/e2e/llama-pin.json | $(PYTHON) -c 'import json,sys; print(json.load(sys.stdin)["commit"])') --chromofold-commit $$(git rev-parse HEAD)
	$(PYTHON) integrations/llama.cpp/e2e/compare_pair_v2.py $(BUILD)/pair/dense/run.json $(BUILD)/pair/chromofold/run.json --output $(BUILD)/pair/comparison.json

hardware-proof-capacity: hardware-proof-build
	@test -n "$(MODEL)" || (echo "MODEL is required" && exit 2)
	$(PYTHON) integrations/llama.cpp/e2e/capacity_sweep_v2.py --binary $(LLAMA_BIN) --model $(MODEL) --prompt $(PROMPT) --contexts $(CONTEXTS) --output $(BUILD)/capacity.json

hardware-proof-validate:
	$(PYTHON) integrations/llama.cpp/e2e/validate_evidence_v2.py $(BUILD)/evidence.json

hardware-proof-false-claim:
	mkdir -p $(BUILD)
	printf '%s\n' '{"schema_version":2}' > $(BUILD)/fake.json
	! $(PYTHON) integrations/llama.cpp/e2e/validate_evidence_v2.py $(BUILD)/fake.json

hardware-proof-clean:
	rm -rf $(BUILD)
