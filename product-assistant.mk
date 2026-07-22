.PHONY: product-test product-smoke product-all

product-test:
	python3 tests/test_chromofold_product.py

product-smoke:
	python3 tools/chromofold.py catalog >/dev/null
	python3 tools/chromofold.py recommend --goal balanced >/dev/null
	@tmp=$$(mktemp -d); \
	python3 tools/chromofold.py configure --profile safe --output $$tmp >/dev/null; \
	test -f $$tmp/chromofold.json; \
	test -x $$tmp/run-chromofold.sh; \
	rm -rf $$tmp

product-all: product-test product-smoke
