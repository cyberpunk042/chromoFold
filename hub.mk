.PHONY: hub-test hub-bundle hub-all

PYTHON ?= python3
VERSION ?= dev
DIST ?= dist

hub-test:
	$(PYTHON) tests/test_chromofold_hub.py

hub-bundle:
	mkdir -p $(DIST)
	SOURCE_DATE_EPOCH=0 $(PYTHON) tools/build_product_bundle.py \
		--version $(VERSION) \
		--output $(DIST)/chromofold-$(VERSION).tar.gz

hub-all: hub-test hub-bundle
