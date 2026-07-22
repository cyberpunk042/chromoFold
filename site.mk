.PHONY: site-test site-build site-all

PYTHON ?= python3
SITE_DIST ?= dist/site

site-test:
	$(PYTHON) tests/test_public_site.py
	$(PYTHON) tests/test_evidence_workbench.py
	$(PYTHON) tests/test_qualification_session.py
	$(PYTHON) tests/test_kv_crossover_campaign.py

site-build:
	$(PYTHON) tools/build_public_site.py --output $(SITE_DIST)

site-all: site-test site-build
