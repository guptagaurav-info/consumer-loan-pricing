PY ?= python
export PYTHONPATH := src

.PHONY: all ingest run test
all: ingest run
ingest:
	$(PY) -m loanprice.ingest
run:
	$(PY) -m loanprice.pipeline
test:
	$(PY) -m pytest -q tests
