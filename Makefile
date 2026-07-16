PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: venv data features test sim bench redis-up redis-down clean

venv:
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e "python[dev]"

# Download + subsample Avazu into data/splits. Needs Kaggle creds; fails loud otherwise.
data:
	cd python && ../$(PY) -m pacer.data.download

# Fit the feature transform on train, write artifact + Phase 1 report/plots.
features:
	cd python && ../$(PY) -m pacer.data.report

test:
	cd python && ../$(PY) -m pytest -q

redis-up:
	docker compose up -d redis

redis-down:
	docker compose down

clean:
	rm -rf python/**/__pycache__ python/.pytest_cache
