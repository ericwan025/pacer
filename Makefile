PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: venv data features train test sim bench redis-up redis-down clean

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

# Phase 2: train LR + DeepFM, calibrate, export ONNX + calibrator + transform.
train:
	cd python && ../$(PY) -m pacer.models.train_ctr

# Phase 6: run all strategies x {smooth,bursty}, print tables, write plots.
# Add --synthetic for a demo run without Kaggle data (numbers not for README).
sim:
	cd python && ../$(PY) -m pacer.eval.sim_harness

sim-demo:
	cd python && ../$(PY) -m pacer.eval.sim_harness --synthetic --campaigns 200

test:
	cd python && ../$(PY) -m pytest -q

go-test:
	cd go && go test -race ./...

redis-up:
	docker compose up -d redis

redis-down:
	docker compose down

clean:
	rm -rf python/**/__pycache__ python/.pytest_cache
