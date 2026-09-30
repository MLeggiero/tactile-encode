PY ?= .venv/bin/python

.PHONY: venv install assets test test-all lint strikes calibrate replay

venv:
	python3 -m venv .venv

install:
	$(PY) -m pip install -e .[dev,gym]

assets:
	$(PY) -m tactile_sim.assets.fetch_menagerie

test:
	$(PY) -m pytest -q -m "not network"

test-all:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

strikes:
	$(PY) -m tactile_sim.run_strikes --n 10 --out runs/demo.h5 --seed 0

calibrate:
	$(PY) -m tactile_sim.calibrate pulse

replay:
	$(PY) -m tactile_sim.viewer.export --n 8 --preset default --preset low-torsion --preset weak-grip --out runs/replay.html
