PY ?= python3
NODES ?= 100
ALPHA ?= 0.2
MODE ?= fhe
ROUNDS ?= 15

.PHONY: setup test test-all lint data sim validate local-net compose up down plain-up hooks streak demo clean

setup:            ## create venv + install deps + enable git hooks
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
	-sh scripts/install_hooks.sh

lint:
	$(PY) -m ruff check .

test:             ## fast tests (no spawned processes)
	$(PY) -m pytest -m "not integration"

test-all:         ## everything, incl. real aggregator + node processes
	$(PY) -m pytest

data:             ## non-IID shards + heterogeneity report: make data NODES=200 ALPHA=0.1
	$(PY) -m swarm.data.prepare --nodes $(NODES) --alpha $(ALPHA) --out data/shards

sim:              ## in-process plain-vs-encrypted comparison (no docker)
	$(PY) -m swarm.sim.run --shards data/shards --mode both --rounds $(ROUNDS)

validate:         ## Week-2 evidence: mathematical validation of encrypted aggregation
	$(PY) scripts/validate_encrypted_aggregation.py

local-net:        ## real aggregator + node processes on localhost
	$(PY) -m swarm.sim.local_network --shards data/shards --nodes 12 --rounds 5 --mode $(MODE)

compose:          ## generate docker-compose for the shards in data/shards
	$(PY) scripts/gen_compose.py --shards data/shards --mode $(MODE) --rounds $(ROUNDS) --out docker-compose.yml

up:               ## build + run the container swarm (needs `make data compose` first)
	mkdir -p runs && chmod 777 runs
	docker compose up --build --abort-on-container-exit --exit-code-from aggregator

down:
	docker compose down -v

demo:             ## shards -> compose -> swarm, 12 nodes
	$(MAKE) data NODES=12 ALPHA=0.3
	$(MAKE) compose
	$(MAKE) up

hooks:
	sh scripts/install_hooks.sh

streak:           ## daily-commit audit: make streak START=2026-11-20
	$(PY) scripts/commit_streak.py --start $(START) --days 14

clean:
	rm -rf runs/* keys data/shards .pytest_cache .ruff_cache
