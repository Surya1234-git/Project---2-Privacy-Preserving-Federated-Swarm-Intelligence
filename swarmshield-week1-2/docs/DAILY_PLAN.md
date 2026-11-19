# Commit plan for Weeks 1-2 (15 commit-sized steps over 14 days)

The code in this repository is complete; the evaluation counts daily commits. Add and commit it
in the order below **as you read, run and understand each part** - you will be asked to explain it
in review. Each line is one commit-sized step. (Do not backdate commits.)

## Week 1
1. `chore(repo): add project scaffold, requirements and gitignore`  (pyproject, requirements, .gitignore)
2. `feat(sim): add device-class profiles for the edge swarm`  (swarm/profiles.py)
3. `feat(model): add numpy mlp with flat parameter interface`  (+ tests/test_model.py)
4. `feat(data): add capacity-aware dirichlet partitioner`  (partition.py + tests)
5. `feat(data): add shard writer and heterogeneity report`  (prepare.py, datasets.py)
6. `feat(fl): add fedavg aggregation and client selection`  (aggregation.py PlainAggregator, selection.py)
7. `feat(sim): add in-process swarm simulator`  (sim/engine.py, sim/run.py, tests/test_simulation.py)
8. `feat(server): add round coordinator and aggregator api`  (server/*, wire.py)
9. `feat(node): add edge node client and local trainer`  (node/*)
10. `feat(docker): add dockerfile and compose generator`  (Dockerfile, scripts/gen_compose.py)

## Week 2
11. `feat(crypto): add ckks codec and key authority`  (crypto/ckks.py, keygen.py)
12. `feat(crypto): add encrypted weighted aggregation`  (EncryptedAggregator, tests/test_crypto.py)
13. `feat(security): sign uploads and validate ciphertexts`  (security.py, coordinator checks, protocol tests)
14. `ci(ci): add commit policy, crypto gate and docker jobs`  (.github, hooks, Makefile)
15. `docs(docs): add validation report and security model`  (reports/, docs/)

Run `make test-all` before each push.
