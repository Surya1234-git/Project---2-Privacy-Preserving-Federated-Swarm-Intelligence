# Week 1 commit plan (daily commits, own branch each)

The SOP requires real commit activity on every day before a review (10 different days before
the Mid Review, **all 20 days before the Final Review, no gaps**). Don't use empty or
back-dated commits - evaluators can see them. Instead, split the work below into small,
genuine commits so there is *something real* to commit every day.

Team Lead: create the repo first, add everyone, protect `main`, and give each member their own
branch (`feat/<name>`). Members commit to their own branch; the Team Lead merges to `main`.
Use clear messages (`feat:`, `fix:`, `test:`, `docs:`, `ci:`).

| Day | Focus | Suggested commits |
|-----|-------|-------------------|
| 1 | Repo + data split | `chore: init repo, .gitignore, requirements` / `feat: add Dirichlet/shard/IID partitioning` / `test: partition coverage and non-IID checks` |
| 2 | Model + local training | `feat: add SmallCNN` / `feat: local_train with optional FedProx term` / `feat: evaluate()` |
| 3 | Aggregation | `feat: weighted FedAvg` / `feat: pickle-free weight serialization` / `test: fedavg and serialization` |
| 4 | Server | `feat: coordinator round state machine` / `feat: Flask API (register/task/model/update)` / `feat: update validation (shape, NaN)` / `test: server protocol` |
| 5 | Edge clients + Docker | `feat: client container hosting multiple logical nodes` / `build: Dockerfile (CPU torch)` / `build: docker-compose swarm` |
| 6 | Experiments | `feat: in-process simulation` / `feat: plots (partition heatmap, accuracy curves)` / `docs: first results (IID vs non-IID)` |
| 7 | Polish + review prep | `ci: GitHub Actions` / `docs: README + architecture` / `docs: work distribution` / `fix: ...` |

Suggested split for 4 people: **(1)** repo/CI/merges + docs, **(2)** partitioning + plots,
**(3)** model/training/FedAvg, **(4)** server/client protocol + Docker.
Each person should be able to explain the whole pipeline in the review, not just their part.

Before the review: run `pytest -q`, run `docker compose up --build`, and keep `results/` screenshots
(`partition_heatmap.png`, `fedavg_curve.png`, `summary.json`) ready to show.
