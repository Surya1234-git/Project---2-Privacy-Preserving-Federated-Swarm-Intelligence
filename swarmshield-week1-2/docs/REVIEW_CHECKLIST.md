# Mid-project review (end of Week 2) - evidence map

| Review item | How to demonstrate | Expected |
|---|---|---|
| Simulated edge-node network operational, holding localized data | `make data NODES=100 ALPHA=0.2 && make compose && make up`; show `data/shards/heterogeneity.png`, `manifest.json` | 100 containers enrol; each mounts one shard; non-IID score ~0.4 |
| Baseline FedAvg syncing weights | `make compose MODE=plain && make up` (or `make local-net MODE=plain`), `runs/latest/history.jsonl` | `eval_acc` rises round over round |
| Encrypted ciphertext aggregation mathematically validated | `make validate`, open `reports/encrypted_aggregation_validation.md` | all sections PASS; error ~1e-7 |
| Aggregator never decrypts | blindness section of the report; `tests/test_crypto.py::test_aggregator_cannot_hold_or_use_secret_key` | PASS |
| Crypto peer review enforced | open a PR touching `swarm/crypto/`: CI fails without `crypto-reviewed` label | gate demonstrated |
| Daily commits | `python scripts/commit_streak.py --start <day1> --days 14` | no MISSING days |

Talking points: ~18x ciphertext expansion, why weights (not gradients) are encrypted each round
(depth never accumulates), why accuracy is evaluated federatedly, and the limitations in
`docs/SECURITY_MODEL.md` (shared key, metadata, no DP yet).
