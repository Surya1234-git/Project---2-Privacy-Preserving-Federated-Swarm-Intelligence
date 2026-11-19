"""Mathematical validation of encrypted FedAvg (the Week-2 mid-review evidence).

Each check returns a dict {name, passed, ...measurements}. ``run_all`` aggregates them into one
report. Nothing here is mocked: real CKKS ciphertexts, real aggregator code.

What is validated
  1. Homomorphism      Dec(sum_k p_k * Enc(w_k)) == sum_k p_k * w_k   within CKKS precision
  2. Algebraic laws    order invariance, idempotence (all-equal inputs), linearity, dropout
                       renormalisation (aggregating a subset == FedAvg over that subset)
  3. Blindness         the aggregator context cannot decrypt, never contains the secret key,
                       refuses to start with one, and rejects malformed ciphertexts
  4. Equivalence       full training runs, plain vs encrypted, same seeds: trajectories match
  5. Cost              measured ciphertext expansion factor vs plaintext float64
"""

from __future__ import annotations

import itertools
import tempfile
import time
from pathlib import Path

import numpy as np

from swarm.config import ExperimentConfig
from swarm.crypto import ckks
from swarm.fl.aggregation import EncryptedAggregator, PlainAggregator

# CKKS(N=8192, scale 2^40) typically errs ~1e-7 on O(1) data; we allow 100x headroom.
ATOL = 1e-5


def _setup(cfg: ExperimentConfig | None = None):
    cfg = cfg or ExperimentConfig()
    sk = ckks.new_secret_context(cfg.poly_modulus_degree, cfg.coeff_mod_bit_sizes, cfg.scale_bits)
    pub = ckks.load_context(ckks.export_public(sk), require_secret=False)
    return cfg, sk, pub


def check_homomorphism(trials: int = 12, seed: int = 0) -> dict:
    cfg, sk, pub = _setup()
    rng = np.random.default_rng(seed)
    agg, plain = EncryptedAggregator(pub), PlainAggregator()
    worst, rows = 0.0, []
    for dim, mag, k in itertools.product((10, 4522, 9000), (1e-3, 1.0, 10.0), (2, 7)):
        if len(rows) >= trials:
            break
        codec = ckks.CKKSCodec(sk, dim, cfg.poly_modulus_degree)
        ws = [rng.normal(0, mag, dim) for _ in range(k)]
        ns = [int(x) for x in rng.integers(1, 5000, k)]
        got = codec.decrypt(agg.aggregate([codec.encrypt(w) for w in ws], ns))
        err = float(np.max(np.abs(got - plain.aggregate(ws, ns))))
        worst = max(worst, err / max(1.0, mag))
        rows.append({"dim": dim, "magnitude": mag, "clients": k, "chunks": codec.n_chunks,
                     "max_abs_err": err})
    return {"name": "homomorphism: Dec(sum p_k Enc(w_k)) == sum p_k w_k", "passed": worst < ATOL,
            "worst_relative_err": worst, "tolerance": ATOL, "trials": rows}


def check_algebraic_laws(seed: int = 1) -> dict:
    cfg, sk, pub = _setup()
    rng = np.random.default_rng(seed)
    dim, k = 4522, 6
    codec = ckks.CKKSCodec(sk, dim, cfg.poly_modulus_degree)
    agg = EncryptedAggregator(pub)
    ws = [rng.normal(size=dim) for _ in range(k)]
    ns = [int(x) for x in rng.integers(5, 900, k)]
    cts = [codec.encrypt(w) for w in ws]
    base = codec.decrypt(agg.aggregate(cts, ns))

    perm = rng.permutation(k)
    permuted = codec.decrypt(agg.aggregate([cts[i] for i in perm], [ns[i] for i in perm]))
    order_err = float(np.max(np.abs(base - permuted)))

    same = rng.normal(size=dim)
    ct_same = codec.encrypt(same)
    idem = codec.decrypt(agg.aggregate([ct_same] * 4, [10, 20, 30, 40]))
    idem_err = float(np.max(np.abs(idem - same)))

    alpha = 3.7
    scaled = codec.decrypt(agg.aggregate([codec.encrypt(alpha * w) for w in ws], ns))
    lin_err = float(np.max(np.abs(scaled - alpha * base)))

    sub = [0, 2, 5]
    drop = codec.decrypt(agg.aggregate([cts[i] for i in sub], [ns[i] for i in sub]))
    ref = PlainAggregator().aggregate([ws[i] for i in sub], [ns[i] for i in sub])
    drop_err = float(np.max(np.abs(drop - ref)))

    errs = {"order_invariance": order_err, "idempotence": idem_err, "linearity": lin_err,
            "dropout_renormalisation": drop_err}
    return {"name": "algebraic laws of encrypted FedAvg", "passed": max(errs.values()) < ATOL * 4,
            "max_abs_errors": errs, "tolerance": ATOL * 4}


def check_blindness() -> dict:
    cfg, sk, pub = _setup()
    codec = ckks.CKKSCodec(sk, 100, cfg.poly_modulus_degree)
    chunks = codec.encrypt(np.arange(100, dtype=float))
    results: dict[str, bool] = {}

    results["public context has no secret key"] = not pub.has_secret_key()
    try:
        import tenseal as ts

        ts.ckks_vector_from(pub, chunks[0]).decrypt()
        results["aggregator context cannot decrypt"] = False
    except Exception:
        results["aggregator context cannot decrypt"] = True
    try:
        ckks.load_context(ckks.export_secret(sk), require_secret=False)
        results["aggregator refuses secret context"] = False
    except ckks.CryptoError:
        results["aggregator refuses secret context"] = True
    try:
        EncryptedAggregator(sk)
        results["EncryptedAggregator refuses secret context"] = False
    except ckks.CryptoError:
        results["EncryptedAggregator refuses secret context"] = True
    blob = ckks.export_public(sk)
    results["public blob smaller than secret blob"] = len(blob) < len(ckks.export_secret(sk))
    for label, bad in {"garbage bytes": [b"not a ciphertext"],
                       "wrong chunk count": chunks + chunks,
                       "truncated ciphertext": [chunks[0][:200]]}.items():
        try:
            ckks.validate_ciphertext_chunks(pub, bad, 100, cfg.poly_modulus_degree // 2)
            results[f"rejects {label}"] = False
        except ckks.CryptoError:
            results[f"rejects {label}"] = True
    # a ciphertext must not reveal its plaintext: serialized bytes carry no float64 pattern
    raw = np.arange(100, dtype=np.float64).tobytes()
    results["ciphertext bytes do not contain the plaintext"] = raw not in chunks[0]
    return {"name": "aggregator blindness & input hardening", "passed": all(results.values()),
            "checks": results}


def check_end_to_end_equivalence(rounds: int = 6) -> dict:
    from swarm.data.prepare import prepare
    from swarm.sim.engine import run_simulation

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        prepare(d, dataset="synthetic", n_samples=5000, nodes=16, alpha=0.3, seed=3, min_size=40)
        res = {}
        for mode in ("plain", "fhe"):
            cfg = ExperimentConfig(mode=mode, rounds=rounds, participation=0.5, min_selected=3)
            res[mode] = run_simulation(d, cfg, mode=mode, keep_weights=True)
    drift = [float(np.max(np.abs(a - b))) for a, b in
             zip(res["plain"].global_weights, res["fhe"].global_weights, strict=True)]
    acc_gap = max(abs(a["test_acc"] - b["test_acc"]) for a, b in
                  zip(res["plain"].history, res["fhe"].history, strict=True))
    return {"name": "training equivalence: plain FedAvg vs encrypted FedAvg",
            "passed": max(drift) < 1e-4 and acc_gap < 0.01,
            "rounds": rounds, "max_weight_drift_per_round": drift,
            "max_weight_drift": max(drift), "max_test_acc_gap": acc_gap,
            "final_acc_plain": res["plain"].final_acc, "final_acc_fhe": res["fhe"].final_acc}


def check_cost(n_params: int = 4522) -> dict:
    cfg, sk, pub = _setup()
    codec = ckks.CKKSCodec(sk, n_params, cfg.poly_modulus_degree)
    w = np.random.default_rng(0).normal(size=n_params)
    t0 = time.perf_counter()
    ct = codec.encrypt(w)
    enc_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    codec.decrypt(ct)
    dec_s = time.perf_counter() - t0
    size = sum(len(c) for c in ct)
    return {"name": "cost of encryption (informational)", "passed": True, "n_params": n_params,
            "ciphertext_chunks": len(ct), "ciphertext_bytes": size, "plaintext_bytes": n_params * 8,
            "expansion_factor": size / (n_params * 8), "encrypt_s": enc_s, "decrypt_s": dec_s}


def run_all(quick: bool = False) -> dict:
    checks = [check_homomorphism(trials=6 if quick else 12), check_algebraic_laws(),
              check_blindness(), check_end_to_end_equivalence(rounds=4 if quick else 6),
              check_cost()]
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def to_markdown(report: dict) -> str:
    lines = ["# Encrypted aggregation - validation report", "",
             f"**Overall: {'PASS' if report['passed'] else 'FAIL'}**", ""]
    for c in report["checks"]:
        lines += [f"## {'PASS' if c['passed'] else 'FAIL'} - {c['name']}", ""]
        for k, v in c.items():
            if k in ("name", "passed", "trials", "max_weight_drift_per_round"):
                continue
            if isinstance(v, dict):
                lines.append(f"- **{k}**:")
                lines += [f"  - {kk}: `{vv}`" for kk, vv in v.items()]
            else:
                lines.append(f"- **{k}**: `{v}`")
        if "trials" in c:
            lines += ["", "| dim | magnitude | clients | chunks | max abs err |", "|---|---|---|---|---|"]
            lines += [f"| {t['dim']} | {t['magnitude']} | {t['clients']} | {t['chunks']} | "
                      f"{t['max_abs_err']:.2e} |" for t in c["trials"]]
        lines.append("")
    return "\n".join(lines)
