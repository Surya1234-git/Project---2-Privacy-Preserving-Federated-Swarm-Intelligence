"""Key authority (simulation): create the CKKS keys once and split them by trust domain.

    python -m swarm.crypto.keygen --public keys/public.ctx --secret keys/secret.ctx

In docker-compose this runs as a one-shot ``keygen`` service writing into two *separate*
named volumes; the aggregator mounts only the public one. Production note: a single secret
key shared by all nodes is a simulation shortcut - see docs/SECURITY_MODEL.md (threshold /
multi-party key generation is the real-world answer).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from swarm.config import ExperimentConfig
from swarm.crypto import ckks


def generate(public_path: Path, secret_path: Path, cfg: ExperimentConfig | None = None) -> None:
    cfg = cfg or ExperimentConfig.from_env()
    ctx = ckks.new_secret_context(cfg.poly_modulus_degree, cfg.coeff_mod_bit_sizes, cfg.scale_bits)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_bytes(ckks.export_public(ctx))
    secret_path.write_bytes(ckks.export_secret(ctx))
    os.chmod(secret_path, 0o644 if os.environ.get("SWARM_KEYS_WORLD_READABLE") else 0o600)
    # world-readable is needed inside docker where nodes run as a different uid than keygen
    print(f"[keygen] public -> {public_path} ({public_path.stat().st_size} B), "
          f"secret -> {secret_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default="keys/public.ctx")
    ap.add_argument("--secret", default="keys/secret.ctx")
    a = ap.parse_args()
    generate(Path(a.public), Path(a.secret))


if __name__ == "__main__":
    main()
