#!/usr/bin/env python3
"""Run the full encrypted-aggregation validation and write reports/*.json|md.

    python scripts/validate_encrypted_aggregation.py            # full
    python scripts/validate_encrypted_aggregation.py --quick    # CI-speed
Exit code 1 if any check fails (used as the CI crypto gate).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm.crypto.validation import run_all, to_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    report = run_all(quick=a.quick)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "encrypted_aggregation_validation.json").write_text(json.dumps(report, indent=2))
    (out / "encrypted_aggregation_validation.md").write_text(to_markdown(report))
    for c in report["checks"]:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['name']}")
    print("OVERALL:", "PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
