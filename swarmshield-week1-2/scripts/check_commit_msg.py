#!/usr/bin/env python3
"""Commit-message policy (used by the commit-msg hook AND by CI over a PR's commit range).

Format:   type(scope): subject
  type   feat fix docs test refactor perf build ci chore revert
  scope  data model fl crypto server node sim docker ci docs security deps repo
  rules  subject 5-72 chars, starts lowercase, no trailing period, blank line before body

Cryptography rule: any commit that touches swarm/crypto/ (or tests/test_crypto.py) MUST use
scope `crypto`, so cryptographic changes are always visible in history and in review filters.

    check_commit_msg.py .git/COMMIT_EDITMSG            # hook mode (also inspects staged files)
    check_commit_msg.py --range origin/main..HEAD      # CI mode (inspects each commit's files)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

TYPES = "feat|fix|docs|test|refactor|perf|build|ci|chore|revert"
SCOPES = {"data", "model", "fl", "crypto", "server", "node", "sim", "docker", "ci", "docs",
          "security", "deps", "repo"}
PATTERN = re.compile(rf"^(?P<type>{TYPES})\((?P<scope>[a-z-]+)\): (?P<subject>.+)$")
CRYPTO_PATHS = ("swarm/crypto/", "tests/test_crypto.py")


def check_message(msg: str, files: list[str] | None = None) -> list[str]:
    lines = [ln for ln in msg.splitlines() if not ln.startswith("#")]
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ["empty commit message"]
    head = lines[0]
    if head.startswith(("Merge ", "Revert \"")):
        return []
    errors: list[str] = []
    m = PATTERN.match(head)
    if not m:
        return [f"header must look like 'type(scope): subject' with type in [{TYPES}]; got: {head!r}"]
    if m["scope"] not in SCOPES:
        errors.append(f"unknown scope {m['scope']!r}; allowed: {sorted(SCOPES)}")
    subj = m["subject"]
    if not 5 <= len(subj) <= 72:
        errors.append(f"subject must be 5-72 characters (is {len(subj)})")
    if subj[0].isupper():
        errors.append("subject must start lowercase")
    if subj.endswith("."):
        errors.append("subject must not end with a period")
    if len(lines) > 1 and lines[1].strip():
        errors.append("leave a blank line between header and body")
    if files and any(f.startswith(CRYPTO_PATHS) for f in files) and m["scope"] != "crypto":
        errors.append("this commit touches cryptographic code, so its scope must be `crypto`")
    return errors


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("msgfile", nargs="?")
    ap.add_argument("--range")
    a = ap.parse_args()
    bad = 0
    if a.range:
        for sha in _git("rev-list", "--no-merges", a.range).split():
            msg = _git("log", "-1", "--format=%B", sha)
            files = _git("diff-tree", "--no-commit-id", "--name-only", "-r", sha).split()
            errs = check_message(msg, files)
            if errs:
                bad += 1
                print(f"commit {sha[:8]}: {msg.splitlines()[0]!r}")
                print("\n".join(f"  - {e}" for e in errs))
    elif a.msgfile:
        msg = open(a.msgfile, encoding="utf-8").read()
        try:
            files = _git("diff", "--cached", "--name-only").split()
        except subprocess.CalledProcessError:
            files = []
        errs = check_message(msg, files)
        if errs:
            bad = 1
            print("Commit rejected by policy (see docs/GIT_WORKFLOW.md):")
            print("\n".join(f"  - {e}" for e in errs))
    else:
        ap.error("give a message file or --range")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
