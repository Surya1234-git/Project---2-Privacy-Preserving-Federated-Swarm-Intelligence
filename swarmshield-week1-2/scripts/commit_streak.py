#!/usr/bin/env python3
"""Daily-commit audit. The programme evaluates you on daily commits, so check them yourself.

    python scripts/commit_streak.py --start 2026-11-20 --days 14
    python scripts/commit_streak.py --start 2026-11-20 --days 14 --skip-weekends

Prints one line per calendar day with the number of commits, and exits 1 if any day up to today
has none. Reads `git log` only - it reports the truth, it cannot and should not alter history.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from datetime import date, timedelta


def commit_days(since: date, until: date, author: str | None = None) -> Counter:
    cmd = ["git", "log", "--all", f"--since={since.isoformat()} 00:00",
           f"--until={until.isoformat()} 23:59:59", "--format=%ad", "--date=short"]
    if author:
        cmd.append(f"--author={author}")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split()
    return Counter(out)


def audit(start: date, days: int, today: date, counts: Counter, skip_weekends: bool) -> list[tuple]:
    rows = []
    for i in range(days):
        d = start + timedelta(days=i)
        if d > today:
            break
        skipped = skip_weekends and d.weekday() >= 5
        rows.append((d, counts.get(d.isoformat(), 0), skipped))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="first project day, YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--author")
    ap.add_argument("--skip-weekends", action="store_true")
    a = ap.parse_args()
    start = date.fromisoformat(a.start)
    today = date.today()
    counts = commit_days(start, start + timedelta(days=a.days), a.author)
    missing = 0
    for d, n, skipped in audit(start, a.days, today, counts, a.skip_weekends):
        flag = "weekend" if skipped else ("ok" if n else "MISSING")
        missing += (flag == "MISSING")
        print(f"{d} {d.strftime('%a')}  {n:3d} commit(s)  {flag}")
    print(f"\n{missing} day(s) without a commit")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
