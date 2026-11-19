import importlib.util
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cm = _load("check_commit_msg")
streak = _load("commit_streak")


def test_good_messages_pass():
    for m in ("feat(data): add capacity-aware dirichlet partitioner",
              "fix(crypto): reject ciphertext chunks with wrong slot count\n\nbody text here",
              "Merge branch 'develop' into main"):
        assert cm.check_message(m) == [], m


def test_bad_messages_fail():
    for m in ("updated stuff", "feat: no scope here", "feat(nope): unknown scope used",
              "feat(data): Starts uppercase", "feat(data): ends with a period.",
              "feat(data): x", "feat(data): ok subject\nno blank line", ""):
        assert cm.check_message(m), m


def test_crypto_files_force_crypto_scope():
    assert cm.check_message("feat(node): tweak things around", ["swarm/crypto/ckks.py"])
    assert not cm.check_message("feat(crypto): tweak things around", ["swarm/crypto/ckks.py"])
    assert not cm.check_message("feat(node): tweak things around", ["swarm/node/client.py"])


def test_streak_audit_flags_missing_days():
    counts = Counter({"2026-11-20": 2, "2026-11-22": 1})
    rows = streak.audit(date(2026, 11, 20), 4, date(2026, 11, 23), counts, skip_weekends=False)
    assert [(r[1]) for r in rows] == [2, 0, 1, 0]
    rows = streak.audit(date(2026, 11, 20), 4, date(2026, 11, 21), counts, skip_weekends=False)
    assert len(rows) == 2  # future days are not judged
    sat = streak.audit(date(2026, 11, 21), 1, date(2026, 11, 22), Counter(), skip_weekends=True)
    assert sat[0][2] is True  # 2026-11-21 is a Saturday


def test_streak_reads_real_git_log(tmp_path):
    env = {"GIT_AUTHOR_DATE": "2026-11-20T10:00:00", "GIT_COMMITTER_DATE": "2026-11-20T10:00:00",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "PATH": __import__("os").environ["PATH"]}
    run = lambda *a: subprocess.run(a, cwd=tmp_path, env=env, check=True, capture_output=True)  # noqa: E731
    run("git", "init", "-q")
    (tmp_path / "f").write_text("x")
    run("git", "add", "f")
    run("git", "commit", "-q", "-m", "feat(repo): first commit of project")
    import os

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        c = streak.commit_days(date(2026, 11, 20), date(2026, 11, 21))
    finally:
        os.chdir(cwd)
    assert c["2026-11-20"] == 1
