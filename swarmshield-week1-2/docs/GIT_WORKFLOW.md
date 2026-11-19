# Git workflow (strict mode)

## Setup (once)
```
git init && git add . && git commit -m "chore(repo): import swarmshield scaffold"
sh scripts/install_hooks.sh          # enables commit-msg + pre-push hooks
pre-commit install --hook-type pre-commit --hook-type commit-msg   # optional, richer hooks
```

## Branches
* `main`   - protected, always green, only receives PRs from `develop` (or hotfixes)
* `develop`- integration branch
* `feat/<scope>-<topic>` e.g. `feat/data-dirichlet`, `feat/node-docker`
* `crypto/<topic>` - anything under `swarm/crypto/`; always needs a second reviewer
* `test/<topic>`, `fix/<topic>`, `docs/<topic>`

## Commit format (enforced by hook and by CI over every commit in a PR)
`type(scope): subject`
* type: feat fix docs test refactor perf build ci chore revert
* scope: data model fl crypto server node sim docker ci docs security deps repo
* subject: 5-72 chars, lowercase start, no trailing period; blank line before any body
* a commit touching `swarm/crypto/` must use scope `crypto`

Good: `feat(crypto): add chunked ckks codec for multi-ciphertext models`

## GitHub settings to enable (these cannot be stored in the repo)
Settings -> Branches -> protect `main` and `develop`:
* require a pull request with **1 approving review**, and **Require review from Code Owners**
* require status checks: `commit-policy`, `lint`, `test (3.11)`, `test (3.12)`, `crypto-gate`, `docker`
* create the label `crypto-reviewed`; the reviewer applies it after approving a crypto PR
  (CI fails any crypto PR without it)
* edit `.github/CODEOWNERS` with real handles

## Daily commit audit
`python scripts/commit_streak.py --start YYYY-MM-DD --days 14` lists every day and flags days
without commits. It only reads `git log`; commit honestly as you work.
