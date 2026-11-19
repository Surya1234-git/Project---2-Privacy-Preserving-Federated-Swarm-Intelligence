## What & why
<!-- one paragraph -->

## Type
- [ ] feat  - [ ] fix  - [ ] docs  - [ ] test  - [ ] refactor  - [ ] ci

## Checklist
- [ ] Commits follow `type(scope): subject` (hook enforced)
- [ ] `make lint test` passes locally
- [ ] No secrets / key material / raw datasets committed
- [ ] Docs updated if behaviour changed

## Cryptography changes (touches `swarm/crypto/**`, `swarm/security.py`)
- [ ] `python scripts/validate_encrypted_aggregation.py` passes (attach `reports/*.md`)
- [ ] Aggregator still never receives the secret key (blindness check green)
- [ ] Reviewed by a second person -> maintainer adds label `crypto-reviewed`
