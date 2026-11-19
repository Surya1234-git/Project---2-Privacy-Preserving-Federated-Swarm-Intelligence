# Security model

## What the design protects
* **Raw data** stays in the node (per-node shard mounted into exactly one container).
* **Individual model updates** are CKKS ciphertexts; the aggregator has only the public context,
  and the code refuses to start with a secret-key context (`tests/test_crypto.py`,
  `swarm.crypto.validation.check_blindness`).
* **Upload authenticity / replay**: HMAC over node, round, attempt and body.
* **Malformed input**: wrong chunk count, wrong slot length, non-ciphertext bytes, NaN weights,
  plaintext in encrypted mode, duplicate / unselected / stale-round uploads are all rejected
  (`tests/test_server_protocol.py`).

## What it does NOT protect (be explicit in your report)
1. **Shared secret key.** All nodes hold the same CKKS secret key. A malicious or compromised node
   can decrypt anything encrypted under it. Production fix: threshold / multi-party HE, so no
   single party can decrypt.
2. **Metadata.** The aggregator sees sample counts, loss/accuracy, who participated and when,
   and ciphertext sizes.
3. **Leakage through the aggregate.** Every node decrypts the global model; membership-inference
   on that model is possible. Mitigation is Differential Privacy (Week 3), not FHE.
4. **"Zero-knowledge".** The brief uses this phrase loosely. FHE gives *confidentiality of updates
   from the aggregator*; it is not a zero-knowledge proof system and gives no proof that a node
   trained honestly. Use "the aggregator learns nothing about individual updates" in the report.
5. **Poisoning.** An authenticated node can still upload a malicious (but valid) update; encrypted
   updates make inspection harder. Byzantine tolerance is Week 3.
6. **Transport.** Plain HTTP + bearer token + HMAC is lab-grade; add TLS and real enrolment (mTLS or
   per-device credentials) for anything real. The enrolment secret is shared.
7. **CKKS is approximate.** Errors are ~1e-7 relative here (measured), far below SGD noise, but
   results are not bit-exact.

## Security-relevant files (require `crypto-reviewed` label to merge)
`swarm/crypto/**`, `swarm/security.py`, `tests/test_crypto.py`
