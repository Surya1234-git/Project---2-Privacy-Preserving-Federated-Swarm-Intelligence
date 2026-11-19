# Architecture

## Components

| Component | Process | Holds | Never holds |
|---|---|---|---|
| Key authority (`swarm.crypto.keygen`) | one-shot | both keys (briefly) | data |
| Aggregator (`swarm.server`) | 1 | **public** context, ciphertexts, metadata | secret key, raw data, plaintext weights (fhe mode) |
| Edge node (`swarm.node.client`) | N | its own shard, secret context | other nodes' data |

## Round protocol (semi-synchronous)

```
waiting --(expected_nodes registered, or min_nodes + start_grace)--> collecting r=1
collecting r --(all reported | quorum+grace | deadline+quorum)--> aggregating --> collecting r+1
collecting r --(deadline, no quorum)--> collecting r (attempt+1, fresh sample)
after round R --> eval (every node scores global v_R) --> finished
```

Per selected node in round r:
1. `GET /global` -> ciphertext of global model v(r-1) (204 for v0: every node derives the same
   seeded initial model, nothing needs to be sent).
2. decrypt, score on local validation data (metric reported with the upload).
3. device emulation: dropout draw, straggler slowdown.
4. local SGD (optionally FedProx), encrypt weights **on the device**, sign, `POST /update`.

Aggregator: `w_global = sum_k (n_k / N) * Enc(w_k)` over the nodes that reported (plaintext-scalar
multiplication + ciphertext addition; no ciphertext x ciphertext multiplication, so no relinearisation
keys or rotations are needed and the circuit depth is 1 of the 2 available levels).

## Parameter packing

CKKS packs N/2 = 4096 reals per ciphertext (N=8192). A model with P parameters uses
ceil(P/4096) ciphertexts; the default 4,522-parameter MLP needs 2. Every round starts from fresh
client encryptions, so noise/levels never accumulate across rounds.

CKKS parameters: N=8192, modulus chain [60,40,40,60] (200 bits, within the 128-bit-security bound
of 218 bits for this N in the HE standard), scale 2^40.

## Determinism

Selection, dropout, local training and initialisation are all seeded from
(seed, node_id, round, attempt), so plain and encrypted runs are comparable step by step.

## Where Week 3 plugs in

* gradient quantisation / sparsification: transform the flat vector in `LocalTrainer` before
  `CKKSCodec.encrypt` (fewer ciphertext chunks).
* Differential privacy: clip + noise the flat update in `LocalTrainer.train`.
* Byzantine robustness: cannot inspect ciphertexts, so score updates client-side or via
  secure-aggregation-compatible statistics; hook point is `Coordinator._aggregate`.
