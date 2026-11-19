# Encrypted aggregation - validation report

**Overall: PASS**

## PASS - homomorphism: Dec(sum p_k Enc(w_k)) == sum p_k w_k

- **worst_relative_err**: `4.360288237137411e-07`
- **tolerance**: `1e-05`

| dim | magnitude | clients | chunks | max abs err |
|---|---|---|---|---|
| 10 | 0.001 | 2 | 1 | 2.86e-09 |
| 10 | 0.001 | 7 | 1 | 3.10e-09 |
| 10 | 1.0 | 2 | 1 | 2.58e-07 |
| 10 | 1.0 | 7 | 1 | 8.80e-08 |
| 10 | 10.0 | 2 | 1 | 2.19e-06 |
| 10 | 10.0 | 7 | 1 | 1.99e-06 |
| 4522 | 0.001 | 2 | 2 | 1.28e-08 |
| 4522 | 0.001 | 7 | 2 | 2.46e-08 |
| 4522 | 1.0 | 2 | 2 | 4.36e-07 |
| 4522 | 1.0 | 7 | 2 | 2.19e-07 |
| 4522 | 10.0 | 2 | 2 | 3.66e-06 |
| 4522 | 10.0 | 7 | 2 | 2.42e-06 |

## PASS - algebraic laws of encrypted FedAvg

- **max_abs_errors**:
  - order_invariance: `0.0`
  - idempotence: `5.905822186846876e-07`
  - linearity: `7.260356227334341e-08`
  - dropout_renormalisation: `3.585391317173503e-07`
- **tolerance**: `4e-05`

## PASS - aggregator blindness & input hardening

- **checks**:
  - public context has no secret key: `True`
  - aggregator context cannot decrypt: `True`
  - aggregator refuses secret context: `True`
  - EncryptedAggregator refuses secret context: `True`
  - public blob smaller than secret blob: `True`
  - rejects garbage bytes: `True`
  - rejects wrong chunk count: `True`
  - rejects truncated ciphertext: `True`
  - ciphertext bytes do not contain the plaintext: `True`

## PASS - training equivalence: plain FedAvg vs encrypted FedAvg

- **rounds**: `6`
- **max_weight_drift**: `7.952808325217831e-07`
- **max_test_acc_gap**: `0.0`
- **final_acc_plain**: `0.5066666666666667`
- **final_acc_fhe**: `0.5066666666666667`

## PASS - cost of encryption (informational)

- **n_params**: `4522`
- **ciphertext_chunks**: `2`
- **ciphertext_bytes**: `661802`
- **plaintext_bytes**: `36176`
- **expansion_factor**: `18.29395179124281`
- **encrypt_s**: `0.010900900000706315`
- **decrypt_s**: `0.0032145000295713544`
