"""Crypto-critical tests. Changes under swarm/crypto/ must keep these green (CI crypto gate)."""

import numpy as np
import pytest

from swarm.crypto import ckks, validation
from swarm.fl.aggregation import EncryptedAggregator, PlainAggregator, fedavg_weights


@pytest.fixture(scope="module")
def keys():
    sk = ckks.new_secret_context()
    return sk, ckks.load_context(ckks.export_public(sk), require_secret=False)


def test_roundtrip_multi_chunk(keys):
    sk, _ = keys
    codec = ckks.CKKSCodec(sk, 9000)
    assert codec.n_chunks == 3
    w = np.random.default_rng(0).normal(size=9000)
    assert np.max(np.abs(codec.decrypt(codec.encrypt(w)) - w)) < 1e-5


def test_fedavg_weights_sum_to_one_exactly():
    for ns in ([1, 2, 3], [7] * 13, [10**6, 1, 1]):
        assert sum(fedavg_weights(ns)) == pytest.approx(1.0, abs=1e-15)
    with pytest.raises(ValueError):
        fedavg_weights([0, 0])


def test_encrypted_fedavg_matches_plaintext(keys):
    sk, pub = keys
    rng = np.random.default_rng(1)
    ws, ns = [rng.normal(size=4522) for _ in range(5)], [10, 200, 35, 70, 5]
    codec = ckks.CKKSCodec(sk, 4522)
    got = codec.decrypt(EncryptedAggregator(pub).aggregate([codec.encrypt(w) for w in ws], ns))
    assert np.max(np.abs(got - PlainAggregator().aggregate(ws, ns))) < 1e-5


def test_aggregator_cannot_hold_or_use_secret_key(keys):
    sk, pub = keys
    with pytest.raises(ckks.CryptoError):
        ckks.load_context(ckks.export_secret(sk), require_secret=False)
    with pytest.raises(ckks.CryptoError):
        EncryptedAggregator(sk)
    with pytest.raises(ckks.CryptoError):
        ckks.load_context(ckks.export_public(sk), require_secret=True)
    codec = ckks.CKKSCodec(sk, 10)
    import tenseal as ts

    with pytest.raises(ValueError):
        ts.ckks_vector_from(pub, codec.encrypt(np.ones(10))[0]).decrypt()


def test_aggregate_rejects_bad_inputs(keys):
    sk, pub = keys
    codec = ckks.CKKSCodec(sk, 20)
    ct = codec.encrypt(np.ones(20))
    with pytest.raises(ckks.CryptoError):
        ckks.aggregate_weighted(pub, [ct, ct], [0.9, 0.2])      # weights != 1
    with pytest.raises(ckks.CryptoError):
        ckks.aggregate_weighted(pub, [], [])
    with pytest.raises(ckks.CryptoError):
        ckks.aggregate_weighted(pub, [ct, ct + ct], [0.5, 0.5])  # chunk count mismatch


def test_ciphertext_validation_rejects_malformed(keys):
    sk, pub = keys
    codec = ckks.CKKSCodec(sk, 50)
    good = codec.encrypt(np.ones(50))
    ckks.validate_ciphertext_chunks(pub, good, 50, 4096)
    for bad in ([b"junk"], good + good, [good[0][:100]]):
        with pytest.raises(ckks.CryptoError):
            ckks.validate_ciphertext_chunks(pub, bad, 50, 4096)
    other = ckks.CKKSCodec(sk, 10).encrypt(np.ones(10))   # right format, wrong length
    with pytest.raises(ckks.CryptoError):
        ckks.validate_ciphertext_chunks(pub, other, 50, 4096)


def test_validation_suite_quick():
    assert validation.check_homomorphism(trials=4)["passed"]
    assert validation.check_algebraic_laws()["passed"]
    assert validation.check_blindness()["passed"]
