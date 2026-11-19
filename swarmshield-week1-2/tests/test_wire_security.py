import pytest

from swarm import security, wire


def test_wire_roundtrip_and_rejects_garbage():
    parts = [b"", b"abc", b"\x00" * 1000]
    assert wire.unpack(wire.pack(parts)) == parts
    for bad in (b"", b"\x00\x00\x00\x05", wire.pack(parts)[:-1], wire.pack(parts) + b"x"):
        with pytest.raises(ValueError):
            wire.unpack(bad)


def test_signature_binds_node_round_attempt_and_body():
    tok = security.new_token()
    body = b"ciphertext"
    sig = security.sign_update(tok, "node-1", 3, 0, body)
    assert security.verify_update(tok, "node-1", 3, 0, body, sig)
    assert not security.verify_update(tok, "node-2", 3, 0, body, sig)   # other node
    assert not security.verify_update(tok, "node-1", 4, 0, body, sig)   # replay in later round
    assert not security.verify_update(tok, "node-1", 3, 1, body, sig)   # other attempt
    assert not security.verify_update(tok, "node-1", 3, 0, body + b"x", sig)  # tampered
    assert not security.verify_update("other-token", "node-1", 3, 0, body, sig)
    assert not security.verify_update(tok, "node-1", 3, 0, body, "")
