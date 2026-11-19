"""Authenticity + anti-replay for node uploads (simulation-grade, stdlib only).

Registration hands each node a random bearer token. Every upload carries an HMAC-SHA256 over
(node_id | round | attempt | sha256(body)) keyed with that token, so the aggregator can tell
that a ciphertext really came from the enrolled node for *this* round attempt, and was not
altered or replayed from an earlier round. This protects integrity/origin; confidentiality of
the weights comes from CKKS, not from this module.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets


def new_token() -> str:
    return secrets.token_hex(24)


def sign_update(token: str, node_id: str, round_id: int, attempt: int, body: bytes) -> str:
    msg = f"{node_id}|{round_id}|{attempt}|".encode() + hashlib.sha256(body).digest()
    return hmac.new(token.encode(), msg, hashlib.sha256).hexdigest()


def verify_update(token: str, node_id: str, round_id: int, attempt: int, body: bytes,
                  signature: str) -> bool:
    expected = sign_update(token, node_id, round_id, attempt, body)
    return hmac.compare_digest(expected, signature or "")


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())
