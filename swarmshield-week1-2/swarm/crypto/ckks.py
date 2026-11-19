"""CKKS homomorphic encryption layer (TenSEAL / Microsoft SEAL).

Trust split enforced by this module:

  key authority  : creates a context WITH the secret key, exports two blobs
                     secret.ctx -> given to edge nodes only
                     public.ctx -> given to the aggregator (public key, no secret key)
  edge node      : encrypts its weights, decrypts the aggregated model
  aggregator     : can add ciphertexts and multiply them by *plaintext* scalars. It cannot
                   decrypt anything: its context has no secret key (checked on load).

FedAvg needs exactly those two operations:  w_global = sum_k (n_k / N) * w_k  -> a plaintext-
scalar multiplication (consumes 1 of the 2 available levels) plus ciphertext additions.
"""

from __future__ import annotations

import numpy as np
import tenseal as ts


class CryptoError(ValueError):
    pass


def new_secret_context(poly_modulus_degree: int = 8192,
                       coeff_mod_bit_sizes: tuple[int, ...] = (60, 40, 40, 60),
                       scale_bits: int = 40) -> ts.Context:
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly_modulus_degree,
                     coeff_mod_bit_sizes=list(coeff_mod_bit_sizes))
    ctx.global_scale = 2.0 ** scale_bits
    return ctx


def export_secret(ctx: ts.Context) -> bytes:
    return ctx.serialize(save_public_key=True, save_secret_key=True,
                         save_galois_keys=False, save_relin_keys=False)


def export_public(ctx: ts.Context) -> bytes:
    return ctx.serialize(save_public_key=True, save_secret_key=False,
                         save_galois_keys=False, save_relin_keys=False)


def load_context(blob: bytes, *, require_secret: bool | None = None) -> ts.Context:
    """Load a context. ``require_secret=False`` is what the aggregator uses: it refuses to run
    if handed a context that contains the secret key."""
    ctx = ts.context_from(blob)
    has_sk = ctx.has_secret_key()
    if require_secret is False and has_sk:
        raise CryptoError("refusing to load a context that contains the SECRET key "
                          "(the aggregator must only ever hold the public context)")
    if require_secret is True and not has_sk:
        raise CryptoError("this context has no secret key; edge nodes need secret.ctx")
    return ctx


def n_chunks(n_params: int, slots: int) -> int:
    return -(-n_params // slots)


class CKKSCodec:
    """Edge-node side: encrypt a flat weight vector into ciphertext chunks and back."""

    def __init__(self, ctx: ts.Context, n_params: int, poly_modulus_degree: int = 8192):
        self.ctx = ctx
        self.n_params = n_params
        self.slots = poly_modulus_degree // 2  # CKKS packs N/2 real numbers per ciphertext
        self.n_chunks = n_chunks(n_params, self.slots)

    def encrypt(self, vec: np.ndarray) -> list[bytes]:
        vec = np.asarray(vec, dtype=np.float64).ravel()
        if vec.shape[0] != self.n_params:
            raise CryptoError(f"expected {self.n_params} values, got {vec.shape[0]}")
        out = []
        for c in range(self.n_chunks):
            piece = vec[c * self.slots:(c + 1) * self.slots]
            out.append(ts.ckks_vector(self.ctx, piece.tolist()).serialize())
        return out

    def decrypt(self, chunks: list[bytes]) -> np.ndarray:
        if len(chunks) != self.n_chunks:
            raise CryptoError(f"expected {self.n_chunks} chunks, got {len(chunks)}")
        parts = [np.array(ts.ckks_vector_from(self.ctx, b).decrypt(), dtype=np.float64)
                 for b in chunks]
        return np.concatenate(parts)[: self.n_params]


def validate_ciphertext_chunks(pub_ctx: ts.Context, chunks: list[bytes], n_params: int,
                               slots: int) -> None:
    """Cheap structural validation of an upload before it is queued for aggregation."""
    expected = n_chunks(n_params, slots)
    if len(chunks) != expected:
        raise CryptoError(f"expected {expected} ciphertext chunks, got {len(chunks)}")
    for i, blob in enumerate(chunks):
        try:
            vec = ts.ckks_vector_from(pub_ctx, blob)
        except Exception as exc:
            raise CryptoError(f"chunk {i} is not a valid CKKS ciphertext: {exc}") from exc
        want = min(slots, n_params - i * slots)
        if vec.size() != want:
            raise CryptoError(f"chunk {i} has size {vec.size()}, expected {want}")


def aggregate_weighted(pub_ctx: ts.Context, uploads: list[list[bytes]],
                       weights: list[float]) -> list[bytes]:
    """sum_k weights[k] * Enc(w_k), computed entirely on ciphertexts.

    ``weights`` are plaintext scalars (FedAvg: n_k / N). They must sum to 1 so that the result
    is a convex combination and stays inside the CKKS value range.
    """
    if not uploads:
        raise CryptoError("nothing to aggregate")
    if len(uploads) != len(weights):
        raise CryptoError("uploads/weights length mismatch")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise CryptoError(f"weights must sum to 1, got {sum(weights)}")
    n_c = len(uploads[0])
    if any(len(u) != n_c for u in uploads):
        raise CryptoError("uploads have different chunk counts")
    result: list[bytes] = []
    for c in range(n_c):
        acc = None
        for u, w in zip(uploads, weights, strict=True):
            term = ts.ckks_vector_from(pub_ctx, u[c]) * float(w)
            acc = term if acc is None else acc + term
        result.append(acc.serialize())
    return result
