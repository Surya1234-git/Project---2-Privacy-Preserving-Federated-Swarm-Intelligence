"""Safe (pickle-free) wire format for model weights: a compressed .npz in memory."""
from __future__ import annotations

import io

import numpy as np


def weights_to_bytes(weights: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **weights)
    return buf.getvalue()


def bytes_to_weights(data: bytes) -> dict:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
