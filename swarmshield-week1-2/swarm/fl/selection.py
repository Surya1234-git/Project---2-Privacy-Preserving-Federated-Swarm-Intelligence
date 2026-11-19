"""Deterministic client sampling shared by the aggregator and the in-process simulator."""

from __future__ import annotations

import math

import numpy as np


def select_participants(node_ids: list[str], participation: float, min_selected: int,
                        seed: int, round_id: int, attempt: int) -> list[str]:
    ids = sorted(node_ids)
    k = min(len(ids), max(min_selected, math.ceil(participation * len(ids))))
    rng = np.random.default_rng([seed, round_id, attempt])
    pick = rng.choice(len(ids), size=k, replace=False)
    return sorted(ids[i] for i in pick)


def dropout_draw(node_id_crc: int, round_id: int, attempt: int, p: float) -> bool:
    """True => this device silently vanishes this round (separate RNG stream from training)."""
    return bool(np.random.default_rng([node_id_crc, round_id, attempt, 99]).random() < p)
