"""Tiny length-prefixed binary framing: [u32 count][u32 len, bytes]*  (no base64 overhead)."""

from __future__ import annotations

import struct


def pack(parts: list[bytes]) -> bytes:
    out = [struct.pack(">I", len(parts))]
    for p in parts:
        out.append(struct.pack(">I", len(p)))
        out.append(p)
    return b"".join(out)


def unpack(data: bytes, max_parts: int = 4096) -> list[bytes]:
    try:
        (n,) = struct.unpack_from(">I", data, 0)
        if n > max_parts:
            raise ValueError("too many parts")
        pos, parts = 4, []
        for _ in range(n):
            (ln,) = struct.unpack_from(">I", data, pos)
            pos += 4
            if pos + ln > len(data):
                raise ValueError("truncated frame")
            parts.append(data[pos:pos + ln])
            pos += ln
        if pos != len(data):
            raise ValueError("trailing bytes")
        return parts
    except struct.error as exc:
        raise ValueError(f"malformed frame: {exc}") from exc
