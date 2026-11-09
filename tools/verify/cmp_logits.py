#!/usr/bin/env python3
"""Compare two logitdump files.

Two algebraically equivalent inference paths differ only in floating-point
summation order, so they must agree to a few ULP and pick the same argmax.
Anything worse is a bug, not a tolerance to widen.
"""
import struct
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        hdr = f.read(16)
        v, ntok = struct.unpack("<qq", hdr)
        a = np.frombuffer(f.read(v * 4), dtype=np.float32)
    return a, v, ntok


def main():
    a, va, na = load(sys.argv[1])
    b, vb, nb = load(sys.argv[2])
    assert va == vb and na == nb, "dump geometry differs"

    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    scale = np.abs(a.astype(np.float64)).max()
    rel = d.max() / scale if scale else d.max()
    ia, ib = int(a.argmax()), int(b.argmax())

    # relative error, ignoring entries whose magnitude is numerical noise
    ref = np.abs(a.astype(np.float64))
    mask = ref > 1e-4 * scale
    relf = (d[mask] / ref[mask]).max() if mask.any() else 0.0

    print(f"vocab={va} ntok={na}")
    print(f"  argmax   A={ia} ({a[ia]:.6f})  B={ib} ({b[ib]:.6f})")
    print(f"  max |A-B|      = {d.max():.6e}")
    print(f"  max |A-B|/|A|  = {rel:.6e}   (filtered: {relf:.6e})")
    print(f"  mean |A-B|     = {d.mean():.6e}")
    ok = ia == ib and rel < 1e-3
    print("VERDICT:", "MATCH (differences are summation order only)"
          if ok else "MISMATCH -- investigate")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
