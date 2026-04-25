#!/usr/bin/env python3
"""
For each feature: find the single CPU border that MLX is missing, and report
its position in CPU's sorted grid.

If the missing border for feature 0 sits below 0.014 (the L4-verdict reported
split value), then MLX bin=64 corresponds to a different physical interval
than CPU bin=3. That's the iter≥2 divergence vector.
"""

import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"


def load(path):
    by_feat = {}
    for line in (DATA / path).read_text().splitlines():
        if not line.strip():
            continue
        f, v = line.split("\t")
        by_feat.setdefault(int(f), []).append(float(v))
    return by_feat


def fp32_ulp(a, b):
    ai = int(np.array([a], dtype=np.float32).view(np.int32)[0])
    bi = int(np.array([b], dtype=np.float32).view(np.int32)[0])
    if ai < 0:
        ai = 0x80000000 - ai
    if bi < 0:
        bi = 0x80000000 - bi
    return abs(ai - bi)


def find_missing(m, c, ulp_tol=1):
    """Return the CPU borders not present in MLX (with greedy match)."""
    c_used = [False] * len(c)
    for v in m:
        for k, h in enumerate(c):
            if not c_used[k] and fp32_ulp(v, h) <= ulp_tol:
                c_used[k] = True
                break
    missing = [(k, c[k]) for k in range(len(c)) if not c_used[k]]
    return missing


def main():
    mlx = load("mlx_borders.tsv")
    cpu = load("cpu_borders.tsv")

    print("Per-feature: which CPU border is MLX missing?")
    print(f"{'feat':>4}  {'idx_in_cpu':>10}  {'value':>12}  {'cpu_below':>12}  {'cpu_above':>12}")
    for feat in sorted(mlx):
        c = cpu[feat]
        missing = find_missing(mlx[feat], c)
        for idx, val in missing:
            cb = c[idx-1] if idx > 0 else float("-inf")
            ca = c[idx+1] if idx + 1 < len(c) else float("inf")
            print(f"{feat:>4}  {idx:>10}  {val:>12.6g}  {cb:>12.6g}  {ca:>12.6g}")

    print()
    print("=== Feature 0 deep dive ===")
    print("L3 verdict: CPU split_index=3 (in stored borders), MLX bin=64 of 127 grid")
    print("L4 verdict: both correspond to border=0.014169 — to be verified now")
    print()
    f0_cpu = cpu[0]
    f0_mlx = mlx[0]
    print(f"CPU feature 0 stored-borders count = 128 (full grid)")
    print(f"MLX feature 0 borders count = 127")
    print()
    print("CPU borders[0..15]:")
    for i in range(16):
        print(f"  {i:>3}: {f0_cpu[i]:.9g}")
    print("...")
    print("CPU borders around bin=64 (in 128 grid):")
    for i in range(60, 70):
        print(f"  {i:>3}: {f0_cpu[i]:.9g}")
    print()
    print("MLX borders[0..15]:")
    for i in range(16):
        print(f"  {i:>3}: {f0_mlx[i]:.9g}")
    print("...")
    print("MLX borders around bin=64 (in 127 grid):")
    for i in range(60, 70):
        print(f"  {i:>3}: {f0_mlx[i]:.9g}")

    # The L4 claim: bin=64 in MLX 127-grid == 0.014169 == split at median(X[:,0]).
    # Let's compute median(X[:,0]) and find where each side has it.
    print()
    print("=== Median test ===")
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50_000, 20)).astype(np.float32)
    med = float(np.median(X[:, 0]))
    print(f"median(X[:,0]) = {med:.6f}")
    print(f"upper_bound in CPU borders -> bin index: {sum(1 for v in f0_cpu if v <= med)}")
    print(f"upper_bound in MLX borders -> bin index: {sum(1 for v in f0_mlx if v <= med)}")
    print()

    # And: where exactly do MLX bin=64 (the L3 finding) sit?
    print("=== L3 finding cross-check ===")
    print(f"MLX f0 borders[63..65]:")
    for i in [63, 64, 65]:
        if 0 <= i < len(f0_mlx):
            print(f"  borders[{i}] = {f0_mlx[i]:.9g}")
    # In MLX's binning, doc with feature in (borders[i-1], borders[i]] sits in bin i+1
    # (per upper_bound + binOffset). So bin=64 means feature value in (borders[63], borders[64]].


if __name__ == "__main__":
    main()
