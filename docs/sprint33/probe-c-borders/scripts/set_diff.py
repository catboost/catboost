#!/usr/bin/env python3
"""
Set-overlap analysis between MLX and CPU borders for the anchor.

Hypothesis to test: MLX's 127 borders per feature are a (near-)subset of
CPU's 128 borders per feature. If true, the iter≥2 mechanism narrows to
"MLX has 1 fewer border per feature, and that absent border is a real one
CPU could split on."

If false (the value sets diverge globally), the mechanism is in the
GreedyLogSum scoring path itself.
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


def find_in(needle, haystack, ulp_tol=1):
    """Return index of haystack value within ulp_tol of needle, or -1."""
    for j, h in enumerate(haystack):
        if fp32_ulp(needle, h) <= ulp_tol:
            return j
    return -1


def main():
    mlx = load("mlx_borders.tsv")
    cpu = load("cpu_borders.tsv")

    print("=" * 72)
    print("Set overlap analysis (ULP tolerance = 1)")
    print("=" * 72)
    print()
    print(f"{'feat':>4} {'mlx':>4} {'cpu':>4} {'mlx∩cpu':>8} {'mlx-only':>8} {'cpu-only':>8}")
    total_mlx = 0
    total_cpu = 0
    total_overlap = 0
    total_mlx_only = 0
    total_cpu_only = 0
    feat_with_unique_mlx = []
    for feat in sorted(mlx):
        m = mlx[feat]
        c = cpu[feat]
        c_used = [False] * len(c)
        overlap = 0
        mlx_only = []
        for v in m:
            j = -1
            for k, h in enumerate(c):
                if not c_used[k] and fp32_ulp(v, h) <= 1:
                    j = k
                    break
            if j >= 0:
                c_used[j] = True
                overlap += 1
            else:
                mlx_only.append(v)
        cpu_only = [c[k] for k in range(len(c)) if not c_used[k]]

        total_mlx += len(m)
        total_cpu += len(c)
        total_overlap += overlap
        total_mlx_only += len(mlx_only)
        total_cpu_only += len(cpu_only)
        marker = "  ← MLX has unique" if mlx_only else ""
        print(f"{feat:>4} {len(m):>4} {len(c):>4} {overlap:>8} {len(mlx_only):>8} {len(cpu_only):>8}{marker}")
        if mlx_only:
            feat_with_unique_mlx.append((feat, mlx_only, cpu_only))

    print()
    print(f"TOTALS: mlx={total_mlx}  cpu={total_cpu}  overlap={total_overlap}")
    print(f"  mlx-only (in MLX but not CPU): {total_mlx_only}")
    print(f"  cpu-only (in CPU but not MLX): {total_cpu_only}")
    print()
    if total_mlx_only == 0:
        print("VERDICT: MLX is a STRICT SUBSET of CPU. Each MLX border is also in CPU's grid.")
        print(f"  MLX is missing {total_cpu_only} = {total_cpu - total_mlx} borders per feature on average.")
        print()
        print("  Mechanism class: \"missing borders\". MLX's GreedyLogSum truncates 1 border")
        print("  earlier than CPU's. The missing border is real (CPU has it) and could win a split")
        print("  on iter≥2 — or the bin assignment for some docs differs (the missing border")
        print("  carries docs that, in CPU, sit in their own bin, but in MLX get folded into")
        print("  the adjacent bin).")
    else:
        print("VERDICT: NOT a subset. MLX has borders CPU does NOT have:")
        for feat, mo, co in feat_with_unique_mlx[:5]:
            print(f"  feat={feat}: mlx-only={mo[:5]}{'...' if len(mo) > 5 else ''}")
            print(f"             cpu-only={co[:5]}{'...' if len(co) > 5 else ''}")
        print()
        print("  Mechanism class: \"divergent grid construction\". GreedyLogSum scoring/order")
        print("  differs between MLX's port and CatBoost's reference. This is independent of the")
        print("  127 vs 128 cap — the borders themselves diverge.")


if __name__ == "__main__":
    main()
