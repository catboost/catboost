#!/usr/bin/env python3
"""
S33-L3-ITER2 reproducible diff script.

Loads both CPU and MLX binary/JSON dumps from data/ and re-runs the
per-stage diff. Use this to verify findings without re-running the
full experiment.

Usage:
  python docs/sprint33/l3-iter2/scripts/diff_l3.py [data_dir]

data_dir defaults to docs/sprint33/l3-iter2/data/

Stage definitions and noise thresholds:
  S1 GRADIENT  : max_rel > 1e-4 OR frac_diverging > 0.1%  => DIVERGENT
  S2 SPLIT     : feat != feat OR bin != bin                 => DIVERGENT
  S3 LEAF      : max_rel > 1e-4 OR frac_diverging > 0.1%  => DIVERGENT
  S4 APPROX    : max_rel > 1e-4 OR frac_diverging > 0.1%  => DIVERGENT
"""

import json
import sys
from pathlib import Path

import numpy as np

REL_DIFF_THRESHOLD  = 1e-4
FRAC_DIFF_THRESHOLD = 0.001   # 0.1%


def read_f32_bin(path: Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.float32)


def read_u32_bin(path: Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.uint32)


def read_json_array(path: Path) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list) and data and "leaf_value" in data[0]:
        return np.array([e["leaf_value"] for e in data], dtype=np.float32)
    return np.array(data, dtype=np.float32)


def read_json_obj(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def diff_arrays(label: str, cpu: np.ndarray, mlx: np.ndarray) -> dict:
    if cpu.shape != mlx.shape:
        print(f"  [{label}] SIZE MISMATCH cpu={cpu.shape} mlx={mlx.shape}")
        return {"label": label, "status": "SIZE_MISMATCH"}

    abs_diff = np.abs(cpu.astype(np.float64) - mlx.astype(np.float64))
    denom    = np.where(np.abs(cpu) < 1e-12, 1e-12, np.abs(cpu.astype(np.float64)))
    rel_diff = abs_diff / denom

    max_abs  = float(abs_diff.max())
    max_rel  = float(rel_diff.max())
    mean_abs = float(abs_diff.mean())
    frac_div = float((rel_diff > REL_DIFF_THRESHOLD).sum()) / len(cpu)

    is_noise = (max_rel <= REL_DIFF_THRESHOLD) and (frac_div <= FRAC_DIFF_THRESHOLD)
    status   = "CLEAN" if is_noise else "DIVERGENT"

    print(f"  [{label:<22}] max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
          f"frac>{REL_DIFF_THRESHOLD:.0e}={frac_div:.4%}  -> {status}")

    # Histogram of rel_diff for DIVERGENT cases
    if not is_noise:
        buckets = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, np.inf]
        counts, _ = np.histogram(rel_diff, bins=buckets)
        print(f"    rel_diff histogram:")
        for i, (lo, hi) in enumerate(zip(buckets[:-1], buckets[1:])):
            if counts[i] > 0:
                print(f"      [{lo:.0e}, {hi:.0e}): {counts[i]:>8d} elements "
                      f"({counts[i]/len(cpu):.2%})")

    return {
        "label": label, "status": status,
        "max_abs_diff": max_abs, "max_rel_diff": max_rel,
        "mean_abs_diff": mean_abs, "frac_diverging": frac_div,
        "n_elements": len(cpu),
    }


def diff_split(cpu_json: dict, mlx_json: dict) -> dict:
    cpu_feat = cpu_json.get("feat", -1)
    mlx_feat = mlx_json.get("feat", -1)
    cpu_bin  = cpu_json.get("bin", -1)
    mlx_bin  = mlx_json.get("bin", -1)
    cpu_gain = float(cpu_json.get("gain", -1))
    mlx_gain = float(mlx_json.get("gain", -1))

    feat_match = (cpu_feat == mlx_feat)
    bin_match  = (cpu_bin  == mlx_bin)
    gain_rel   = (abs(cpu_gain - mlx_gain) / max(abs(cpu_gain), 1e-12)
                  if cpu_gain > 0 else float("nan"))

    status = "CLEAN" if (feat_match and bin_match) else "DIVERGENT"
    print(f"  [S2-SPLIT               ] "
          f"CPU(feat={cpu_feat:>3},bin={cpu_bin:>3},gain={cpu_gain:>10.4f})  "
          f"MLX(feat={mlx_feat:>3},bin={mlx_bin:>3},gain={mlx_gain:>10.4f})  "
          f"feat_ok={feat_match}  bin_ok={bin_match}  gain_rel={gain_rel:.3e}  -> {status}")

    return {
        "label": "S2-SPLIT", "status": status,
        "cpu_feat": cpu_feat, "mlx_feat": mlx_feat, "feat_match": feat_match,
        "cpu_bin": cpu_bin,   "mlx_bin": mlx_bin,   "bin_match": bin_match,
        "cpu_gain": cpu_gain, "mlx_gain": mlx_gain, "gain_rel_diff": gain_rel,
    }


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "data"
    )
    print(f"diff_l3.py: loading from {data_dir}\n")

    # Check required files exist
    required = [
        "cpu_grad_iter2.bin",  "mlx_grad_iter2.bin",
        "cpu_hess_iter2.bin",  "mlx_hess_iter2.bin",
        "cpu_approx_iter2.bin","mlx_approx_iter2.bin",
        "mlx_bestsplit_d0_iter2.json",
        "cpu_bestsplit_d0_iter2.json",
        "mlx_leafvalues_iter2.json",
        "cpu_leafvalues_iter2.json",
    ]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        print(f"ERROR: missing files in {data_dir}:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    print("=== S1 GRADIENT ===")
    cpu_grad = read_f32_bin(data_dir / "cpu_grad_iter2.bin")
    mlx_grad = read_f32_bin(data_dir / "mlx_grad_iter2.bin")
    cpu_hess = read_f32_bin(data_dir / "cpu_hess_iter2.bin")
    mlx_hess = read_f32_bin(data_dir / "mlx_hess_iter2.bin")
    d_grad   = diff_arrays("S1-GRADIENT-g", cpu_grad, mlx_grad)
    d_hess   = diff_arrays("S1-GRADIENT-h", cpu_hess, mlx_hess)
    s1_status = "DIVERGENT" if any(d["status"] == "DIVERGENT" for d in [d_grad, d_hess]) else "CLEAN"
    print()

    print("=== S2 SPLIT ===")
    cpu_split = read_json_obj(data_dir / "cpu_bestsplit_d0_iter2.json")
    mlx_split = read_json_obj(data_dir / "mlx_bestsplit_d0_iter2.json")
    d_split   = diff_split(cpu_split, mlx_split)

    # Histogram diff (MLX side has actual per-bin histogram)
    if (data_dir / "mlx_hist_d0_iter2.bin").exists():
        mlx_hist = read_f32_bin(data_dir / "mlx_hist_d0_iter2.bin")
        print(f"  [S2-HIST] MLX hist: {len(mlx_hist)} float32 values  "
              f"mean={mlx_hist.mean():.6f}  absmax={np.abs(mlx_hist).max():.6f}")
        # CPU hist not available from Python API; note in output
        print(f"  [S2-HIST] CPU hist: not available from CatBoost Python API")
    print()

    print("=== S3 LEAF VALUES ===")
    cpu_leaf = read_json_array(data_dir / "cpu_leafvalues_iter2.json")
    mlx_leaf = read_json_array(data_dir / "mlx_leafvalues_iter2.json")
    d_leaf   = diff_arrays("S3-LEAF-VALUES", cpu_leaf, mlx_leaf)
    print()

    print("=== S4 APPROX ===")
    cpu_approx = read_f32_bin(data_dir / "cpu_approx_iter2.bin")
    mlx_approx = read_f32_bin(data_dir / "mlx_approx_iter2.bin")
    d_approx   = diff_arrays("S4-APPROX", cpu_approx, mlx_approx)
    print()

    # ---- Class call: first divergent stage ----
    all_diffs = [
        {"label": "S1-GRADIENT", "status": s1_status},
        d_split,
        d_leaf,
        d_approx,
    ]

    print("=== CLASS CALL ===")
    first_div = next((d["label"] for d in all_diffs if d.get("status") == "DIVERGENT"), "CLEAN")
    print(f"  First divergent stage: {first_div}")

    print()
    print("=== SUMMARY TABLE ===")
    print(f"  {'stage':<22} {'status':<12} {'max_rel_diff':<16} {'frac_diverging'}")
    print("  " + "-" * 70)
    for d in all_diffs:
        mr = d.get("max_rel_diff", d.get("gain_rel_diff", float("nan")))
        fd = d.get("frac_diverging", float("nan"))
        print(f"  {d['label']:<22} {d['status']:<12} {mr:<16.4e} {fd:.4%}")

    return first_div


if __name__ == "__main__":
    r = main()
    print(f"\nResult: {r}")
