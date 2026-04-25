#!/usr/bin/env python3
"""PROBE-A — verify L4 claim that CatBoost CPU accumulates borders dynamically.

L4 verdict (`docs/sprint33/l4-fix/verdict.md` lines 90-122) claims:
  - feature 0: 95 borders
  - feature 1: 71 borders
  - features 2-19: 0 borders each
  - total candidates: 166

This claim conflates AVAILABLE borders (set at quantization time, depends only
on `feature_border_type` + `border_count` and the value distribution) with USED
thresholds (split points the trained model actually references).

Standard CatBoost quantizes UPFRONT, before the first tree. Every numeric
feature gets up to `border_count` borders regardless of whether splits later
choose it. This probe extracts both quantities via three independent paths and
compares them.

Anchor (must match L4 exactly): N=50000, ST grow_policy, Cosine score, RMSE
loss, depth=6, bins=128, iter=50, seed=42, random_seed=0, L1 determinism stack
(bootstrap_type='No', random_strength=0.0, has_time=True). Dataset: same RNG
protocol as `run_phase1.py` (np.random.default_rng(42), X[:,0..19] ~ N(0,1),
y = 0.5*X[:,0] + 0.3*X[:,1] + 0.1*N(0,1)).
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor, Pool

DATA_DIR = Path(__file__).resolve().parent

# ----- dataset (matches run_phase1.py exactly) -----
N = 50_000
SEED = 42
rng = np.random.default_rng(SEED)
X = rng.standard_normal((N, 20)).astype(np.float32)
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(N) * 0.1).astype(np.float32)

print(f"[data] X: {X.shape} dtype={X.dtype}")
print(f"[data] y: {y.shape} dtype={y.dtype} mean={y.mean():.6f} std={y.std():.6f}")

# ===================================================================
# Step 1 — Train CatBoost CPU at the L4 anchor
# ===================================================================
model = CatBoostRegressor(
    iterations=50,
    depth=6,
    border_count=128,
    grow_policy='SymmetricTree',
    score_function='Cosine',
    loss_function='RMSE',
    bootstrap_type='No',
    random_strength=0.0,
    has_time=True,
    random_seed=0,
    verbose=0,
)
model.fit(X, y)

cbm_path = DATA_DIR / "cpu_anchor.cbm"
json_path = DATA_DIR / "cpu_anchor.json"
model.save_model(str(cbm_path))
model.save_model(str(json_path), format='json')

cbm_sha = hashlib.sha256(cbm_path.read_bytes()).hexdigest()
print(f"[model] saved cbm sha256={cbm_sha}")
print(f"[model] saved json={json_path}")

# ===================================================================
# Step 2 — AVAILABLE borders per feature
# ===================================================================
# IMPORTANT: there are TWO distinct "border" concepts.
#
#   (a) UPFRONT QUANTIZATION GRID: the borders CatBoost computes at training
#       prep time from the feature value distribution, before any tree is
#       built. Controlled by `feature_border_type` × `border_count`. This
#       is what csv_train.cpp's static grid is supposed to mirror.
#
#   (b) STORED-IN-MODEL BORDERS: what the saved CBM/JSON model file
#       contains. CatBoost prunes unused borders from the saved model as a
#       storage optimization. `model.get_borders()` returns this pruned
#       set, NOT the upfront grid.
#
# The L4 claim "feature N: K borders" appears to be reading (b) and
# treating it as (a). To resolve definitively, we extract both:
#
#   Method (i)   model.get_borders()                  -> view (b)
#   Method (ii)  JSON features_info.float_features    -> view (b)
#   Method (iii) reloaded CBM .get_borders()          -> view (b)
#   Method (iv)  Pool.quantize(...) + save_quantization_borders -> view (a)
#
# The resolution between (a) and (b) is the discriminator.
print("\n=== Step 2: AVAILABLE borders per feature ===")

# Method (i) — Python API on the live model
method_i = {}
try:
    borders_i = model.get_borders()
    for f in range(20):
        method_i[f] = len(borders_i.get(f, []))
    print(f"[method i  — model.get_borders()] OK ({sum(method_i.values())} total)")
except Exception as e:
    method_i = None
    print(f"[method i  — model.get_borders()] FAILED: {e!r}")

# Method (ii) — saved JSON model
method_ii = {}
try:
    with open(json_path) as f:
        jm = json.load(f)
    fi_block = jm.get('features_info', {})
    ff = fi_block.get('float_features', [])
    for entry in ff:
        idx = entry.get('feature_index')
        bs = entry.get('borders', [])
        method_ii[idx] = len(bs)
    for f in range(20):
        if f not in method_ii:
            method_ii[f] = 0
    print(f"[method ii — JSON features_info]   OK ({sum(method_ii.values())} total)")
except Exception as e:
    method_ii = None
    print(f"[method ii — JSON features_info]   FAILED: {e!r}")

# Method (iii) — reload CBM and call get_borders
method_iii = {}
try:
    m2 = CatBoostRegressor()
    m2.load_model(str(cbm_path))
    borders_iii = m2.get_borders()
    for f in range(20):
        method_iii[f] = len(borders_iii.get(f, []))
    print(f"[method iii— reloaded CBM]         OK ({sum(method_iii.values())} total)")
except Exception as e:
    method_iii = None
    print(f"[method iii— reloaded CBM]         FAILED: {e!r}")

# Method (iv) — UPFRONT QUANTIZATION GRID via Pool.quantize() (no training)
method_iv = {}
quant_tsv = DATA_DIR / "upfront_quantization_borders.tsv"
try:
    pool = Pool(X, y)
    pool.quantize(border_count=128)
    pool.save_quantization_borders(str(quant_tsv))
    counts = {}
    with open(quant_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    fi = int(parts[0])
                    counts[fi] = counts.get(fi, 0) + 1
                except ValueError:
                    pass
    for f in range(20):
        method_iv[f] = counts.get(f, 0)
    print(f"[method iv — Pool.quantize upfront] OK ({sum(method_iv.values())} total)")
except Exception as e:
    method_iv = None
    print(f"[method iv — Pool.quantize upfront] FAILED: {e!r}")

# Save CSV — keep both views so the discriminator is preserved
with open(DATA_DIR / "available_borders.csv", "w", newline='') as f:
    w = csv.writer(f)
    w.writerow([
        "feature_idx",
        "stored_in_model_method_i",
        "stored_in_model_method_ii_json",
        "stored_in_model_method_iii_reload",
        "upfront_quantization_method_iv",
    ])
    for fi in range(20):
        w.writerow([
            fi,
            method_i[fi] if method_i is not None else "FAILED",
            method_ii[fi] if method_ii is not None else "FAILED",
            method_iii[fi] if method_iii is not None else "FAILED",
            method_iv[fi] if method_iv is not None else "FAILED",
        ])
print("[available] saved -> available_borders.csv")
print()
print(f"  {'feat':>4} | {'stored':>6} | {'upfront':>7}")
print(f"  {'----':>4} | {'------':>6} | {'-------':>7}")
for fi in range(20):
    s = method_i[fi] if method_i is not None else -1
    u = method_iv[fi] if method_iv is not None else -1
    print(f"  {fi:>4} | {s:>6} | {u:>7}")

# ===================================================================
# Step 3 — USED thresholds per feature (parse 50 oblivious trees)
# ===================================================================
print("\n=== Step 3: USED thresholds per feature ===")

used = {f: set() for f in range(20)}
otrees = jm.get('oblivious_trees', [])
print(f"[trees] {len(otrees)} oblivious trees in JSON model")
total_splits = 0
for tree in otrees:
    for sp in tree.get('splits', []):
        total_splits += 1
        if 'float_feature_index' in sp:
            fi = sp['float_feature_index']
            br = sp.get('border', None)
            if br is not None:
                used[fi].add(round(float(br), 8))
print(f"[trees] total splits across all trees = {total_splits}")
print(f"[trees] expected = 50 trees × 6 levels = 300")

with open(DATA_DIR / "used_thresholds.csv", "w", newline='') as f:
    w = csv.writer(f)
    w.writerow(["feature_idx", "num_unique_used_thresholds"])
    for fi in range(20):
        w.writerow([fi, len(used[fi])])
print("[used] saved -> used_thresholds.csv")
for fi in range(20):
    print(f"  feat {fi:2d}: {len(used[fi])} unique thresholds used")

# ===================================================================
# Step 5 — sanity sub-check: feature_border_type and border_count in params
# ===================================================================
print("\n=== Step 5: sanity sub-check ===")
print(f"[sanity] features_info.float_features count: {len(ff)}")
for entry in ff[:3]:
    print(f"  feat {entry.get('feature_index')}: "
          f"feature_border_type={entry.get('feature_border_type', 'MISSING')}, "
          f"nan_value_treatment={entry.get('nan_value_treatment', 'MISSING')}, "
          f"has_nans={entry.get('has_nans', 'MISSING')}, "
          f"borders_n={len(entry.get('borders', []))}")

params_block = jm.get('model_info', {}).get('params', None)
if isinstance(params_block, str):
    try:
        params_block = json.loads(params_block)
    except Exception:
        pass
if isinstance(params_block, dict):
    dpo = params_block.get('data_processing_options', {})
    print(f"[sanity] data_processing_options keys: {list(dpo.keys())}")
    for key in ('border_count', 'feature_border_type'):
        if key in dpo:
            print(f"  data_processing_options[{key}] = {dpo[key]}")
    fpo = dpo.get('float_features_binarization', None)
    if isinstance(fpo, dict):
        print(f"[sanity] float_features_binarization = {fpo}")
    pfqo = dpo.get('per_float_feature_quantization', None)
    if pfqo:
        print(f"[sanity] per_float_feature_quantization = {pfqo}")

print("\n[probe] done.")
