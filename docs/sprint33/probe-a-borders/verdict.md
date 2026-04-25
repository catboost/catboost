# PROBE-A — Verdict: FALSIFIED

**Date**: 2026-04-24
**Branch**: `mlx/sprint-33-iter2-scaffold`
**Branch tip**: `9dfd62ccc3` (probe runs against this commit; no code modified)
**Kernel md5** (`catboost/mlx/kernels/kernel_sources.h`): `9edaef45b99b9db3e2717da93800e76f` (verified intact pre/post)
**Investigation type**: claim-verification (no implementation)
**Class call**: **FALSIFIED**

---

## Anchor

| Setting | Value |
|---|---|
| N | 50000 |
| grow_policy | SymmetricTree |
| score_function | Cosine |
| loss_function | RMSE |
| depth | 6 |
| border_count | 128 |
| iterations | 50 |
| dataset RNG | `np.random.default_rng(42)` (matches `run_phase1.py`) |
| `bootstrap_type` | `'No'` |
| `random_strength` | `0.0` |
| `has_time` | `True` |
| CatBoost `random_seed` | `0` (verified `42` gives identical result) |
| catboost version | 1.2.10 |

**Dataset**: `X[:,0..19] ~ N(0,1)`, `y = 0.5·X[:,0] + 0.3·X[:,1] + 0.1·N(0,1)`.
Features 2–19 are pure noise, identical generation as `docs/sprint33/l4-fix/run_phase1.py`.

**Saved model SHA-256**: `9a9281cb986c572773e789bdbc96685bb2fb3d3a8c7d660daf5e05d24202677c` (`cpu_anchor.cbm`).

---

## L4 claim under test

`docs/sprint33/l4-fix/verdict.md` lines 90–122 ("Phase 2: True Root Cause"):

> CatBoost's quantization is **dynamic and target-aware in the aggregate** […]
> features 2–19 are pure N(0,1) noise […] They are never chosen, so they get 0 borders.
>
> ```
> CatBoost CPU at 50 iterations:
>   Feature 0: 95 borders    Feature 1: 71 borders
>   Features 2-19: 0 borders each
>
> Effective bin-feature search space:
>   CatBoost: 95 + 71 = 166 bin-features (useful signal only)
>   csv_train.cpp: 20 × 127 = 2540 bin-features (mostly noise)
> ```

The claim has two distinct factual components:

1. **Mechanism**: CatBoost accumulates borders dynamically as splits are chosen.
2. **Numerics**: At the L4 anchor, features 2–19 each end up with 0 borders.

Both must hold for the L4 closure narrative ("Option 3 — accept the divergence")
to stand.

---

## Methodology

There are **two distinct "border" concepts** that the L4 verdict conflates:

| Concept | When it's set | What controls it |
|---|---|---|
| (a) Upfront quantization grid | Training prep, before any tree | `feature_border_type` × `border_count` |
| (b) Stored-in-model borders | Model save (CBM/JSON) | Pruning of unused thresholds for storage |

CatBoost's saved model files only retain borders that the trained trees
reference — this is a model-storage compression, not a training-time choice.
`model.get_borders()` returns view (b), not view (a). The static-vs-dynamic
question is about (a). To resolve it, the probe extracts both views via four
independent paths.

| Method | Returns view | Result |
|---|---|---|
| (i) `model.get_borders()` on live trained model | (b) stored | OK |
| (ii) JSON `features_info.float_features[*].borders` count | (b) stored | OK |
| (iii) Reload CBM, then `get_borders()` | (b) stored | OK |
| (iv) `Pool.quantize(border_count=128)` + `save_quantization_borders()` *(no training)* | (a) upfront | OK |

All four methods ran successfully. Methods (i)/(ii)/(iii) agree on a per-feature
basis (sum 207). Method (iv) gives the upfront grid (sum 2560 = 20 × 128).

---

## Comparison table

| feature | L4 claim | upfront — method (iv) | stored — methods (i,ii,iii) | used in 50 trees |
|---|---:|---:|---:|---:|
| 0 | 95 | **128** | 73 | 73 |
| 1 | 71 | **128** | 59 | 59 |
| 2 | 0 | **128** | **4** | 4 |
| 3 | 0 | **128** | **7** | 7 |
| 4 | 0 | **128** | **6** | 6 |
| 5 | 0 | **128** | **5** | 5 |
| 6 | 0 | **128** | **3** | 3 |
| 7 | 0 | **128** | **4** | 4 |
| 8 | 0 | **128** | **5** | 5 |
| 9 | 0 | **128** | **4** | 4 |
| 10 | 0 | **128** | **7** | 7 |
| 11 | 0 | **128** | **3** | 3 |
| 12 | 0 | **128** | **3** | 3 |
| 13 | 0 | **128** | **2** | 2 |
| 14 | 0 | **128** | **6** | 6 |
| 15 | 0 | **128** | **5** | 5 |
| 16 | 0 | **128** | **1** | 1 |
| 17 | 0 | **128** | **4** | 4 |
| 18 | 0 | **128** | **2** | 2 |
| 19 | 0 | **128** | **4** | 4 |
| **total** | **166** | **2560** | **207** | **207** |

The only quantity that is identical to "used thresholds" is "stored-in-model
borders" — they are the same number per feature, exactly. This is consistent
with CatBoost's CBM save format pruning borders unreferenced by any tree.

The L4 claim of "0 borders for features 2–19" matches *neither* view of the data:
- Upfront grid (iv): all 20 features have **128** borders, not 0.
- Stored/used: every noise feature has **1 to 7** borders, never 0.
- The L4 "166 candidate bin-features" total contradicts both the actual
  upfront candidate count (2560) and the actual used-threshold count (207).

The L4 numbers (95 for feat 0, 71 for feat 1) are also off from this run's
stored counts (73, 59), but the discrepancy is small and likely sensitive to
the exact dataset RNG path or CatBoost version. The structural finding —
"features 2–19 have 0" — is wrong by both readings, regardless of the
small numeric drift in features 0/1.

---

## Class call: FALSIFIED

The agent's claimed mechanism is wrong. The evidence is unambiguous:

1. **CatBoost quantizes upfront, not dynamically.** Method (iv) ran the
   identical quantization protocol with **no training at all** and produced
   128 borders for every feature. Borders are computed from the value
   distribution at training-prep time, controlled by `border_count` and
   `feature_border_type` (`GreedyLogSum` here, confirmed in
   `data_processing_options.float_features_binarization`). Whether a feature
   later gets chosen for a split is irrelevant to whether the border exists at
   training time — they are made and held in memory, then used as candidates
   on every split-selection pass.

2. **The 95/71/0 numbers are stored-in-model borders, not training-time
   candidates.** The fact that stored borders == used thresholds (exact
   per-feature equality, sum 207) confirms this. Saved-model border lists are
   compressed by removing borders no tree references; this is a serialization
   detail, not the training algorithm.

3. **Features 2–19 have ~128 *available* borders during training, not 0.**
   Both CPU CatBoost (this probe) and `csv_train.cpp` evaluate full
   per-feature border grids on every split-selection pass. The L4
   "166 vs 2540 candidate bin-features" comparison is built on a
   non-existent asymmetry.

---

## Implication for L4 closure

**The L4 closure mechanism does not stand.** The claim that "csv_train.cpp's
static 127-border grid wastes tree depth on noise that CatBoost CPU does not
even evaluate" is built on a misreading of the saved-model border list as the
training-time candidate set. CatBoost CPU evaluates the same dense
per-feature grid that csv_train.cpp evaluates; both train against the same
upfront 128-border-per-feature candidate space.

DEC-036 (ST+Cosine 52.6% iter≥2 RMSE drift) therefore remains genuinely
**OPEN**. The iter≥2 mechanism that the L0→L1→L2→L3 chain narrowed to
"per-iter persistent divergence at S2 histogram/split selection" is **not yet
identified**. The L4 verdict's "Phase 2: True Root Cause" must be retracted;
the L3 hypothesis being falsified does not entail that the L4 hypothesis is
correct.

DEC-041 ("port dynamic border accumulation") is built on the same
misdiagnosis — there is no dynamic border accumulation in CatBoost CPU to
port. It should not drive S34 scope as currently written.

This probe makes no claim about what the iter≥2 divergence mechanism actually
is. Possible candidates that remain unexplored:
- A genuine difference in the *score* computed per (feature, border) pair
  (Cosine numerator/denominator floating-point path differences not yet
  ruled out at iter≥2 — DEC-035 closed precision class but only at iter=1).
- An asymmetry in how csv_train.cpp constructs per-feature border *values*
  (GreedyLogSum on a different sample? from a different RNG path?) versus
  the model that L4 ran against.
- A bug in csv_train.cpp's split-search dispatch at iter≥2 only.
- An off-by-one or coordinate-system issue between csv_train.cpp's bin
  indexing and CatBoost's, surviving the depth=0 cancellation that S31/S32
  cleared at iter=1.

None of these are claims by this probe — they are open questions that the
L4 verdict's premature closure was hiding.

---

## Sanity sub-check

`features_info.float_features[*].feature_border_type` is missing from the JSON
entries (CatBoost omits it when default). The dataset processing block carries
the truth:

```
float_features_binarization = {
  'border_count': 128,
  'border_type': 'GreedyLogSum',
  'dev_max_subset_size_for_build_borders': 200000,
  'nan_mode': 'Min',
}
```

`border_count=128` and `border_type=GreedyLogSum` confirm standard upfront
quantization with the expected algorithm. No noise feature is using a
non-default `feature_border_type`. No `per_float_feature_quantization` overrides
are present.

---

## Raw artifacts

- `data/cpu_anchor.cbm` — saved trained model (sha256 above).
- `data/cpu_anchor.json` — same model, JSON format, used for parsing tree
  splits and float-features blocks.
- `data/available_borders.csv` — per-feature counts under all four methods,
  exhibiting the (b)-vs-(a) discrepancy.
- `data/used_thresholds.csv` — per-feature unique split thresholds across the
  50 oblivious trees.
- `data/upfront_quantization_borders.tsv` — full 2560-row upfront grid from
  `Pool.quantize(border_count=128)`, exported via `save_quantization_borders`.
- `data/script.py` — the Python script used to produce all of the above.

---

## Caveats

1. The probe used CatBoost 1.2.10. Version drift in CBM serialization could
   shift the *numeric* values on features 0/1 (the probe got 73/59; L4
   reported 95/71). The *structural* finding — "features 2–19 do not have 0
   borders, and the upfront grid is dense for all features" — does not depend
   on these numbers and was verified at two `random_seed` settings (0 and 42)
   with identical structural result.
2. The probe does not investigate *why* csv_train.cpp diverges from CatBoost
   CPU at iter≥2 by 52.6% RMSE. It only refutes the L4 explanation. The
   mechanism remains to be identified.
3. The probe does not examine the actual *border values* — only counts. If
   csv_train.cpp's per-feature borders differ in value from CatBoost CPU's
   (e.g., one rounds float32 differently, or uses a different sample to
   compute GreedyLogSum), that would be a separate, also-open question.

