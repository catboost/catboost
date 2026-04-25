# S33-L0-CONFIG Verdict

**Date:** 2026-04-24
**Branch:** `mlx/sprint-33-iter2-scaffold`
**Tip at audit:** `1fcb6ad981`
**Task:** #119 S33-L0-CONFIG
**Anchor:** N=50000, grow_policy=SymmetricTree, score_function=Cosine, loss=RMSE, depth=6, bins=128 (DEC-039 cap: 127 borders), iter=50, seeds={42,43,44}, random_seed=0 (random_strength=0)

---

## Method

CPU effective config extracted from `CatBoostRegressor` model JSON (`model_info.params`) after fitting at the anchor config. MLX effective config read directly from `TConfig` struct defaults in `catboost/mlx/tests/csv_train.cpp` plus the CLI flags used in all prior audit harnesses (run_g3a.py and equivalents).

Source files:
- CPU params: `docs/sprint33/l0-config/data/cpu_params.json`
- MLX params: `docs/sprint33/l0-config/data/mlx_params.txt`
- kernel_sources.h md5: `9edaef45b99b9db3e2717da93800e76f` — VERIFIED (v5, byte-identical)

---

## Field-by-Field Diff Table

| Field | CPU value | MLX value | Class | Note |
|-------|-----------|-----------|-------|------|
| `boosting_type` | `Plain` | N/A (Plain-equivalent) | SOFT-DIFF | MLX has no Ordered path; all training is functionally Plain. Not a behavioral difference at this anchor. |
| `bootstrap_type` | `No` | `"no"` | NO-DIFF | String case differs but both mean no bootstrap. |
| `bagging_temperature` | 1.0 (unused) | 1.0 (unused) | NO-DIFF | Both bootstrap=No; parameter inactive on both sides. |
| `subsample` (rsm) | 1.0 | 1.0 | NO-DIFF | No feature subsampling on either side. |
| `leaf_estimation_method` | `Newton` | Newton (single step: −lr·gSum/(hSum+l2)) | NO-DIFF | MLX implements the Newton formula exactly. |
| `leaf_estimation_iterations` | 1 | 1 (single step only) | NO-DIFF | CPU=1 iteration, MLX=1 iteration; single Newton step on both sides. |
| `leaf_estimation_backtracking` | `AnyImprovement` | None (not implemented) | SOFT-DIFF | With `leaf_estimation_iterations=1`, CPU's `CreateBacktrackingObjective` sets `haveBacktrackingObjective = (iterations > 1 && ...)` = `false`. Backtracking is structurally disabled. MLX not implementing it is non-functional at this anchor. Source: `approx_calcer_helpers.cpp:14`. |
| `langevin` | false / not set | N/A | NO-DIFF | Not active on either side at this anchor. |
| `diffusion_temperature` | N/A | N/A | NO-DIFF | Langevin=false; irrelevant on both sides. |
| `model_shrink_rate` | 0 | N/A | SOFT-DIFF | CPU=0 means no model shrinkage applied. MLX does not implement shrinkage. Functionally equivalent when CPU value=0. |
| `model_shrink_mode` | `Constant` | N/A | SOFT-DIFF | Irrelevant when model_shrink_rate=0. |
| `score_function` | `Cosine` | `"Cosine"` | NO-DIFF | Identical. |
| `has_time` | `false` | N/A | SOFT-DIFF | CPU=false means no temporal ordering of data. MLX processes data in file order (equivalent to has_time=false). |
| `ordered` (boosting_type=Ordered) | `Plain` (not Ordered) | N/A | SOFT-DIFF | Both are Plain. No ordered boosting on either side. |
| `sampling_unit` | `PerTree` (sampling_frequency) | N/A | SOFT-DIFF | SubsampleRatio=1.0 on MLX, no row subsampling; sampling_unit is irrelevant. |
| `random_strength` | 0.0 | 0.0 | NO-DIFF | No score perturbation on either side. |
| `fold_permutation_block` | 0 (auto) | N/A | SOFT-DIFF | Fold permutation only applies to Ordered boosting. CPU=Plain, MLX=Plain-equivalent. |
| `approx_on_full_history` | `false` | N/A | SOFT-DIFF | CPU source: `CB_ENSURE(!(approx_on_full_history && boosting_type == Plain))` — Plain+full_history is an invalid combination; CPU enforces false. MLX equivalent. |
| `l2_leaf_reg` | 3.0 | 3.0 | NO-DIFF | Identical. MLX scales by sumAllWeights/docCount = 1.0 at uniform weights. |
| `learning_rate` | 0.03 | 0.03 | NO-DIFF | Identical. |
| `depth` | 6 | 6 | NO-DIFF | Identical. |
| `border_count` | 127 | 127 | NO-DIFF | DEC-039 cap applied on MLX side: std::min(128, 127u)=127. CPU border_count=127 set explicitly. |
| `feature_border_type` | `GreedyLogSum` | `GreedyLogSum` | NO-DIFF | DEC-037: MLX restored greedy unweighted algorithm. |
| `gpu_cat_features_storage` | N/A (CPU task) | N/A | NO-DIFF | CPU task_type; field irrelevant on both sides. |
| `fold_len_multiplier` | 2 | N/A | SOFT-DIFF | Only used in `IsOrderedBoosting=true` branch (learn_context.cpp:496). CPU=Plain → branch never entered. |
| `min_data_in_leaf` | 1 | 1 | NO-DIFF | Identical. |
| `max_leaves` | 64 | 31 (default, unused) | SOFT-DIFF | `max_leaves` is only consumed by Lossguide policy. GrowPolicy=SymmetricTree on both sides; parameter is structurally unused. CPU resolves to 64; MLX default is 31. Neither value is active. |
| `boost_from_average` | `true` | `true` (always) | NO-DIFF | MLX always calls CalcBasePrediction for RMSE; computes weighted mean of targets = mean target. CPU model JSON shows boost_from_average=true. Functionally identical. |
| `nan_mode` | `Min` | `"min"` | NO-DIFF | NaN → bin 0 on both sides. |
| `rsm` (feature subsample) | 1.0 | 1.0 | NO-DIFF | ColsampleByTree=1.0 on MLX. |

---

## SOFT-DIFF Justifications

All SOFT-DIFFs are structurally non-functional at this specific anchor configuration:

1. **boosting_type** — MLX never implements Ordered; CPU is in Plain mode. Same effective algorithm.
2. **leaf_estimation_backtracking=AnyImprovement** — Disabled at runtime by `leaf_estimation_iterations=1`. CPU source confirms: `haveBacktrackingObjective = (1 > 1 && ...)` = false.
3. **model_shrink_rate/mode** — CPU rate=0 = no-op; MLX omits the feature. Numerically identical.
4. **has_time=false** — Both process data in input order with no temporal constraint.
5. **ordered / approx_on_full_history** — Plain boosting on both sides; Ordered path unreachable.
6. **sampling_unit** — SubsampleRatio=1.0 in MLX; no subsampling happens, so unit is irrelevant.
7. **fold_permutation_block** — Fold permutation only applies to Ordered boosting; not active.
8. **fold_len_multiplier=2** — Used only in IsOrderedBoosting=true branch. Plain boosting → dead code.
9. **max_leaves** — GrowPolicy=SymmetricTree ignores max_leaves on both sides.

---

## Overall Class Call

**NO-DIFF**

No HARD-DIFF identified. All differences are either string-casing of equivalent values, or parameters that are structurally disabled at this anchor config (Plain boosting type with bootstrap=No, rs=0, single leaf-estimation iteration, SymmetricTree grow policy).

---

## Decision

**L0-PASS.** Frame C-config is falsified for this anchor. No config field is causally active that differs between CPU and MLX sides. The 52.6% iter=50 drift cannot be attributed to config mismatch.

Proceed to **#120 S33-L1-DETERMINISM**.

---

## Notes

- The MLX binary has no formal config dump mechanism. The effective config is derived by reading `TConfig` struct defaults plus the CLI flags used by the audit harnesses. An `CATBOOST_MLX_DUMP_PARAMS=1` guard was not added; the struct is small enough that source-code reading is authoritative.
- The one parameter absent from the mandatory audit list (`random_seed=0` in the task spec) was `random_seed`. Clarification: the anchor uses `--seed 42` (the data-generation and training seed), and `random_strength=0` (noise disabled). The spec's `random_seed=0` refers to the global random_strength=0 setting, not a seed value. Both sides use seed=42 for data and training; rs=0 on both sides. NO-DIFF.
- kernel_sources.h md5 confirmed `9edaef45b99b9db3e2717da93800e76f` before and after this audit (no code changes). DEC-012 atomicity preserved: this commit is docs-only.
