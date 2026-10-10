# Greedy PairLogit

Depthwise, Lossguide and Region now connect PairLogit to native and standalone
Metal training. The four reported Armijo/permutation discrepancies are resolved,
and focused acceptance and preserved-snapshot recovery pass on M3 Pro.
**Release acceptance is complete.** Checkpoint `20260913T195124Z` is installed;
the full, alternate, CLI, recovery and installed-package checks pass.

## Source mapping and behavior

- CUDA's `cuda/train_lib/querywise_non_symmetric.cpp` and
  `querywise_region.cpp` register PairLogit for the three policies.
  `cuda/targets/querywise_targets_impl.h::InitPairLogit` constructs original
  incident pair mass; `cuda/targets/kernel/pair_logit.cu` defines edge
  derivatives. `cuda/train_lib/train_template.h::NeedZeroAverage` requests
  unweighted zero-average leaf finalization.
- [metal_greedy_trainer.mm](native/metal_greedy_trainer.mm) connects those
  semantics to structure search, iterative leaf estimation, backtracking and
  centering through [metal_pairwise_runtime.h](native/metal_pairwise_runtime.h)
  and [metal_pairwise_kernels.h](native/metal_pairwise_kernels.h).
  Supplied edges retain their literal weights, duplicates and zero weights.
  Extra object/group weights do not multiply them; each tree's original leaf
  mass sums to twice the supplied edge mass. Bootstrap affects structure search.
- The counted `cbm_greedy_session_create_pair` constructor checks pair/offset
  counts before pointer reads. [train.cpp](train_lib/train.cpp) and
  [greedy_session.h](train_lib/greedy_session.h) connect prepared native Pool
  pairs. [ranker.py](python/catboost_metal/ranker.py),
  [_greedy.py](python/catboost_metal/_greedy.py) and
  [_greedy_training.py](python/catboost_metal/_greedy_training.py) connect the
  standalone API and snapshot lifecycle.
- Each permutation retains its own bins, routing, prediction cursor and leaf
  estimates. The selected dataset supplies structure; the final dataset supplies
  exported leaves. Training and validation edges/weights participate in snapshot
  identity. Recovery restores the complete untrimmed state, including permutation
  cursors, even when the returned model is trimmed to its best iteration.

## Resolved failures

The initial private suite passed 265 cases and failed four cursor comparisons:
Bayesian P7 Region, Bernoulli P4/P7 Depthwise and Bernoulli P7 Region. Late Armijo
attempts subtracted separately rounded absolute float32 losses. Near convergence,
the improvement was smaller than their representable spacing, changing an
accept/reject decision and later predictions.

`ReducePairwiseObjectiveDifference` now computes signed edge loss changes before
reduction. For small margin updates it uses a stable `log1p`/`expm1` identity;
float expansions preserve margin differences and compensated reduction preserves
small signed totals. Point additions retain the training cursor's float32
semantics. Nonfinite trials are rejected, and zero-weight edges remain inert.
The diagnostic maximum cursor difference from independent equations fell from
approximately `1.386e-4` to `1.1920928955078125e-7`. Existing failing-case
tolerances were not widened. Exact snapshot replay is a separate requirement.

Native unlabeled Pools exposed a second issue: shared GPU preprocessing creates
placeholder zero targets, and `libs/metrics/metric.cpp::CheckPreprocessedTarget`
exempted PairLogit losses but omitted PairAccuracy. Its guard now uses
`!IsPairwiseMetric(lossFunction)`. PairAccuracy consumes supplied edges without
requiring relevance labels; RMSE and Logloss still reject constant targets unless
explicitly allowed. Native tests cover both sides of that boundary.

## Supported entry points

| Capability | Native `CatBoostRanker(task_type="GPU")` | Standalone `CatBoostMetalRanker` |
|---|---|---|
| Training | Plain; Depthwise/Lossguide/Region; Newton/Gradient; all seven structure scores | Same greedy policies, methods and scores |
| Sampling and leaves | No/Bayesian/Bernoulli/Poisson; No/AnyImprovement/Armijo backtracking | Same |
| Data | Numeric, one-hot and four simple CTR types; raw/quantized Pools; Sample/Group histories; supplied pairs and existing shared pair generation | Numeric/one-hot arrays or supported numeric Pools; explicit supplied pairs; prepared permutation banks through the private lifecycle |
| Lifecycle | Unlabeled supplied-pair Pools, baseline/init_model, validation metrics, callbacks, best-model/early stopping, snapshots and exact extension | Validation metrics, callbacks, best-model/early stopping and snapshots; public baseline/init_model and raw ranking CTR construction remain unsupported |
| Readers | GPU prediction plus standard CPU, CBM and JSON readers | GPU prediction plus standard CPU, CBM and JSON readers |

P1/P4/P7 permutation recovery is covered. Greedy PairLogitPairwise, greedy
YetiRank, Ordered greedy boosting and MVS greedy sampling remain outside this
integration. Existing grouping, category, memory and option validation still
applies; this extension does not enable compound CTRs.

## Acceptance evidence

| Check | Verified result |
|---|---|
| [Private PairLogit suite](tests/test_greedy_pairwise.py) | 302 passed: edge equations, scores, leaf estimates, centering, signed loss differences, permutations and counted ABI validation |
| Adjacent numerical/runtime selection | 489 passed; together with the private suite, 791 focused checks |
| [Standalone acceptance](tests/test_greedy_pair_public.py) | 111 passed: literal weights, numeric/one-hot, readers, metrics, callback/best-model behavior and exact P1/P4/P7 snapshot extension |
| [Native acceptance](tests/test_native_greedy_pair.py) | 126 passed: all scores/methods, unlabeled Pools, CTRs, quantization, metrics, baseline/init_model, readers and snapshot identity/recovery |
| Preserved older snapshots | All 194 fixtures originating with `20260913T151643Z`, plus eight fresh `20260913T155047Z` baseline fixtures, replay exactly on the fixed build |

These selections may overlap; they are not additive release totals. Snapshot
checks compare saved arrays exactly and verify that callbacks begin at the first
new iteration. Original snapshots and old-build expected results are preserved.
The [release-check runner](examples/greedy_pairlogit_acceptance/README.md) records
package identity, source checksums, readers and recovery evidence.

## Installed release

Checkpoint `20260913T195124Z` is installed in `catboost/metal/.venv`.

- Full matrix: **11,099 passed plus 16 subtests**, with no skipped cases.
- Alternate extension: **3,744 passed** (3,665 main cases plus 79 supplemental one-hot reader cases; disjoint selections).
- Installed package: **237 passed**, plus **18 GPU smoke configurations**.
- CLI: **171 configurations passed** (153 preceding plus 18 new PairLogit cases).
- Snapshot compatibility: **202 exact recoveries** (194 preserved older fixtures plus eight fresh fixtures from the preceding installed checkpoint).
- Wheel: `catboost-1.2.10-cp312-cp312-macosx_11_0_arm64.whl`
- Wheel SHA256: `22a03c3d80c85a15197999c7a87a2c19e785112900df61efb6b561b4333777b8`
- Standard extension SHA256: `36b30f0b61c3f8e8e2053aff83e95622d373f8000013a571cede94a7e953e03f`
- Alternate extension SHA256: `50ea086eb6045de55263fcc3976a0b384257c0b7bff9423a76d5a95cfd2e4106`

Test selections overlap; the counts are not summed. Sources, both extensions,
CLI, original snapshots, commands and raw evidence are retained under
`.build/releases/20260913T195124Z/`. The preceding `20260913T155047Z` wheel is
preserved. Source provenance is Git commit `933ff4a86cfe1f9a1f5cccd799e8a030ae7f3eb4` plus
`source-changes.patch` and `source-overlay/` in the new release.

## Reproduction

Run from the workspace directory containing the `catboost/` checkout, on an
Apple Silicon Mac with the standalone environment installed:

```sh
PYTHONPATH=catboost/metal/python \
  catboost/metal/.venv/bin/python -m pytest -q \
  catboost/metal/tests/test_greedy_pairwise.py \
  catboost/metal/tests/test_greedy_pair_public.py

CATBOOST_NATIVE_METAL_QUERY_TESTS=1 \
PYTHONPATH=catboost/metal/.build/greedy-pair-acceptance/fixed/standard:catboost/metal/python \
  catboost/metal/.venv/bin/python -m pytest -q \
  catboost/metal/tests/test_native_greedy_pair.py
```

Use the sibling `fixed/no_cuda` package for the alternate extension. The
[release-check instructions](examples/greedy_pairlogit_acceptance/README.md)
provide CLI, installed smoke, legacy replay and fresh-baseline replay commands;
each replay requires a new output directory.

Full CUDA feature, numerical and performance parity remains open. Remaining
work includes compound/tree CTR scheduling, other ranking/Ordered/combination/
custom objectives, text/embedding pipelines, remaining GPU options/random-state
semantics, complete memory accounting and wider hardware/workload validation.
This work used no CPU CatBoost fitting or live NVIDIA comparison. Only M3 Pro
execution is established; the diagnostic agreement is with independent equations.
