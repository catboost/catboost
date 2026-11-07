---
title: Support model-based feature analysis
status: done
priority: 7
created: 2026-09-13
tags: [catboost, metal, parity]
---

# Support model-based feature analysis

[Board](README.md)

Discovered during the card 4 training-options audit. This is the separate
feature-ablation analysis workflow exposed by `model-based-eval`.

- [x] Port the CUDA trainer's `ModelBasedEval` workflow and feature-set controls.
- [x] Restore a pre-existing baseline snapshot and reconstruct experiment prefix cursors, including per-permutation history; ordinary `init_model` continuation is not equivalent.
- [x] Reuse the registered Metal training paths for baseline and feature-set runs.
- [x] Validate fold histories, baselines, output artifacts and supported CLI options.
- [x] Test explicit rejection of unsupported configurations and interrupted runs.

Completed and installed in checkpoint `20260914T052913Z`. Native Plain
DocParallel `model-based-eval` restores compatible baseline prefixes and writes
per-trial metric histories through Metal training. The baseline must already
exist; the analysis command does not fit it or export trial models.

Both baseline policies, feature names/ranges/overlaps, per-permutation history,
registered scalar/query/vector paths, legacy snapshots, interruption and output
failures passed acceptance. Evaluated candidates are numeric features with
borders; supported categorical features can remain background inputs. Ordered,
FeatureParallel, multitarget/custom objectives and text/embedding estimators are
explicitly rejected by this workflow.

The full matrix passed 14,551 tests plus 16 subtests; alternate passed 6,187;
installed passed 2,606. All three include the 69 CLI feature-analysis cases.
The full and installed matrices additionally include 106 leaf-export cases.
The 350 original snapshot recoveries, 348 CLI configurations, both 203-case
smoke gates and 143 host checks passed. These counts overlap.

Metal random-state helpers still restart per experiment, and rounded vector
prefix replay can differ from live fused cursor updates. Shared CUDA RNG
carryover and broader numerical agreement remain on deferred card 5. This card
completes the supported analysis workflow without claiming full CUDA numerical
or performance parity.

[Release report and usage](../catboost/metal/MODEL_BASED_EVAL_PORT.md).

CUDA loads the baseline snapshot rather than training the baseline in this
command. Its experiments start at prefixes of that baseline and produce local
metric histories under `feature_set<index>_fold<index>` directories. Functional
coverage can use small datasets on the current M3 Pro. FeatureParallel boosting
explicitly rejects this workflow in CUDA.

[Metal entry point](../catboost/metal/train_lib/train.cpp)
· [CUDA implementation](../catboost/cuda/train_lib/train.cpp)
