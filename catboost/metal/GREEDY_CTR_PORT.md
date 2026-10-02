# Native greedy CTR datasets

Depthwise, Lossguide and Region now accept all four connected simple CUDA
CTR types: Borders, Buckets, FloatTargetMeanValue and FeatureFreq. The native
CatBoost estimators, raw/prequantized Pools and CLI use the existing GPU CTR
histories, shared grids and standard non-symmetric model tables. Sample and
Group histories are supported. `has_time=true` retains the P1 fallback;
ordinary categorical training can use the default P4.

## CUDA mapping

- `cuda/methods/doc_parallel_boosting.h::Fit` chooses the structure-search
  dataset from the absolute iteration seed, clones its topology, estimates
  each dataset's leaves at its own cursor, and exports the final dataset.
  The common `TMetalDocParallelPermutations` adapter now dispatches to the
  greedy runtime as well as scalar/vector symmetric sessions. Its existing
  literal CUDA selection modulus is retained.
- `cuda/methods/greedy_subsets_searcher/greedy_search_helper.cpp` updates CTR
  model-size penalties and used-CTR state only for symmetric search. Native
  greedy CTRs therefore keep the existing constant feature weights; they do
  not acquire symmetric-only state in their snapshots.
- `native/metal_greedy_trainer.mm` keeps one GPU bin matrix and raw cursor per
  dataset. After searching, it replays the flat tree on each remaining bank,
  stably sorts original row indices by leaf ID on Metal, and runs the same
  original-weight Newton/Gradient/Exact or backtracking estimator. Sampling
  affects structure search; fixed leaves retain original sample weights.
- The final bank supplies exported leaf values, weights, loss and prediction
  cursor even when a private caller selects it for structure search. The P1
  path preserves its prior arithmetic and avoids extra routing/sorting.

The additive C ABI permits 1–64 banks with identical geometry, split grids
and original object order. Configuration is allowed once before training.
Inputs and finite cursors are validated and copied. MVS state is rejected,
matching greedy CUDA registrations. Workspace checks include bank copies,
flat-tree storage and retained radix scratch under the existing 1 GiB cap.

Native snapshots persist every P4 cursor, check its geometry and require the
export cursor to match the final bank. Existing P1 snapshots retain their
empty permutation arrays. Absolute iteration offsets preserve sampling and
dataset selection through callbacks, initial models and resumed fitting.

## Validation

- 146 private GPU cases compare all twelve private scalar objectives against
  independent leaf equations and single-dataset structure searches. They
  cover original/zero weights, equality predicates, distinct cursors, every
  search dataset, backtracking, Exact quantiles, all four samplers, recovery,
  P1/P2/P4/P7/P64, empty candidate sets, depth zero, deep paths and invalid
  state. The private Lq extension remains excluded from native CUDA-style
  greedy registrations.
- 204 new native cases cover eleven registered scalar losses, all four CTR
  types, P1/P4, Sample/Group histories, raw/prequantized Pools, exact snapshots,
  initial models/baselines, backtracking, Exact leaves and CBM/JSON CPU/GPU
  readers. Twenty-four reconstruct exclusive CTR banks independently from
  the stored shared-Pool shuffle fixture and compare resident forests. They
  also verify CUDA's absence of changing symmetric model-size penalties.
- The focused native run includes 81 preceding one-hot cases: 285 passed.
  OnAll threshold tests now verify the transition from one-hot to CTR.
- The first native run exposed a shared snapshot saver calling the symmetric
  feature-state API for greedy sessions. That call is now restricted to the
  applicable backend. Reader fixtures also permit valid root-only trees.
  The failure log is retained with recovery artifacts.

The combined, alternate-library, installed-package and CLI acceptance results
are recorded with the packaged checkpoint below once complete. No CPU
CatBoost fitting or NVIDIA execution is involved in these checks.

## Remaining scope

Standalone greedy estimators still expose numeric/one-hot inputs; their CTR
lifecycle needs all-bank fingerprinting and snapshot integration. Native
compound/dynamic CTRs, vector greedy training, grouped/categorical Ordered,
complete CUDA GPU mutable random state, additional options and broader
memory/performance validation remain open. This is M3 Pro validation, not a
claim of complete CUDA feature or speed parity.

Checkpoint `20260913T113921Z` is installed: **7497 tests +16 subtests**,
**1938 alternate acceptance cases**, **223 installed GPU paths**, **65 CLI
variants**. Wheel SHA256 `9ece3715d82bc136b173b65c90bd9baf3588f9d9e95ffa80b3b1f283a1cc0b82`. Both extensions, CLI, changed sources,
scripts and logs are preserved. Prior checkpoint `20260913T111808Z` remains intact.
