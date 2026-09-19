# Ordered one-hot training on Metal

The Ordered session now accepts mixed numeric and one-hot candidates. Numeric
splits use `value > border`; one-hot splits use `value == category`. The native
CatBoost GPU API and standalone scalar estimators both use this typed session.
Ordered simple CTR histories are connected by the later [ORDERED_CTR_PORT.md](ORDERED_CTR_PORT.md) checkpoint. Grouped scalar folds are
connected by the later [ORDERED_GROUPS_PORT.md](ORDERED_GROUPS_PORT.md) checkpoint.

## CUDA translation

CUDA's `data/binarizations_manager.cpp::UseForOneHotEncoding` applies the
one-hot threshold to `OnAll`, including evaluation-only categories. The Metal
frontends use this same cardinality decision. Original category hashes remain
in model splits and snapshot identity. A constant categorical column contributes
no candidates; it does not trigger an Ordered CTR calculation.

CUDA's `methods/kernel/split_properties_helpers.cuh` distinguishes equality
histograms from numeric prefixes. Metal retains its paired histogram sums and
recovers an equality bin from adjacent prefixes, then subtracts it from the
whole partition for the other child. This also works with tiled histograms,
sibling reuse and the bounded direct-candidate fallback for deeper trees.
The native partition update and exported split type use the same comparison.

The private candidate-pair buffer packs equality in its high bit. Existing
numeric pairs, buffer widths and bindings retain their previous representation.
The old session-create C ABI remains available; an additive typed entry point
accepts comparison flags. Numeric snapshot fingerprints remain unchanged.
Typed fingerprints include the flags; public snapshots also retain original
learn/evaluation category hashes, even when category renaming preserves dense
bin numbers.

## Validation

- 186 private checks cover independent histogram sums, CUDA-derived scalar
  equations, every prefix cursor, all five samplers, backtracking, P1/P4,
  equality bin 255, malformed inputs and snapshot type identity.
- 108 new native/public checks cover weighted scalar fitting, fixed private
  runtime comparisons, block permutations, raw/quantized pools, validation,
  unknown categories, thresholds, original labels, class weights, CBM/JSON/GPU
  readers, baselines/initial models, best-model trimming and exact recovery.
- The first targeted API run passes all 147 selected tests, including existing
  numeric Ordered acceptance.
- 8981 combined tests plus 16 subtests; 2737 alternate acceptance cases.
- 379 installed GPU fitting/prediction/recovery paths; 102 CLI variants,
  including Ordered one-hot RMSE, Logloss, Poisson, Lq and Quantile at P1/P4.
- Twenty-four numeric Ordered snapshots written by the preceding native and
  standalone sources resume exactly: RMSE, Logloss and Huber, P1/P4, No/MVS.
  Saved reference leaves, weights, GPU predictions, histories and every
  standalone prefix/selection cursor agree. Original snapshots are preserved.
- Installed checkpoint `20260913T135844Z`, wheel SHA256 `cd087d7a2b7b3926076bc504e08cbb1462a8f68e894d52b0ead10e40e0b5ece5`.
  Both extensions, CLI, source hashes, snapshots and raw evidence are archived;
  preceding checkpoint `20260913T132447Z` remains intact.

No CPU CatBoost fitting, NVIDIA comparison or other M-series hardware run is
used for this port. Equation and Metal checks do not establish complete CUDA
quality or performance parity.
