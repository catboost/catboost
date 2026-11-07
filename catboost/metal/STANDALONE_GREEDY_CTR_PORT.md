# Standalone greedy CTR lifecycle

`CatBoostMetalRegressor` and `CatBoostMetalClassifier` now train
Depthwise/Lossguide/Region with the standalone backend's Borders and
FeatureFreq CTRs. The native backend's four CTR types remain documented in
[GREEDY_CTR_PORT.md](GREEDY_CTR_PORT.md).

The frontend preserves the default P4 for categories requiring CTRs, permits
explicit P1–P64, and uses the shared CUDA DocParallel dataset selector at
each absolute iteration. GPU sessions keep separate cursors, search one
dataset and estimate fixed leaves independently. Greedy training skips the
dynamic model-size penalty that CUDA applies only to symmetric search.

Learn plus validation category counts now control the one-hot/CTR threshold.
The shared preprocessing helper accepts those counts without adding eval
rows to the learn CTR histories or final tables. DataFrame names, column
order and categorical indices remain checked. Numeric/one-hot input still
uses one dataset. Categorical Pool extraction remains unsupported by the
standalone wrapper; native CatBoost Pools support these features.

Snapshots bind every bank's quantized matrix and both learn/evaluation
original category hashes. P4 snapshots retain all raw cursors, validate their
shape and finiteness, and require the exported cursor to equal the final
bank. Snapshot state is written before best-model trimming. Resuming uses
all saved cursors and the absolute iteration; trimming evaluates retained
trees on the final history bank. P1 array names and fingerprints remain
compatible with the preceding release.

## Evidence

- 130 new GPU tests cover eleven registered losses, both CTR types, four
  samplers, backtracking, Exact quantiles, best-model selection, P1/P2/P7/P64,
  exact resident forests and cursors, standard CBM/JSON CPU/GPU readers,
  malformed snapshot arrays, original category identity and OnAll thresholds.
- 116 existing one-hot and greedy lifecycle cases pass. The former threshold
  rejection now checks the actual one-hot-to-CTR transition.
- Six snapshots produced using the preceding release's Python sources resume
  exactly: numeric and one-hot data for each of the three grow policies.
  Predictions, node arrays, leaf values/weights and histories match direct
  runs. The compatibility script copies legacy Python outside the preserved
  release before execution.

The native wheel is reused byte-for-byte from `20260913T113921Z`; native
training, extension binaries and CLI have not changed in this source phase.
Combined/alternate acceptance, installed standalone paths and recovery hashes
are recorded in the source checkpoint after verification. The 65 native CLI
checks from that identical binary remain preserved.

No CPU CatBoost fitting or NVIDIA execution was performed. Compound CTRs,
additional standalone CTR types, vector greedy training, grouped/categorical
Ordered, complete CUDA GPU random state and wider hardware/performance
validation remain open.

Source checkpoint `20260913T115334Z` passes **7627 tests +16 subtests**,
**2068 alternate acceptance cases**, **289 installed GPU paths**, and six
legacy P1 snapshot recoveries. Wheel SHA256 `9ece3715d82bc136b173b65c90bd9baf3588f9d9e95ffa80b3b1f283a1cc0b82` matches installed
checkpoint `20260913T113921Z` exactly. Its **65 CLI variants** and unchanged CLI binary
are retained as prior verification; both extensions, changed sources, scripts
and logs are preserved with this checkpoint.
