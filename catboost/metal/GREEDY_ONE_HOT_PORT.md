# One-hot categories for greedy trees

Native CatBoost and the standalone Metal estimators now accept one-hot
categorical inputs for Depthwise, Lossguide and Region. All eleven
CUDA-registered scalar losses use the existing GPU one-hot histograms,
equality routing, variable-tree leaf estimators and model builders.

The native numeric-only feature gate is removed. Categorical preparation
still rejects greedy CTR scheduling before computing histories. Numeric and
one-hot inputs use one dataset regardless of requested permutation count.
Original category hashes and feature indices are retained in normal models
and snapshots; inference handles unseen categories with equality routing.

Standalone arrays/DataFrames use the existing categorical layout and variable
tree JSON builder. Their preflight counts the union of original learn and
validation hashes against one_hot_max_size. Snapshot metadata adds original
validation hash digests for categorical fits, so replacing one unseen value
with another cannot silently reuse a snapshot. Existing numeric snapshot
metadata is unchanged. Categorical/prequantized Pool extraction remains a
native-interface capability.

## Validation

- 81 native cases: eleven losses across three policies, fixed-bin resident
  forest comparisons, original weights, raw/quantized Pools, four samplers,
  exact snapshots, weighted Exact leaves, binary backtracking, initial models,
  baselines, pure/mixed/constant/ignored category columns, OnAll thresholds,
  and CBM/JSON/GPU readers.
- 75 standalone cases: exact resident trees and cursors, all eleven losses,
  four samplers, snapshots, original learn/eval hash identity, backtracking,
  best-model selection, probabilities, DataFrame names/order, unseen values
  and standard model readers. Broader focused runs include 151 existing
  native greedy cases and 55 existing standalone cases.

The first native recovery fixture exceeded its own one-hot threshold after
adding a changed category. Its limit was corrected to reach snapshot identity
validation; training code was unchanged. This failed fixture log is preserved.

Greedy CTR scheduling/penalties and vector objectives remain open. Dynamic
categorical tensors, grouped Ordered, full CUDA GPU random-buffer behavior,
remaining feature pipelines/options and wider hardware comparisons also
remain open. No CPU CatBoost fitting or NVIDIA execution was used.

Checkpoint `20260913T111808Z` is installed: **7147 tests +16 subtests**,
**1734 alternate acceptance cases**, **178 installed GPU paths**, **53 CLI
variants**. Wheel SHA256 `607b0648328a7e72deae1565c4afc5e8915e8b2780449f62bf3e3879d73a19fe`. Sources, both extensions, CLI, scripts
and logs are preserved; prior checkpoint `20260913T110339Z` remains intact.
