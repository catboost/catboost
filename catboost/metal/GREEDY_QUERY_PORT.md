# Greedy QueryRMSE and QuerySoftMax

Native and standalone Metal training now connect QueryRMSE and QuerySoftMax
to Depthwise, Lossguide and Region. The preceding installed checkpoint
`20260913T151643Z` remains preserved. The new checkpoint is `20260913T155047Z`.

## CUDA mapping

- `cuda/train_lib/querywise_non_symmetric.cpp` and `querywise_region.cpp`
  register both query objectives for the three grow policies.
- `cuda/targets/querywise_targets_impl.h` and
  `cuda/targets/kernel/query_rmse.cu`, `query_softmax.cu` define whole-query
  weighted derivatives. Gradient structure scores use original object weights;
  Newton scores use query curvature. QueryRMSE uses the CUDA diagonal weight.
- `cuda/methods/leaves_estimation/{descent_helpers,step_estimator,walker}.cpp`
  define projection, diagonal regularization and the shared trial budget.

The greedy runtime reuses the established query point, derivative, projection
and metric kernels. Each query is normalized across its complete original row
range at the current point. Derivatives are then projected through the current
variable-node leaf partition. Bootstrap affects structure search; leaf weights
and query normalization retain the original prepared object/group weights.

The counted `cbm_greedy_session_create_query` constructor checks offset counts
before reading boundaries, validates strict coverage and accounts for resident
query workspaces. The original constructors and public parameter layout remain
unchanged. Invalid Newton structure curvature or nonpositive regularized leaf
diagonals fail explicitly. Gradient leaves ignore unused curvature.

Every CTR dataset has its own feature bins, leaf routing and prediction cursor.
The selected dataset supplies structure; each dataset estimates that topology
from its own whole-query point. The final dataset supplies exported leaves and
metrics. Native pools support all four simple CTR types and Sample/Group
histories. Standalone rankers support numeric and one-hot inputs; the private
lifecycle also accepts prepared query CTR banks. Standalone raw ranking CTR
construction remains open.

Native snapshots and standalone numeric snapshots preserve every dataset
cursor, original grouping and query parameters. Group IDs, subgroup metric
hashes, evaluation groups and beta/lambda participate in standalone snapshot
identity. Validation, callbacks, model trimming and GPU/CBM/JSON prediction
use the existing variable-node model path.

## Verification

- 240 private cases: two objectives, three policies, numeric/equality splits,
  six equation-checked scores, Newton/Gradient leaves, whole-query backtracking,
  overshoot rejection, signed curvature, P1/P4/P7 dataset cursors and counted
  ABI validation.
- 216 native cases: seven scores, both leaf methods, object/group weights,
  raw/quantized categories and four CTR types, all four supported samplers,
  baseline/initial-model continuation, exact snapshots and standard readers.
- 123 public cases: all policies/objectives, four samplers, weights, one-hot,
  metric/subgroup handling, best-model trimming, extended snapshots and private
  grouped CTR lifecycle recovery.
- Together with 105 existing greedy checks, 684 focused cases pass.
- 10560 full regression cases +16 subtests and 3507 alternate acceptance cases pass.
- 685 installed GPU paths, 153 CLI variants and 194 exact preceding snapshot
  recoveries pass. Old-build outputs and original snapshots are archived.

The three-step independent backtracking equation check has a maximum observed
leaf difference of 1.49e-8. At seven steps near convergence, float32 query loss
can change Armijo's final accept/reject branch relative to the independent
double-precision loss: maximum leaf difference 6.997e-5, metric difference
1.478e-5 in the diagnostic fixture. Tests retain strict early-step equations,
bound the convergence difference, and separately require overshoot rejection.
These are comparisons with independent equations, not NVIDIA timing or
bit-equivalence claims. Exact same-build snapshot recovery is tested separately.

PairLogit and YetiRank greedy registration, Ordered query objectives, dynamic
compound CTR scheduling and remaining GPU RNG/options still need porting.
MVS and Ordered greedy boosting remain rejected as in CUDA's registrations.
No CPU CatBoost training was used.

## Installed checkpoint

Checkpoint `20260913T155047Z` is installed in `catboost/metal/.venv`.

- Wheel: `catboost-1.2.10-cp312-cp312-macosx_11_0_arm64.whl`
- Wheel SHA256: `ed582f8039f7f7041ea05a36ee319e204c7dd5dc902fa915e76521134b2a3ef5`
- Standard extension: `5b85ebc6aac1d2536ff43c724fdfbaaab302022aed0c502f2fa3d840e69403c4`
- Alternate extension: `01fc7d223f19e1ede985f8444cbd98ce82b8918c506de64ab690a97fce8aaff1`
- 10560 tests +16 subtests pass; 3507 alternate acceptance cases pass.
- 685 installed GPU paths and 153 CLI variants pass.
- 194 preceding snapshots recover exactly against saved old-build outputs:
  96 Ordered scalar cases and 98 greedy scalar/vector/symmetric-query cases.

QueryRMSE and QuerySoftMax now support Depthwise, Lossguide and Region through
native CatBoost and standalone Metal APIs. Whole-query derivatives feed
structure scores, iterative leaves, backtracking and normalized metrics.
Native simple CTR datasets retain independent query cursors. Native baseline/
initial-model continuation, categorical/quantized Pools, snapshots, callbacks,
validation and standard GPU/CBM/JSON readers pass. Standalone rankers support
numeric/one-hot inputs and complete query metric/snapshot lifecycle.

All sources, both extensions, CLI, original old snapshots, scripts and logs are
archived. Preceding checkpoint 20260913T151643Z remains intact. CUDA parity is incomplete:
PairLogit/YetiRank greedy training, Ordered query objectives, dynamic compound
CTRs, remaining GPU RNG/options and wider workload/hardware verification are
still open. Near-convergence float32 Armijo differences from independent
double equations are quantified in GREEDY_QUERY_PORT.md; same-build snapshot
recovery is exact. No CPU CatBoost fitting, NVIDIA comparison or other M-series
execution occurred.
