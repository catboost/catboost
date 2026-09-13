# Metal implementation status

Updated 2026-09-13 (local time). CUDA source revision: `e68a1c021f`.
This is an active port. Standalone features, rebuilt native packages, and
components awaiting integration are distinguished below. The earlier
[PORT_REVIEW.md](PORT_REVIEW.md) remains a historical source inventory.

## Current local checkpoint

Compound categorical CTRs are complete for native scalar symmetric Plain/Ordered
FeatureParallel training and installed in checkpoint `20260913T220233Z`.
The release includes Sample/Group histories, retained permutation grids, exact
snapshots and standard model tables. Full regression, alternate-extension,
CLI, prior-snapshot and installed-package checks pass; see
[COMPOUND_CTR_PORT.md](COMPOUND_CTR_PORT.md).
The release records base commit `470931d02c`, the local source patch and added
files. Greedy PairLogit acceptance remains included; the previous installed
checkpoint `20260913T195124Z` is preserved. Earlier dated checkpoint notes below
are historical.

## Verified interfaces on M3 Pro

| Area | Standalone Metal estimators | Latest tested native package |
|---|---|---|
| Entry points | Regressor, Classifier, Ranker | Ordinary CatBoost estimators, Pool, CLI, `task_type="GPU"` |
| Scalar losses | RMSE, Logloss, CrossEntropy, Poisson, Huber, Expectile, Lq, Tweedie, LogLinQuantile, Quantile, MAE, MAPE | All twelve |
| Vector losses | MultiClass, MultiClassOneVsAll, MultiRMSE, RMSEWithUncertainty, MultiLogloss, MultiCrossEntropy | All six, including weighted metrics, baselines and snapshots |
| Grouped losses | QueryRMSE, QuerySoftMax, supplied-pair PairLogit/PairLogitPairwise, QueryCrossEntropy, classic YetiRank and YetiRankPairwise; one-hot arrays/DataFrames for all seven | All seven; all seven support native one-hot/CTR P4, metrics and snapshots |
| Symmetric trees | Numeric and simple categorical training, depth 0–16 | Raw/prequantized Pools; scalar Plain/Ordered FeatureParallel compound CTRs |
| Non-symmetric trees | Numeric/one-hot/CTR Depthwise, Lossguide, Region; eleven scalar, three vector and QueryRMSE/QuerySoftMax/PairLogit CUDA-registered losses, seven scalar/five vector scores, Newton/Gradient/Exact where applicable and backtracking | Numeric/one-hot/CTR P4, eleven scalar, three vector and QueryRMSE/QuerySoftMax/PairLogit objectives, snapshots, callbacks, best-model trimming and GPU evaluation |
| Ordered boosting | Numeric/one-hot/simple CTR scalar training, Sample/Group CTR histories, grouped folds, P1-P64 prefix recovery and static FeatureParallel penalties | All four simple/compound CTR types, Sample/Group histories, raw/quantized Pools, baselines/initial models and exact lifecycle |
| Categoricals | One-hot and simple Borders/FeatureFreq CTRs with P4 scalar/multiclass cursors | All four simple CUDA CTR types; scalar symmetric FeatureParallel combinations with category/numeric/one-hot projections, Sample/Group histories, retained P1/P4 grids and final tables |
| Scores | Seven symmetric/greedy scalar scores; vector L2/Cosine/SolarL2/LOOL2/SatL2 | All seven scalar and five vector scores |
| Bootstrap | Symmetric scalar/query: No/Bayesian/Bernoulli/Poisson/MVS; vector and greedy exclude MVS | Scalar/multiclass sampling and exact continuation verified |
| Lifecycle | Weighted validation, stopping, best models, callbacks, safe numeric snapshots, exact optimizer/permutation state | Multiple evaluation sets, shared metrics, callbacks, baselines, initial models, snapshots |
| Inference/export | GPU symmetric and variable-tree evaluation; standard CBM/JSON | Normal GPU prediction for symmetric and variable trees, numeric/category/CTR/prequantized Pools and vector outputs |

Restrictions are checked explicitly. Ordered+Exact remains rejected like CUDA;
private experimental coverage is not a public CUDA capability. Public greedy
Lq is also excluded because CUDA does not register it for those policies.
Lossguide accepts larger requested depth, bounded by max_leaves; actual GPU
training/export at depth 100 passes. Native reader tests additionally cover
much deeper constructed models, without claiming every maximum-size model is
validated. Standalone multiclass and greedy RMSEWithUncertainty accept numeric, one-hot and simple CTR inputs; other standalone multioutput paths currently use numeric data. Standalone rankers accept one-hot categories.

An upstream CatBoost wheel alone does not contain Metal. The development
adapter compiles its runtime beside the installed model/data library. The
native fork includes Metal and uses the existing GPU task type.

## Current verification and packaged checkpoints

The installed checkpoint **20260913T220233Z** passes **11,323 cases plus 16 subtests**. The alternate extension passes **3,892 tests**; the C++ tensor/history/RNG/snapshot/helper suites pass **41 checks**. Counts overlap and are not summed; no native acceptance file is excluded.

Native numeric/one-hot/CTR Depthwise/Lossguide/Region now includes eleven CUDA-registered
scalar objectives plus MultiClass/MultiClassOneVsAll/RMSEWithUncertainty and QueryRMSE/QuerySoftMax/PairLogit, all seven scalar and five vector scores, four samplers, applicable leaf estimators
and backtracking, snapshots, initial models, baselines, callbacks, multiple
validation sets and best-model selection. Compact native GPU evaluation also
supports non-symmetric categorical/CTR and vector models. The shared model
trimmer now handles variable trees, rebasing node/leaf offsets and retaining
weights and scale/bias without exponential tree padding.

Wheel installed in `catboost/metal/.venv`:

```text
catboost/metal/.build/releases/20260913T220233Z/
  catboost-1.2.10-cp312-cp312-macosx_11_0_arm64.whl
SHA256 f1d79ee43aeebfbb65341ac70d10caf83150636f0663c00b11537bf24ba41ffe
```

The embedded extension hash matches the tested isolated package. 385 installed
acceptance cases plus 34 GPU smoke configurations pass. All 187 CLI configurations
and 214 preceding snapshot recoveries pass. Compound CTR acceptance includes
148 native cases with independent table/prediction checks and exact lifecycle,
plus 238 private dynamic runtime/scoring cases in the full matrix.
Classic numeric YetiRank adds
resident target/search/leaves, complete host target RNG, real PFound evaluation,
seven scores, five samplers, snapshots, baseline/initial models and export.
Its 253 component/controller/native cases include independent equations and
14 native fixed-quantization comparisons; see [YETIRANK_PORT.md](YETIRANK_PORT.md).
Numeric native PairLogitPairwise adds full coupled candidate matrices, original-edge
Newton/Gradient leaves, original document weights, four edge samplers, all three
backtracking modes and native/standalone lifecycle. Its 65 resident training,
37 standalone lifecycle and 36 native
cases pass; [PAIRWISE_MATRIX_PORT.md](PAIRWISE_MATRIX_PORT.md) records the source
mapping and remaining categorical/P4 gaps. QueryCrossEntropy adds a full
point diagonal plus query Laplacian, bounded query/candidate tiling, original
weights, Newton/backtracking, scaled GPU metrics and native/standalone lifecycle.
It passes 57 resident, 30 public and 41 native API cases, plus target, metric
and runtime component checks. See [QUERY_CROSS_ENTROPY_PORT.md](QUERY_CROSS_ENTROPY_PORT.md).
The PairLogitPairwise checkpoint `20260913T054109Z` remains preserved.
The YetiRank checkpoint `20260913T044013Z` remains preserved.
The earlier greedy/ranking checkpoint `20260913T035730Z` remains preserved.
The prior checkpoint `20260913T032341Z` (4,295 combined cases, 343 alternate native)
remains preserved, including its recovered vector/baseline/metric/build fixes.
Earlier checkpoints below are retained as historical recovery evidence.

- The earlier Plain wheel `20260913T012122Z` passed 167 native tests on each
  extension variant. The subsequent source checkpoint `20260913T020325Z`
  passed 208 standard tests. Their artifacts remain preserved.
- The recovered aggregate includes the optimized Ordered runtime, dynamic
  append/masks, group-aware CTR kernels, and standalone greedy/vector/ranker
  paths. Pairwise matrix projection, candidate scoring and leaf solving are now
  connected to the native trainer. QueryCrossEntropy now has complete native and
  standalone numeric training with scaled GPU metrics: 57 resident training,
  30 standalone lifecycle and 41 native API cases pass. Its complete regression,
  CLI verification and packaging passed; see QUERY_CROSS_ENTROPY_PORT.md.

## CUDA canonical comparisons

[compare_cuda_fixtures.py](examples/compare_cuda_fixtures.py) reproduces the
checked-in CUDA `test_grow_policies` configuration through native Metal. It
records data/fixture/extension hashes and every learn/validation metric.
These are saved CUDA references, not live NVIDIA execution.

| Objective / score | Maximum validation-history difference over 20 trees |
|---|---:|
| RMSE / L2 | 1.282e-8 |
| RMSE / Cosine | 1.098e-8 |
| Logloss / Cosine | 1.016e-7 |
| MultiClass / Cosine | 9.094e-8 |
| Logloss / L2 | 0.000563444; validation-only tied/empty-child split difference remains |
| MultiClass / L2 | 0.01676455; tied equivalent CTR aliases change later used-feature penalties |

Artifacts: `.build/cuda-fixtures-compensated/`. Compensated histograms/prefixes
and CUDA-style mixed-precision scores corrected the RMSE discrepancies.
A separate one-hot calcer fix preserves CUDA's selected-bin-first accumulation
order. The remaining L2 differences trace to tied candidates. In MultiClass,
two CTR priors have identical learn and validation predicates; the historical
CUDA choice changes later used-feature penalties. An explicitly labelled
counterfactual in [audit_multiclass_ctr_tie.py](examples/audit_multiclass_ctr_tie.py)
matches all 20 histories within 9.50e-8. It diagnoses the source of divergence;
normal training does not force fixture-specific choices or widen tolerances.

## Performance and larger data

Classic YetiRank trained 30 depth-four trees on 32,768 numeric rows and 8
features in 0.286 seconds in one local run with a warmed shader cache. Held-out
8,192-row NDCG@10 improved from 0.315 (constant scores, shared tie convention)
to 0.985. GPU and standard model-reader outputs matched exactly. The preserved
`20260913T044013Z/yeti-quality.json` includes inputs/options and extension hash;
this is synthetic correctness/quality evidence, not a NVIDIA speed comparison.

The million-row numeric example trained 30 depth-six trees on 16 features in
2.074 seconds total fit time, including 1.235 seconds preprocessing; measured
training GPU time was 0.315 seconds. Held-out RMSE improved from 2.571 to 1.028,
and GPU versus standard-reader predictions differed by at most 5.56e-17.
Artifacts: `.build/scaling/results.json`. This is one local development run,
not a NVIDIA comparison; compile/setup and concurrent work affect timings.

The Ordered histogram rewrite's final five-run comparisons improved median
wall time from 1.022 to 0.202 seconds for 32,768 rows (5.07x), and from 3.417
to 0.144 seconds for 65,536 rows (23.78x). The smaller 8,192-row case improved
only 1.04x, with host setup dominating. Every tree and fold cursor remained
bit-identical across the three saved workloads. Raw runs are under
`.build/ordered-histograms-*-final.json`; see [ORDERED_PORT.md](ORDERED_PORT.md).

Earlier Adult and synthetic benchmark artifacts remain available under
`.build/adult-full/` and `.build/iteration2-regression/`. Adult test Logloss
0.278124, accuracy 87.23%, and AUC 0.926768 were measured **before** the later
permutation/CTR-penalty changes; these are historical results, not current
configuration-matched CUDA quality evidence.

## Remaining work

- Extend the standalone estimator frontend to the compound CTR machinery now
  connected through native scalar Plain/Ordered FeatureParallel training.
- Finish remaining CUDA ranking/combination/custom objectives and complete
  CUDA GPU random state. All seven connected grouped objectives now support
  native simple CTR P4, including both classic YetiRank variants and the
  full-matrix objectives. Generated-pair sampling uses explicitly recorded
  Metal streams; see YETIRANK_PAIRWISE_CTR_PERMUTATIONS.md.
- Finish Ordered query objectives, query/ranking FeatureParallel training,
  greedy YetiRank objectives and CUDA GPU random-stream agreement.
- Integrate shared text/embedding estimated features and remaining training
  options/feature penalties; avoid CPU-training fallback.
- Validate complete memory use, wider/deeper workloads, packed feature layouts,
  and performance across M-series generations. Current major working/output
  guards are 1 GiB/512 MiB; host copies and retained category tables need their
  own accounting. Typical scalar bounds remain 255 borders and 2^24 rows.
- Compare numerical agreement and performance on actual NVIDIA and additional
  M-series hardware when available; repeat coherent release acceptance for
  subsequent capability changes.

Only the M3 Pro (18 GPU cores, 18 GB unified memory) has been exercised. No
M1/M2/M4 or live NVIDIA speed/quality comparison has been performed. Full CUDA
feature, numerical, and performance parity is not established.

QCE now chooses 32/64/128/256 query threads, with 18 bitwise boundary checks.
Three alternating warmed runs show 2.05x/1.61x median native speedups at
query sizes 16/64 and unchanged performance at 256. Complete forests and
metric histories match exactly. Full-matrix sessions omit scalar histograms
and reserve core memory before target tiling; twelve wide-feature tests pass.
See QUERY_CROSS_ENTROPY_PORT.md and the preserved benchmark artifacts.

The installed full-matrix build adds Simple leaves for PairLogitPairwise
and QueryCrossEntropy: 100 resident/default, 34 native and 12 public lifecycle
cases pass. It exports the winning sampled split solution and raw matrix
diagonal in model leaf order. See SIMPLE_LEAVES_PORT.md; coherent acceptance, both CLI targets
and packaging passed.

## New YetiRankPairwise integration

The installed native/standalone build passes 91 native API and 23 standalone
lifecycle cases, plus 111 private forests and GPU component tests. It connects
Simple/Newton/Gradient, all seven scores, No/Bayesian/Bernoulli Object or Group,
PFound metrics, exact snapshots and saved GPU models. The 5762-test combined run, 773 alternate acceptance cases, 24 installed paths
and six native CLI variants pass; see YETIRANK_PAIRWISE_PORT.md for the
explicit Metal RNG convention, dense workspace bound and remaining parity.

Generated PFound zero-pair compaction is installed. It preserves all compared
models/histories bit-for-bit and improves median native fit time 1.26x/2.94x/
10.28x for query sizes 16/64/256 in three alternating warmed runs. Adaptive sparse generation now reduces target storage; the original dense
pair-ID limit remains. See YETIRANK_PAIRWISE_PORT.md.

Adaptive sparse PFound generation is installed after 110 new component/runtime
cases. All benchmark model and metric-history bytes match the prior compacted
dense build. GPU workspace scales with sampled rows and permutations when
that is smaller than the dense triangle; detailed timings and memory reports
are preserved. Original logical pair IDs now extend through uint32-minus-sentinel; stored
contributions remain limited to 2^24 within the combined 1 GiB budget.

Standalone PFound now preserves subgroup metadata in Pool/array evaluation,
weighted histories, best-model selection and snapshot identity. The native
helper passes stored uint32 hashes directly to the shared metric to avoid
rehashing collisions. See SUBGROUP_METRICS_PORT.md.

Sparse PFound now trains above the old 24-bit logical pair-ID limit. New
integer prefix scans saturate before addition overflow. 58 prefix cases,
five new sampler cases, four resident high-ID comparisons and twelve native
forest/snapshot/export cases verify the wider domain. See WIDE_PAIR_IDS_PORT.md.

QueryCrossEntropy metric-only workspaces are now reused by native and
standalone progress trackers under a separate 256 MiB retention cap. They
preserve model/history bytes and yield 1.17x/1.23x median native fit speedups
in the measured four-metric/two-validation workload. Default standalone QCE
evaluation also avoids duplicate computation. See QCE_METRIC_WORKSPACES.md.

Native one-hot categorical ranking is connected for YetiRank, YetiRankPairwise,
PairLogitPairwise and QueryCrossEntropy. CUDA OnAll threshold semantics,
unlabeled explicit pairs, raw/prequantized Pools and exact recovery pass 66
new GPU cases. See RANKING_ONE_HOT_PORT.md. Ranking CTR/P4 remains gated.

Standalone categorical ranking arrays/DataFrames now use CUDA OnAll thresholds
and bind original category dictionaries/validation hashes in snapshots. 79 new
GPU cases cover the seven query objectives. See STANDALONE_RANKING_ONE_HOT.md.
The existing native wheel is reused byte-for-byte in this source checkpoint.

Native Group CTR histories are connected to the tested whole-query prefix
primitive. RMSE/QueryRMSE/QuerySoftMax/PairLogit P1/P4, 98 new GPU cases and
four grouped-CTR CLI variants pass. See GROUPED_CTR_PORT.md. Native YetiRank
and full-matrix ranking CTR controllers remain gated.

Native YetiRank/generated/full-matrix ranking accepts all four simple CTR
types with permutation_count=1 or has_time=true. 112 new cases verify
independent histories/forests, exact recovery, penalties and initial models.
Multiple CTR datasets remain gated; see RANKING_CTR_P1_PORT.md.

Native PairLogitPairwise and QueryCrossEntropy now support CTR P4 and private
multi-cursor banks up to P64. CUDA Simple reuses the searched model; other
methods estimate leaves independently. 55 runtime and 99 native cases pass.
See MATRIX_CTR_PERMUTATIONS.md and YETIRANK_CTR_PERMUTATIONS.md; generated-pair YetiRankPairwise P4 remains open.

Classic YetiRank now accepts native CTR P4 with per-dataset oracle seeds,
cursors and MVS state. 97 runtime and 122 native checks include 32 independent
history/forest comparisons. See YETIRANK_CTR_PERMUTATIONS.md.

Generated-pair YetiRankPairwise CTR P4 is installed with per-cursor fixed
Bayesian targets and exact Simple model reuse. 55 new runtime and 156 native
cases pass; full CUDA GPU seed-buffer consumption remains open.
See YETIRANK_PAIRWISE_CTR_PERMUTATIONS.md.

Native and standalone Depthwise/Lossguide/Region now support one-hot
categories for all eleven registered scalar losses. 81 native and 75
standalone cases verify exact routing/recovery and readers. Simple CTR scheduling and native/standalone vector greedy are connected. See GREEDY_ONE_HOT_PORT.md.

Native Depthwise/Lossguide/Region simple CTR P1/P4 is installed. All eleven
registered scalar losses, four CTR types, Sample/Group histories and complete
per-dataset recovery pass 204 new native and 146 private runtime cases.
Standalone greedy CTR lifecycle and native/standalone vector greedy are connected. See GREEDY_CTR_PORT.md.

Standalone greedy Borders/FeatureFreq CTR P1/P4/P64 lifecycle is verified,
including OnAll thresholds, complete cursor snapshots and final-bank trimming.
130 new GPU tests and six exact legacy P1 recoveries passed at the preceding source checkpoint, which reused the 20260913T113921Z native wheel. The current vector greedy checkpoint rebuilds both extensions and reruns native CLI acceptance. See STANDALONE_GREEDY_CTR_PORT.md.

Native vector Depthwise/Lossguide/Region is installed for the three CUDA-registered objectives, including numeric/one-hot/CTR P4, vector-leaf model construction and exact optimizer/published cursor snapshots. 158 shader +273 runtime +225 native new cases, 316 installed paths, 92 CLI variants and nine prior-native scalar snapshot recoveries pass. See VECTOR_GREEDY_PORT.md.

Standalone vector Depthwise/Lossguide/Region lifecycle is installed for all three registered objectives. 68 evaluator +336 public/lifecycle checks cover vector outputs, numeric/one-hot/CTR P1–P64, labels/weights, all scores/samplers, baselines, exact optimizer/published recovery and best-model extension. Nine snapshots written with the preceding standalone sources/runtime resume exactly. Vector evaluation versus repeated scalar Metal evaluation: 3 outputs: 1.10x wall / 2.99x GPU; 7 outputs: 1.21x wall / 3.95x GPU; 64 outputs: 1.25x wall / 10.00x GPU, five alternating warmed runs on 65536 rows and 16 trees with identical predictions. These are Metal-to-Metal measurements, not NVIDIA comparisons. See VECTOR_GREEDY_PORT.md.

Native host validation now avoids constructing error strings for successful literal-message checks across 17 helpers. The full matrix still passes 8687 +16, 2629 alternate, 343 installed and 92 CLI cases; 36 preceding scalar/vector snapshots recover exactly. Five alternating warmed evaluator runs improve wall time 1 output(s): 5.50x; 3 output(s): 11.96x; 7 output(s): 17.38x; 64 output(s): 22.82x. Native fit ratios (roughly unchanged): scalar_greedy: 0.98x; vector_greedy: 1.01x; scalar_ordered: 0.97x. All compared models/predictions/history hashes match. See VALIDATION_FASTPATH.md for workload details and raw evidence.

Ordered one-hot is installed through typed histogram extraction, deep fallback, split routing, native/standalone training, original hash model export and prefix recovery. 186 private and 108 public/native new cases pass; 8981 +16 full, 2737 alternate, 379 installed and 102 CLI cases pass. Twenty-four preceding numeric Ordered snapshots recover exactly against saved old-build outputs. See ORDERED_ONEHOT_PORT.md. Ordered CTR histories remain explicitly gated; grouped scalar folds are connected in the later grouped checkpoint.

Grouped scalar Ordered is installed with whole-group block permutations, variable prefix counts, double-precision group growth and the CUDA zero-quality count guard. 345 private +67 native +86 public new cases pass; 9479 +16 full, 2890 alternate, 433 installed and 111 CLI cases pass. Forty-eight prior numeric/one-hot Ordered snapshots recover exactly. Group-unit sampling, query Ordered objectives and CTR histories remain open. Public fold growth is normalized to float32 like upstream CUDA; no precision migration is required. See ORDERED_GROUPS_PORT.md.

Correction after tracing the upstream option type: `FoldLenMultiplier` is `TOption<float>` and is promoted to double only inside CUDA fold construction. The native and standalone public frontends already follow that normalization. An attempted raw-double migration was rejected and its seven source edits were restored byte-for-byte from checkpoint 20260913T142818Z; no attempted wheel was installed. Private raw-double group helper tests do not change the public option contract.

Ordered simple CTR checkpoint: Checkpoint `20260913T151643Z` is installed in `catboost/metal/.venv`.

- Wheel: `catboost-1.2.10-cp312-cp312-macosx_11_0_arm64.whl`
- Wheel SHA256: `54fd3d73943ed12d9c5e2d799b0ab177aee778d0582cc662e2d6590d9af8c81d`
- Standard extension: `1d053c52f7557f7414dda99f5b175500a653026d106f6d4881406f45109b6a69`
- Alternate extension: `cd82d1d41e963de1c20d4ef125f68d5488256038fc3f1d81158b3fb82bd9df3d`
- 9981 validated tests +16 subtests, 3168 alternate acceptance, 577 installed GPU paths,
  135 CLI variants and 96 exact preceding Ordered snapshot recoveries.

Native and standalone scalar Ordered now support simple CTRs with independent
feature banks and matching group/block histories. Native supplies all four
CTR types; standalone supplies Borders/FeatureFreq with Sample/Group histories.
FeatureParallel static CTR penalties persist after selection. Every prefix,
final-estimation cursor, validation state and host chooser recovers exactly.

Sources, both extensions, CLI, original snapshots, scripts and logs are
preserved. Preceding checkpoint 20260913T142818Z remains intact. CUDA parity is incomplete:
Ordered query objectives, greedy ranking objectives, dynamic compound CTRs,
full GPU RNG, estimated features, remaining options and wider hardware/workload
validation remain open. Group-unit Ordered bootstrap is not an upstream GPU
capability; fold growth already follows the upstream float32 option contract.
No CPU CatBoost fitting, NVIDIA comparison or other M-series run occurred.

See ORDERED_CTR_PORT.md for source mappings and limits.

Greedy query checkpoint: Checkpoint `20260913T155047Z` is installed in `catboost/metal/.venv`.

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

See GREEDY_QUERY_PORT.md for source mappings and precision limits.

## Greedy PairLogit installed checkpoint

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

See [GREEDY_PAIRLOGIT_PORT.md](GREEDY_PAIRLOGIT_PORT.md) for source mappings and precision limits.
