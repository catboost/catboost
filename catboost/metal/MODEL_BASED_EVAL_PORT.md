# Model-based feature analysis on Metal

**Status: accepted and installed as checkpoint `20260914T052913Z`.** Full,
alternate, installed, compatibility and host gates passed on the matching frozen
sources. The preceding `20260914T025201Z` wheel and snapshot fixtures remain
preserved. Numerical agreement and scale/performance cards 5 and 6 stay deferred.

The native `catboost model-based-eval` command evaluates feature sets by running
short training experiments from several prefixes of an existing baseline
snapshot. It writes learn and validation metric histories for each feature set
and prefix. All new tree fitting uses Metal. Prefix reconstruction applies saved
trees to the original quantized data without fitting a CPU reference model.

This is the existing CLI analysis interface. There is no added public Python
`CatBoost.model_based_eval()` method. The result is a collection of metric logs;
the command does not export trial models or a ranked feature-importance report.

## Supported scope

Analysis requires **Plain boosting with DocParallel data partitioning**, an
existing compatible nonempty Metal snapshot, exactly one nonempty evaluation
pool, and writable output. Normal Metal objective, leaf-estimation, score,
bootstrap, depth and grow-policy restrictions still apply.

| Objective family | Analysis grow policies |
| --- | --- |
| RMSE, Logloss, CrossEntropy, Poisson, Huber, Expectile, Tweedie, LogLinQuantile, Quantile, MAE, MAPE | SymmetricTree, Depthwise, Lossguide, Region |
| Lq | SymmetricTree |
| QueryRMSE, QuerySoftMax, PairLogit, classic YetiRank | SymmetricTree, Depthwise, Lossguide, Region |
| PairLogitPairwise, QueryCrossEntropy, YetiRankPairwise | SymmetricTree |
| MultiClass, MultiClassOneVsAll, RMSEWithUncertainty | SymmetricTree, Depthwise, Lossguide, Region |
| Supported scalar/query Combination objectives | SymmetricTree |

Ordered and FeatureParallel analysis are rejected. Multitarget objectives such
as MultiRMSE, MultiLogloss and MultiCrossEntropy, custom objective descriptors,
and active text or embedding estimators are also rejected. Compound CTRs require
FeatureParallel and are outside this command's scope.

Evaluated candidates are **numeric features with quantization borders**.
Categorical features may remain active background inputs, including supported
one-hot and simple-CTR features. Keep those background columns outside the
evaluated feature lists. Constant or non-numeric candidate entries produce a
warning; a set with no eligible numeric candidate is skipped. Later sets retain
their original indices, so output directory numbering can contain gaps.

Feature indices are zero-based external feature indices, excluding Target,
Weight, GroupId and Baseline columns. Names come from the column description or
feature-names file. For `--features-to-evaluate`, commas combine features within a
set, semicolons separate sets, and ranges are inclusive. For example,
`'1,2;4-6'` specifies two sets. Quote these expressions. The separate
`--ignore-features` option uses colons, for example `'1:2'`.

## Author a baseline and run analysis

Run these Bash/Zsh commands from the repository root after selecting the rebuilt
CLI. The data directory must contain `train.tsv`, `test.tsv` and `columns.cd`.
Features 1 and 2 are the candidates; feature 0 remains available as background.
The [example fixture](examples/model_based_eval_acceptance/README.md) creates
suitable numeric files without fitting a model.

```sh
model_eval_cli="$PWD/catboost/metal/.build/releases/20260914T052913Z/catboost"
model_eval_data=/absolute/path/to/data
model_eval_run=/absolute/path/to/a/fresh/analysis-run

model_eval_options=(
  -f "$model_eval_data/train.tsv" -t "$model_eval_data/test.tsv"
  --cd "$model_eval_data/columns.cd"
  --task-type GPU --loss-function RMSE
  --boosting-type Plain --data-partition DocParallel
  --grow-policy SymmetricTree -i 8 --depth 3
  --learning-rate 0.2 --l2-leaf-reg 2
  --leaf-estimation-method Newton --leaf-estimation-iterations 1
  --leaf-estimation-backtracking No --score-function Cosine
  --boost-from-average false --use-best-model false
  --bootstrap-type No --random-strength 0 --random-seed 619
  --permutations 1 --has-time --border-count 16 --thread-count 2
  --one-hot-max-size 2 --max-ctr-complexity 1 --model-size-reg 0
  --metric-period 1 --logging-level Silent --allow-writing-files true
  --learn-err-log learn_error.tsv --test-err-log test_error.tsv
)

"$model_eval_cli" fit "${model_eval_options[@]}" \
  --train-dir "$model_eval_run/baseline" --ignore-features '1:2' \
  --snapshot-file baseline.snapshot --snapshot-interval 0 \
  --model-file "$model_eval_run/baseline/baseline.cbm"

"$model_eval_cli" model-based-eval "${model_eval_options[@]}" \
  --train-dir "$model_eval_run/analysis" \
  --baseline-model-snapshot "$model_eval_run/baseline/baseline.snapshot" \
  --features-to-evaluate '1;2;1-2' \
  --offset 4 --experiment-count 2 --experiment-size 2
```

The default policy expects the baseline to exclude the **union** of every
evaluated set. Here, baseline fitting ignores features 1 and 2, while analysis
receives the original full data and does not carry that union into
`--ignore-features`. Existing unrelated ignored features must remain consistent
between baseline authoring and analysis. An explicitly ignored feature cannot
also be evaluated.

For the alternative policy, author a separate baseline with the candidate
features included: remove `--ignore-features '1:2'` from the baseline fit, then
add `--use-evaluated-features-in-baseline-model` to its analysis command. That
switch takes **no value**. Under either policy, each trial uses background
features plus its current candidate set; other evaluated candidates remain
excluded from that trial's new trees.

Use an absolute `--baseline-model-snapshot` path. A relative path resolves under
the analysis train directory. A `.cbm` model cannot replace the training
snapshot: reconstruction also needs compatible data, parameters and permutation
history. Reuse the baseline's learn/evaluation data and training settings.

## Prefix scheduling and trial state

Let `T` be the retained baseline tree count, `O` the offset, `C` the experiment
count, and `S` the experiment size. Require positive `O`, `C` and `S`, with
`C × S ≤ O ≤ T`. Trial `k`, starting at zero, uses the baseline prefix

```text
start(k) = T - O + floor(O / C) × k,   0 ≤ k < C
```

Integer division is intentional; the offset need not divide evenly by the
count. For `T=8, O=5, C=2, S=2`, the starts are 3 and 5. The example above uses
starts 4 and 6. These “folds” share the supplied learn/evaluation pools and vary
the baseline prefix; they are not a fresh data cross-validation split.

When `--use-best-model true` is selected, `T` follows the retained best prefix,
including the configured minimum best-model tree count. The completed snapshot
can contain more trees than `T`; the offset must fit within the retained prefix.

Each feature set begins from independent copies of the baseline cursors. Every
trial restores each training permutation's prefix predictions and the evaluation
cursor. Its new tree count, metric history, optimizer and bootstrap state start
at local iteration zero. Permutation selection uses the baseline-derived
iteration seed shifted by `start(k)`, matching the upstream scheduling rule.
This is distinct from ordinary snapshot continuation, which restores training
age and optimizer state.

The existing Metal random-state helpers currently restart for each experiment.
CUDA keeps its shared generator alive across the experiment sequence. Carrying
that state between experiments remains part of the deferred card 5 RNG work;
matching options do not establish matching stochastic experiment histories.
The baseline offset above applies only to permutation selection, not to local
bootstrap or score-model age.

Prefix replay adds the float32 leaf increments stored in the snapshot. Existing
vector training can fuse the learning-rate multiplication and cursor addition,
so its live cursor need not exactly equal replay of the rounded stored leaves.
A diagnostic RMSEWithUncertainty case exposed a one-ULP cursor difference that
could change a tied CTR split. A continuous-training metric curve is therefore
not always an exact oracle for these experiments; acceptance reconstructs the
rounded prefix independently. This port leaves the existing training arithmetic
unchanged, with broader numerical agreement deferred to card 5.

Metrics are forced to period 1. With early stopping disabled, each completed
trial writes `iter=0..S-1` in:

```text
analysis/feature_set0_fold0/learn_error.tsv
analysis/feature_set0_fold0/test_error.tsv
analysis/feature_set0_fold1/...
analysis/feature_set1_fold0/...
```

Standard JSON/time logs also use each trial directory. Experiments do not write
new snapshots or trial `.cbm` files, and analysis does not rewrite the baseline
snapshot. Interruption or an output failure can leave partial trial logs; retain
them as partial evidence and use a fresh output directory for the next attempt.

## Snapshot history and compatibility

Fresh supported DocParallel snapshot training with this implementation records
the leaf values needed to reconstruct every permutation at every completed tree.
No additional authoring flag is required beyond ordinary snapshot saving.

The optional trailing record uses tag `0x4D4D4231` and stores three `uint32`
dimensions (permutation count, padded leaf capacity, approximation dimension),
then a `uint64` value count and float32 leaf values. The layout is:

```text
[completed tree][permutation][leaf padded to capacity][approximation dimension]
```

The history capacity matches the runtime's reachable leaf capacity, bounded by
the configured depth. For example, Lossguide with `max_leaves=31` and `depth=2`
stores four leaf slots per permutation and tree in this record, even when an
individual tree uses fewer leaves. The ordinary snapshot still reserves its
existing requested capacity of 31; its layout and compatibility contract are
unchanged. The optional history record carries its own capacity and must not
inherit the ordinary snapshot's padding width.

It follows the existing optional best-learn cursor and, when enabled, Langevin
records. Loading validates ordering, dimensions, exact lengths, finite values
including padding, truncation and trailing data. Permutation count and
approximation dimension are bounded to 1..64; leaf capacity is bounded to
1..65,536. The aggregate history payload is limited to **512 MiB**:
`4 × trees × permutations × leaf_capacity × dimensions` bytes. The reader checks
the declared length and aggregate limit before allocating that array. Existing
learner workspace and objective-specific resource limits also apply.

An older snapshot without this record remains valid for ordinary resume.
Single-permutation analysis can reconstruct prefixes from its ordinary saved
trees. Analysis needing multiple permutations rejects an older snapshot without
the history and requests a **fresh baseline snapshot authored by this build**.
Resuming an old snapshot does not invent the missing earlier per-permutation
leaves. The ordinary resume contract and the 350 preserved compatibility
fixtures remain separate release checks.

## Source mapping and acceptance

| Responsibility | Source |
| --- | --- |
| Upstream baseline policies, prefix scheduling and permutation-seed shift | [CUDA DocParallel boosting](../cuda/methods/doc_parallel_boosting.h), `RunModelBasedEval` |
| CLI dispatch, feature-name conversion and forced metric period | [CLI entrypoint](../app/mode_model_based_eval.cpp), [option bindings](../private/libs/app_helpers/bind_options.cpp), [analysis options](../private/libs/options/model_based_eval_options.cpp) |
| Metal pool subsets, baseline validation, prefix replay and experiment execution | [Native training adapter](train_lib/train.cpp), `TMetalModelBasedRun` and `ModelBasedEval` |
| Per-permutation training state and iteration selection | [Permutation adapter](train_lib/permutations.h) |
| Initial metric/evaluation cursors and local trial histories | [Progress interface](train_lib/progress.h), [implementation](train_lib/progress.cpp) |
| Last-tree leaves for all permutations and dimensions | [Scalar runtime](native/metal_trainer.mm), [greedy runtime](native/metal_greedy_trainer.mm), [vector runtime](native/metal_multiclass.mm), corresponding native headers |
| Optional record, compatibility and resource validation | [Snapshot format](train_lib/snapshot.h), [snapshot-tail unit tests](train_lib/model_based_snapshot_ut.cpp) |
| CLI analysis, independent prefix checks and rejection/artifact coverage | [Native model-based-eval tests](tests/test_native_model_based_eval.py) |

The final matrix passed after correcting Lossguide history capacity and tests
that assumed EOF after older snapshot records. Diagnostic runs are preserved as
development evidence and are not added to the final totals.

| Acceptance gate | Passed |
| --- | ---: |
| Full native/standalone matrix | 14,551 tests plus 16 subtests |
| Alternate extension | 6,187 tests |
| Installed package | 2,606 tests |
| Exact preceding snapshot recoveries | 350 |
| CLI configurations | 348 |
| Preinstall GPU smoke configurations | 203 |
| Installed GPU smoke configurations | 203 |
| Metal C++ helper checks | 135 |
| Estimated/CTR metadata C++ checks | 3 |
| Quantized categorical apply C++ checks | 3 |
| Combination metric C++ checks | 2 |

All gates have zero failures, errors and skips. Selections overlap and must not
be summed. Full, alternate and installed matrices each contain 68 model-based
CLI cases and one contract case. Full and installed additionally contain 106
per-permutation leaf-accessor cases: 24 scalar, 38 greedy and 44 vector. The
17 snapshot-history cases are included in the 135 Metal C++ checks.

Coverage includes both baseline policies, multiple and overlapping feature sets,
named/range inputs, nonzero and nondivisible prefix starts, best-model truncation,
scalar/query/vector routes, categorical background history, legacy snapshots,
skipped constant sets, output failures and interruption. Full, alternate and
installed selections used the same frozen sources and explicitly selected CLI;
the installed extension matches the standard tested binary. The preserved 350
preceding snapshots are checked against original expected results.

Release directory: `catboost/metal/.build/releases/20260914T052913Z/`.
The installed environment is `catboost/metal/.venv`. The release uses 764 frozen
source identities and 11 added-file overlays against source base
`c79c4b96499834457fc06e6819d149611dbd8d6a`; it preserves the source patch,
reconstruction material, both extensions, CLI and acceptance records.

| Artifact | SHA256 |
| --- | --- |
| `catboost-1.2.10-cp312-cp312-macosx_11_0_arm64.whl` | `1cbce1e7b7ded2e7b8f7d4db0c4d850ba47904ef52df632e5ce8ec557335c88e` |
| Standard/installed extension | `9b8814a605e501809840ff0ba9b7e4a9db6cd54acec6f321a9619294345e9073` |
| Alternate extension | `16bc99df3716aec5e8f79e1633d2bc55f32c4c53fc9a6dd0713cae2e3c670d5a` |
| CLI | `915e86c148ad9b13df5a3a8961775f4d04b1c10279aa89562b1eec6833b5696b` |

Card 5 numerical/RNG agreement work remains deferred. This port follows the
upstream algorithm and interface sources, but no NVIDIA execution or
cross-device numerical-parity claim is part of the current evidence.
