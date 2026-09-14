# CatBoost on Apple Metal

This work-in-progress backend translates CatBoost's CUDA algorithms to Apple
Metal. Training and tree evaluation run on the Apple GPU. Shared CatBoost code
handles data preparation, metrics, and standard model files.

There are two entry points:

- **Native CatBoost:** build this fork, then use ordinary `CatBoostRegressor`,
  `CatBoostClassifier`, `Pool`, and `task_type="GPU"`. Darwin ARM64 registers
  the GPU trainer as Metal. The CLI and Python extensions share the backend.
- **Standalone adapter:** `CatBoostMetalRegressor`,
  `CatBoostMetalClassifier`, and `CatBoostMetalRanker` call the runtime directly alongside an installed
  CatBoost wheel. This is convenient for kernel development and has a narrower
  data/API surface than the native trainer.

Installing an upstream CatBoost wheel alone does not install this Metal port.
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) tracks tested capabilities,
remaining CUDA gaps, and native versus standalone evidence. Installed checkpoint
`20260914T000929Z` includes greedy YetiRank, Ordered query/ranking, native
Plain/Ordered FeatureParallel compound training for the registered scalar/query
objectives, Combination losses and custom per-object Metal shaders. Its full
matrix passes 12,745 tests plus 16 subtests; the alternate extension passes
4,901 tests. See [TRAINING_MODES_PORT.md](TRAINING_MODES_PORT.md) for the supported
matrix, source corrections, exact lifecycle and final release evidence.

## Run the standalone adapter

Use an Apple Silicon Mac, Python 3.12, and the Xcode Command Line Tools. From
the repository root:

```sh
python3.12 -m venv catboost/metal/.venv
source catboost/metal/.venv/bin/activate
python -m pip install -r catboost/metal/requirements.txt
export PYTHONPATH="$PWD/catboost/metal/python"
```

```python
import numpy as np
from catboost_metal import CatBoostMetalRegressor

rng = np.random.default_rng(42)
X = rng.normal(size=(4096, 6)).astype(np.float32)
y = (2 * X[:, 0] - X[:, 1] + X[:, 2] ** 2).astype(np.float32)
model = CatBoostMetalRegressor(
    iterations=100, depth=4, learning_rate=0.1,
    bootstrap_type="Bernoulli", subsample=0.8,
    random_seed=42, nan_mode="Min",
)
model.fit(X[:3072], y[:3072], eval_set=(X[3072:], y[3072:]),
          early_stopping_rounds=15, use_best_model=True)
print(model.predict(X[:5], task_type="METAL"))
model.save_model("metal_model.cbm")
model.save_model("metal_model.json", format="json")
```

The adapter compiles Objective-C++ with `xcrun clang++` and embedded shaders
through the Metal API. It does not require the offline `metal` compiler.
Quantization and model loading use the installed CatBoost package; training
never falls back to its CPU trainer. Prediction defaults to the standard CPU
evaluator; choose `task_type="METAL"` or `"GPU"` for Metal tree evaluation.

Classification supports Logloss, soft-target CrossEntropy, MultiClass,
MultiClassOneVsAll, MultiLogloss, and MultiCrossEntropy. Vector regression
supports MultiRMSE and RMSEWithUncertainty. Scalar objectives also include RMSE, Poisson, Huber,
Expectile, Lq, Tweedie, LogLinQuantile, Quantile, MAE, and MAPE, with their
objective-specific parameters and estimation restrictions. Newton/Gradient
leaf estimation and No/AnyImprovement/Armijo backtracking are implemented;
Quantile/MAE/MAPE also support Exact estimation. Symmetric training supports
depths through 16. Inputs include observation/class weights, numeric NaNs,
one-hot categories, and simple CTRs. Native Pool preparation exposes the
widest categorical and prequantized-data support.

Scalar and the registered query/ranking objectives support
`boosting_type="Ordered"`; Plain training supports `grow_policy="Depthwise"`,
`"Lossguide"`, or `"Region"`.
Lossguide growth is bounded by `max_leaves`; Region grows a chain with at most
`depth + 1` leaves. Eleven CUDA-registered scalar objectives support these
policies, with all seven structure scores, No/Bayesian/Bernoulli/Poisson
sampling, score noise, and the applicable leaf estimators/backtracking.

`CatBoostMetalRanker` supports QueryRMSE, QuerySoftMax, supplied-pair
PairLogit/PairLogitPairwise, QueryCrossEntropy, classic YetiRank and
YetiRankPairwise. QueryRMSE, QuerySoftMax, PairLogit and classic YetiRank also
support Ordered training and Plain Depthwise/Lossguide/Region. These routes
retain their objective-specific numeric, one-hot and simple CTR surface, with
PFound evaluation and saved target RNG for classic YetiRank. Full-matrix
PairLogitPairwise, QueryCrossEntropy and YetiRankPairwise remain symmetric Plain
DocParallel. See [TRAINING_MODES_PORT.md](TRAINING_MODES_PORT.md). Pass contiguous
`group_id` values to `fit`; PairLogit additionally
accepts `pairs` and `pairs_weight`. NDCG, MAP, and PFound can select the best
iteration and drive early stopping, with the CUDA query-weight conventions
documented in [RANKING_METRICS.md](RANKING_METRICS.md). Native
`CatBoostRanker(task_type="GPU", loss_function="PairLogitPairwise")` now supports
numeric full-matrix search and leaves, depths through eight, edge sampling,
backtracking and snapshots. The standalone ranker supports that same numeric
trainer with supplied pairs and original document weights; see
[PAIRWISE_MATRIX_PORT.md](PAIRWISE_MATRIX_PORT.md). Native and standalone
QueryCrossEntropy now use a full query matrix, Newton leaves, whole-query
No/Bernoulli sampling and scaled GPU metrics. Queries are limited to 256 rows
and depth to eight. See [QUERY_CROSS_ENTROPY_PORT.md](QUERY_CROSS_ENTROPY_PORT.md). See [QUERYWISE_PORT.md](QUERYWISE_PORT.md) and
[PAIRWISE_PORT.md](PAIRWISE_PORT.md) for exact weighting and sampling semantics.

Snapshots preserve raw predictions, absolute random iteration, and applicable
bootstrap/permutation state. CUDA and Metal support MVS for the scalar path;
multiclass supports No/Bayesian/Bernoulli/Poisson. Use
`fit(..., save_snapshot=True, snapshot_file="training.snapshot")`;
a later estimator with the same data/options and a larger total `iterations`
resumes that checkpoint.

## Build native CatBoost with Metal

`HAVE_METAL` defaults ON for Darwin ARM64; `HAVE_METAL=OFF` disables it.
Xcode/Clang and the macOS SDK are required. Existing Conan dependencies are
fetched by the build. Metal execution requires macOS 13 or newer, although the
wheel retains the project's macOS 11 tag for its CPU functionality.

This reproduces the locally verified build flow. Start from the repository
root with the Python environment activated:

```sh
python -m pip install 'cmake>=3.24,<4.0' ninja 'conan>=2.4.1,<3.0' \
  'cython~=3.0.10' 'numpy<3.0' setuptools wheel
export CATBOOST_METAL_BUILD_ROOT="$PWD/catboost/metal/.build/native"
python build/build_native.py \
  --build-root-dir "$CATBOOST_METAL_BUILD_ROOT" \
  --targets catboost,_catboost \
  --parallel-build-jobs 4 \
  --cmake-extra-args="-DPython3_ROOT_DIR=$VIRTUAL_ENV,-DUSE_INTERNAL_CYTHON=OFF"
```

Package the built extension:

```sh
cd catboost/python-package
python setup.py build \
  --build-base "$CATBOOST_METAL_BUILD_ROOT/wheel-build" --no-widget \
  --prebuilt-extensions-build-root-dir "$CATBOOST_METAL_BUILD_ROOT" \
  bdist_wheel --no-widget \
  --prebuilt-extensions-build-root-dir "$CATBOOST_METAL_BUILD_ROOT" \
  --dist-dir "$CATBOOST_METAL_BUILD_ROOT/dist" \
  --bdist-dir "$CATBOOST_METAL_BUILD_ROOT/wheel-install" \
  --plat-name macosx_11_0_arm64
```

Install the resulting fork wheel in the environment that will run native
CatBoost. It contains the runtime and embedded shaders; a host compiler is not
needed when using that wheel.

```python
from catboost import CatBoostRegressor, Pool

model = CatBoostRegressor(
    task_type="GPU", iterations=100, depth=4,
    boosting_type="Plain", data_partition="DocParallel",
    bootstrap_type="No", random_strength=0,
    leaf_estimation_backtracking="No", verbose=False,
)
model.fit(Pool(X, y))
assert model.get_metadata()["metal_backend"] == "METAL"
```

The native factory uses the existing GPU task type, not a third METAL enum.
Native integration and standalone options are tested separately. Native symmetric
training also accepts supported `Combination` losses and custom per-object GPU
objectives through `calc_ders_range_metal()`. The
[custom shader interface](docs/custom_objectives.md) defines the weighted
value/gradient/curvature contract and supplies an example. These two interfaces,
Plain FeatureParallel and dynamic compound CTR generation are native capabilities;
the standalone estimator frontend does not expose them.

## Validation and examples

From the repository root with the standalone package on `PYTHONPATH`:

```sh
python -m pytest catboost/metal/tests
python catboost/metal/examples/train_regression.py
python catboost/metal/examples/train_datasets.py
```

Native acceptance tests opt in so an unrelated installed wheel cannot be
mistaken for this fork:

```sh
CATBOOST_NATIVE_METAL_TESTS=1 python -m pytest \
  catboost/metal/tests/test_native_api.py \
  catboost/metal/tests/test_native_multiclass.py \
  catboost/metal/tests/test_native_categorical_pool.py \
  catboost/metal/tests/test_native_evaluator.py
```

That environment must contain the rebuilt native wheel, or its isolated
package must be first on `PYTHONPATH`. Default standalone runs skip native-only
checks. Examples save models, source provenance, and measurements under
ignored `.build/` directories. No NVIDIA execution or M1/M2/M4 hardware
comparison has been performed.

## Architecture and limits

Persistent sessions retain GPU buffers across trees. Stable incremental row
partitions feed banked threadgroup histograms; smaller-child computation and
sibling subtraction reuse parent statistics. GPU reductions select split
winners. Compensated partition/leaf and prediction sums improve accuracy,
while mixed-precision CUDA parity remains unproven for close scores.

The installed native wheel supports Plain DocParallel symmetric trees through
depth 16, twelve scalar losses, six vector families and all seven connected
grouped objectives. Symmetric Plain and Ordered FeatureParallel support scalar
losses, QueryRMSE, QuerySoftMax, PairLogit, classic YetiRank, Combination and
custom per-object shaders. Numeric/one-hot/simple CTR Depthwise, Lossguide and
Region cover the CUDA-registered scalar/vector/query objectives, now including
classic YetiRank. Working buffers are capped at 1 GiB, outputs at 512 MiB and
rows at 2²⁴; numeric features support up to 255 borders. These are software caps.

Native and standalone Ordered retain complete prefix cursors and whole-group
histories for the supported scalar/query/ranking objectives. Public
Ordered+Exact remains rejected like CUDA. The new Simple query/Combination/custom
leaf methods are native symmetric capabilities: DocParallel exports sampled weak
statistics, while FeatureParallel performs one Gradient-style leaf step.
Greedy Simple remains unsupported. See
[TRAINING_MODES_PORT.md](TRAINING_MODES_PORT.md) for leaf, score, sampler and
seed-order restrictions.

Native symmetric Plain/Ordered FeatureParallel connects compound CTRs to the
registered scalar/query/ranking objectives and Combination/custom targets, with
Sample/Group histories, retained P1/P4 grids, exact snapshots and standard model
tables. Set `max_ctr_complexity=2` or `3`; see
[COMPOUND_CTR_PORT.md](COMPOUND_CTR_PORT.md) for the categorical machinery and its
preceding checkpoint, and [TRAINING_MODES_PORT.md](TRAINING_MODES_PORT.md) for
current objective coverage. Full-matrix, vector and greedy training do not use
that compound scheduler. Standalone Plain FeatureParallel and compound CTR
frontends, remaining APIs/options, text/embeddings, CUDA device RNG agreement,
numerical ties, memory scaling and wider hardware/performance validation remain
open. No full CUDA feature, numerical or performance parity is claimed.
The [source map](CUDA_PORT.md) explains correspondence and precision differences;
[BOOTSTRAP_PORT.md](BOOTSTRAP_PORT.md) records sampling semantics.
[PORT_REVIEW.md](PORT_REVIEW.md) preserves the earlier dated review; use
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for current status.

Native and standalone greedy training accept one-hot categories for all
eleven registered scalar losses; see [GREEDY_ONE_HOT_PORT.md](GREEDY_ONE_HOT_PORT.md).
All seven connected grouped objectives accept native simple CTR P4, including
YetiRankPairwise; its explicit Metal dataset streams remain distinct from
complete CUDA GPU random-buffer consumption.

Native greedy training now accepts all four simple CTR types with P1/P4
dataset cursors and exact snapshots; see [GREEDY_CTR_PORT.md](GREEDY_CTR_PORT.md).
Standalone scalar greedy estimators also accept Borders/FeatureFreq CTRs.

Standalone greedy estimators now accept Borders/FeatureFreq CTRs with P1–P64
and complete snapshot recovery; see [STANDALONE_GREEDY_CTR_PORT.md](STANDALONE_GREEDY_CTR_PORT.md).

Native `task_type="GPU"` now supports MultiClass, MultiClassOneVsAll and
RMSEWithUncertainty with Depthwise/Lossguide/Region, including categories and
exact recovery. Standalone estimators now support these same three vector greedy
objectives, numeric/one-hot/CTR inputs, GPU evaluation and all-permutation
optimizer snapshots. See [VECTOR_GREEDY_PORT.md](VECTOR_GREEDY_PORT.md).

Host validation allocation overhead has been removed while retaining input and
overflow checks. Measured evaluator and native fit results are recorded in
[VALIDATION_FASTPATH.md](VALIDATION_FASTPATH.md); both extensions and all CLI
acceptance paths have been rebuilt and verified.

Scalar Ordered simple CTR training is connected through native and standalone APIs.
[ORDERED_CTR_PORT.md](ORDERED_CTR_PORT.md) records per-history feature banks,
Sample/Group histories, static FeatureParallel penalties and exact recovery.

QueryRMSE/QuerySoftMax now also support Depthwise, Lossguide and Region in
native and standalone APIs. [GREEDY_QUERY_PORT.md](GREEDY_QUERY_PORT.md)
records whole-query projection, CTR dataset cursors and exact recovery.

Greedy PairLogit now supports Depthwise, Lossguide and Region through native
CatBoost and standalone Metal APIs. Native one-hot/simple CTR permutations,
unlabeled supplied-pair Pools, metrics, model readers and exact snapshot
continuation are accepted. [GREEDY_PAIRLOGIT_PORT.md](GREEDY_PAIRLOGIT_PORT.md)
records the installed release, numerical fix and full validation evidence.

Classic YetiRank now joins those greedy policies in both interfaces, and
QueryRMSE/QuerySoftMax/PairLogit/YetiRank connect Ordered query histories and
native Plain FeatureParallel training. Native Combination and custom shaders
retain baselines, initial models, callbacks, metrics, exact snapshots and ordinary
CBM/JSON export. [TRAINING_MODES_PORT.md](TRAINING_MODES_PORT.md) records the
accepted matrix and the explicit corrections to pinned CUDA source defects.
