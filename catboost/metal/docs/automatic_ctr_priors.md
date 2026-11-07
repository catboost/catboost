# Automatic simple CTR priors

Native Metal training supports `PriorEstimation=BetaPrior` for simple `Borders`
CTRs with one global target border:

```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    task_type="GPU",
    simple_ctr=["Borders:PriorEstimation=BetaPrior"],
    ctr_target_border_count=1,
    one_hot_max_size=2,
    counter_calc_method="SkipTest",
)
model.fit(X, y, cat_features=[0])
```

The adapter uses the shared CUDA
[`TBetaPriorEstimator`](../../cuda/ctrs/prior_estimator.cpp), which is host-only
preprocessing. Its inputs are complete learn-target classes and each original
quantized categorical column, with `OnAll` category cardinality. Observation
weights, group weights, history-unit settings and CTR permutations do not change
these counts. Rows with zero training weight still contribute to prior
estimation. No CPU CatBoost model is fitted.

The estimator performs its existing 50-step beta-binomial optimization, starting
at the global positive-class fraction and using the shared digamma/trigamma
implementation. The fitted prior is stored as two float32 values:
`[alpha, alpha + beta]`. The first is the CTR numerator offset and the second is
the denominator offset. Estimates are resolved before categorical grids and
snapshot identity are prepared, then shared by every permutation and final
model table. Model option metadata records the resolved per-feature priors.

Global simple-CTR defaults are copied only to categorical features without an
explicit per-feature override. As in the CUDA estimator, if any description on
one feature requests automatic estimation, all its eligible `Borders`
descriptions receive the fitted prior, including siblings marked
`PriorEstimation=No`. The original estimation flags remain in option metadata.
Explicit feature indices refer to the original flat feature layout.

Automatic estimation for combinations, `Buckets`, `FloatTargetMeanValue` and
`FeatureFreq` remains unsupported. The source's multi-border branch is retained:
when the actual global target grid has more than one border, configured priors
are left unchanged. If the grid has at most one border but
`ctr_target_border_count` is not 1, an automatic `Borders` request is rejected.
A target containing only one class cannot define a strictly positive Beta
starting point and is rejected.

There is one explicit correction to the inspected CUDA caller's ordering. In
[`cuda/train_lib/train.cpp`](../../cuda/train_lib/train.cpp), both fresh feature
manager paths call `EstimatePriors` before `SetTargetBorders`; the constructor and
`GetTargetBorders` contain no lazy target-grid initialization. That ordering
would give the estimator an empty-border, all-zero class vector. Metal builds
its existing shared global target grid before estimation. This correction uses
the unchanged estimator math and is checked against an independent likelihood
oracle; it does not establish live NVIDIA agreement.

[`test_native_auto_ctr_priors.py`](../tests/test_native_auto_ctr_priors.py)
contains 35 acceptance cases covering an independent SciPy beta-binomial
maximum-likelihood oracle, raw/quantized data, original-row counts and overrides,
model tables/readers, greedy and symmetric training, exact snapshots and
baseline/initial-model behavior. The native build, independent host oracle and
Metal GPU acceptance passed in installed checkpoint `20260914T025201Z`.
