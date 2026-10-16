# Ranking metric integration

Standalone QueryRMSE, QuerySoftMax and supplied-pair PairLogit support NDCG,
MAP and PFound as `eval_metric`, including metric parameters, maximizing best
iteration, early stopping and exact snapshot continuation. Native Metal uses
the same query weighting convention. Relevance labels must be present on the
evaluated data; missing PairLogit labels are not invented for ranking metrics.

The shared CatBoost metric implementations evaluate these metrics, as the
pinned CUDA fallback does. GPU fitting and prediction remain Metal operations.
No CPU CatBoost model is trained. Tests compare independent per-query metric
results followed by the source's aggregate, then exercise actual training,
selection and snapshot recovery. Twenty-seven standalone and eighteen native
cases pass; the native tests are also in the coherent full regression run.

## Source weight contract

`cuda/methods/boosting_metric_calcer.h::CacheQueryInfo` takes the first prepared
row weight in each query. For QueryRMSE/QuerySoftMax those weights already
contain object times group weight. Multiplying again would be incorrect.
Supplied-pair targets use unit query weights, independent of pair incidence,
original object weights and group weights. `use_weights=false` uses units.

`libs/metrics/metric.cpp::TMAPKMetric::EvalSingleThread` ignores weights and
increments its denominator once per query in this source revision. MAP thus
retains equal query averaging even when `use_weights=true`; this is preserved
CUDA behavior, not an implemented weighted-MAP extension. NDCG and PFound
consume the query weights described above. Standard CatBoost prediction ties,
relevance thresholds and top/decay/gain parameters are retained.

The native path preserves subgroup IDs for PFound. The standalone ranker's
current array interface does not expose subgroup IDs; subgroup-aware standalone
input remains an integration item. Full ranking objective coverage, live NVIDIA
agreement, and performance parity are still open.
