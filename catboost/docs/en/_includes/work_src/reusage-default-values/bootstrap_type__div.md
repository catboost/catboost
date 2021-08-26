
- When the `objective` parameter is {{ error-function__QueryCrossEntropy }}, {{ error-function__YetiRankPairwise }}, {{ error-function__PairLogitPairwise }} and the `bagging_temperature` parameter is not set: {{ fit__bootstrap-type__Bernoulli }} with the `subsample` parameter set to 0.5
- Not {{ error-function--MultiClass }} and {{ error-function--MultiClassOneVsAll }}, `task_type` = CPU and `sampling_unit` = `Object`: {{ fit__bootstrap-type__MVS }} with the `subsample` parameter set to 0.8.
- Otherwise: {{ fit__bootstrap-type__Bayesian }}.
