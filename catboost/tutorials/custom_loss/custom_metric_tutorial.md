# Adding custom per-object metric tutorial

If you want to add a metric to observe, to use overfitting detector or to choose best model,
all you need is to implement method `Eval` of the class `TUserDefinedPerObjectMetric`.
These method has the following parameters:
    * `approx` - is the vector of values of the target function for objects.
    * `target` - is the vector of objects targets.
    * `weight` - is the vector of objects weights.
    * `queriesInfo` - is the vector of queries information. You should not use it if you implement PerObjectMetric.
    * `begin` and `end` - The metric should be calculated for objects from the range `[begin, end)`.
    * `executor` - is the element of class for parallelizing calculations. You may not use it.

And then set the parameter `eval_metric` with the value of `UserPerObjMetric`.

Example of the Logloss implementation:
```
TMetricHolder TUserDefinedPerObjectMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end,
    NPar::TLocalExecutor& /*executor*/
) const {
    TMetricHolder error;
    for (int k = begin; k < end; ++k) {
        float w = weight.empty() ? 1 : weight[k];
        error.Error += w * (log(1 + exp(approx[0][k])) - target[k] * approx[0][k]);
        error.Weight += w;
    }
    return error;
}
```

# Adding custom per-object objective function tutorial

If you want to add a metric to optimize it, all you need is to implement
methods `CalcDer` and `CalcDer2` of the class `TUserDefinedPerObjectError`:
These methods have the following parameters:
    * `approx` - is the value of the target function for the object.
    * `target` - is the target for the object.

And then set the parameter `loss_function` with the value of `UserPerObjMetric`.

Example of the Logloss implementation:
```
double CalcDer(double approx, float target) const {
    double approxExp = exp(approx);
    const double p = approxExp / (1 + approxExp);
    return target - p;
}

double CalcDer2(double approx, float /*target*/) const {
    double approxExp = exp(approx);
    const double p = approxExp / (1 + approxExp);
    return -p * (1 - p);
}
```
