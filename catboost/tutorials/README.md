# Tutorials

## Python tutorials

* Main CatBoost tutorial with base features demonstration:
    * __kaggle_titanic_catboost_demo.ipynb__
        * This tutorial shows how to train a model on Titanic kaggle dataset. Examples of cross validation, parameter tuning and overfitting detection are provided.
    * __catboost_base_functions.ipynb__
        * This tutorial shows different usages of CatBoost including training with custom error function, using different classes for training, using weights and others.

* CatBoost performance at different competitions:
    * __kaggle_paribas.ipynb__
        * This tutorial shows how to get to a 9th place on paribas competition with only few lines of code and training a CatBoost model.

    * __mlbootcamp_v_tutorial.ipynb__
        * This is an actual 7th place solution by Mikhail Pershin. Solution is very simple and is based on CatBoost.

* CatBoost and TensorFlow:
    * __quora_catboost_w2v.ipynb__
        * This tutorial shows how to use CatBoost together with TensorFlow if you have text as input data.

* CatBoost and CoreML:
    * __catboost_coreml_export_tutorial.ipynb__
        * This tutorial shows how to convert CatBoost model to CoreML format and use it on an iPhone.

## R tutorials

* Main CatBoost tutorial with base features demonstration:
    * __catboost\_r\_tutorial.ipynb__
        * This tutorial shows how to convert your data to CatBoost Pool, how to train a model and how to make cross validation and parameter tunning.

## Command line tutorials

* Main CatBoost tutorial with base features demonstration:
    * __catboost\_cmdline\_tutorial.md__
        * This tutorial shows how to train and apply model with the command line tool.

## Adding custom per-object error function tutorial

All you need is to implement several methods:

* Methods  `CalcDer` and  `CalcDer2` of the class  `TUserDefinedPerObjectError`.
    These methods have the following parameters:
    * `approx` - is the value of the target function for the object.
    * `target` - is the target for the object.

* Method  `Eval` of the class  `TUserDefinedPerObjectMetric`.
    These method has the following parameters:
    * `approx` - is the vector of values of the target function for objects.
    * `target` - is the vector of objects targets.
    * `weight` - is the vector of objects weights.
    * `begin` and  `end` - The metric should be calculated for objects from the range  `[begin, end)` .

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

TMetricHolder TUserDefinedPerObjectMetric::Eval(const TVector<TVector<double>>& approx,
                                   const TVector<float>& target,
                                   const TVector<float>& weight,
                                   int begin, int end,
                                   NPar::TLocalExecutor& /* executor */) const {
    TMetricHolder error;
    for (int k = begin; k < end; ++k) {
        float w = weight.empty() ? 1 : weight[k];
        error.Error += w * (log(1 + exp(approx[0][k])) - target[k] * approx[0][k]);
        error.Weight += w;
    }
    return error;
}
```
