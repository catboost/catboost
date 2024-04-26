## 1. Enum to replace -X/-Y in CLI

CatBoost uses options -X and -Y to do cross-validation training, doc. Your task is to replace these options with one Enum option that can be extended in future with other types of cross-validation.
Proposed new option --cross-validate TYPE[:param[;param...]]

## 2. Allow `eval_period` be any large, just cut it to ensemble size

**Problem**

CatBoost `eval_metrics()` shows as metrics change with ensemble size.
Computing a metric on a dataset is time-consuming, so it is wise to compute the metrics
not for every tree, but with a step. If you selected a reasonably large step and
then decide to reduce the number of trees to use, you may get
an annoying error. Let us fix it.

**How to see the problem**

```py
import catboost
import catboost.datasets
print(catboost.__version__)
from catboost.datasets import amazon
train, test = catboost.datasets.amazon()
train_pool = catboost.Pool(data=train.iloc[:,1:], label=train.iloc[:,0])
test_pool = catboost.Pool(data=test.iloc[:,1:], label=test.iloc[:,0])
model = catboost.CatBoostClassifier(iterations=100)
model.fit(train_pool, eval_set=test_pool)
model.eval_metrics(test_pool, ['AUC', 'Recall'], eval_period=200)  # error here
```

**Solution**

The error comes from
class `TMetricsPlotCalcer` which checks the step against plot size, [here](https://github.com/catboost/catboost/blob/29f49edf05bd7b25d1a3ecb7c892d81155f3f074/catboost/private/libs/algo/plot.h#L121).

We should
make this check always pass or get rid of this check.

Option one:
get rid of this check and instead cut the large step when instantiating the object.

Option two:
keep the check, but make sure the step is cut by the code that instantiates the object.

Add a test

## 3. Add `eval_metrics()` to R package

**Problem**

Python package has such function ((https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_catboost_eval-metrics-docpage here))
Your task is to add this function to R package.

**How to see**

```R
library(catboost)
library(hashr)  # install.packages('hashr')
pool <- catboost.load_pool(iris[-5], sapply(iris[,5], hash))
fit_params <- list(iterations=10, loss_function='MultiClass')
model <- catboost.train(pool, NULL, fit_params)
head(catboost.predict(model, pool))  # can predict, cannot eval metrics
```

## 4. Add model.compare

Currently to compare models visually you need to use a specific widget class. This task is to implement
CatBoost compare method, so that you can call
`model1.compare(model2)` and see the same graphs.

## 5. Flag to ensure raw features data is not copied unnecessarily

CatBoost python package is able to avoid copying (relying on 'views' into the memory of original python objects instead) features data from python objects when passing it for training or prediction if data types and in-memory layout is compatible with CatBoost's C++ library internal representation.
Specifically, it works when numerical features data is represented as numpy.ndarray, catboost.FeaturesData and pandas.DataFrame with `dtype=numpy.float32` and data is continuous and stored column-wise (order='F')).
It is a useful feature to be able to avoid unnecessary memory copying and CPU resources for data format conversion.

See issue <https://github.com/catboost/catboost/issues/549>.

The task is to make a boolean parameter (flag) (could be called `ensure_no_features_copy` for example) to ensure that usage of this feature is enforced: if this flag is set to `True` (the default should be `False`) then all methods that accept features data as parameters must ensure that zero-copy implementation is used or raise CatBoostError exception otherwise. If flag is set to `False` less effective copying cases are allowed. This feature is useful to ensure that optimal performance implementation is used and avoid unexpected performance degradation.

Github issue for this task is <https://github.com/catboost/catboost/issues/601>.

Methods to add this parameter to:
* `Pool` class constructor (see <https://github.com/catboost/catboost/blob/master/catboost/python-package/catboost/core.py>).
    Should be forwarded to `_init_pool` method of `_PoolBase` class in `_catboost.pyx` Cython module (<https://github.com/catboost/catboost/blob/master/catboost/python-package/catboost/_catboost.pyx>).
* Augment logic in _init_pool method of `_PoolBase` class in `_catboost.pyx` Cython module with checking that zero-copy implementation is used (`do_use_raw_data_in_features_order` flag is `True`).
* All methods in <https://github.com/catboost/catboost/blob/master/catboost/python-package/catboost/core.py> that could create Pool objects internally.
    Pass to `Pool` constructor.

## 6. Weight in greedy binarization

Decision tree learning is a greedy algorithm:  on each iteration for each input feature f with N samples algorithm search for best split of points f > c, where c is some condition. CatBoost doesn't use all possible conditions. Instead we select conditions that will be used for each feature before training. We need to select the most representative conditions. As a result we convert numerical feature to discrete one: feature value is replaced with number of conditions that are true for this feature value (e.g. if we have feature 10 and conditions 1, 2, 8, 12 => 10 will be replaced with 3
 The most straightforward way to find conditions is to use quantiles of feature distribution. But this approach can't deal with highly imbalance features (e.g. 99% of ones and 1% of other). To achieve best quality we want to find such condition that after binarization will be of "similar" size. So in library/cpp/grid_creator/binarization.cpp we have several algorithms to deal with this problem.

 CatBoost uses as a default  TMedianInBinBinarizer. This class implements a greedy algorithm. Assume we have feature with values v_1, …, v_n. First, we create set with 2 conditions c_0 = min(v_1, …, v_n) and c_1 = max(v_1, …, v_n). Then on each iteration goes through all current splits c_0 < c_1 … < c_k. We search for conditions c_{i} and c_{i+1} and new condition c, which will maximise next score: log(#{points with v between c_i and c}) + log(#{points v between c and c_{i+1}). Then new condition c is added to set of all condition and algorithm starts next iterations.

Our algorithm currently can't handle weighted samples. Your task will be to add this functionality. For this you'll need to add new function to TMedianInBinBinarizer:

```cpp
THashSet<float> BestSplit(TVector<float>& featureValues,
                          TVector<float>& weights,
                          int maxBordersCount,
                          bool isSorted) const;
```

## 7. Allow `skip_train` `loss_function` property in cv method.

Description:

Sometimes metric calculation seriously affect training time (e.g. if the metric is complicated or if we have a large number of objects in dataset). For speeding up learning we can turn off metric calculation on a train part of the dataset.

Implementation steps:

1. In `libs/train_lib/cross_validation.cpp`
   a. Fill `TVector<bool> skipMetricOnTrain;` with corresponding metric values
   b. In method `CrossValidate` find all places where the metric on train is used (from line 523)
   c. Change function ComputeIterationResults and corresponding method `AppendOneIterationResults` in structure `TCVResult`
2. In `python-package/catboost/_catboost.pyx` in function _cv (line 2645)
   a. Check `results[metric_idx].AverageTrain.size() == 0` and if so add only test to result dict
3. Add test on `skip_train` metric parameter -- in python-package/ut/medium/test.py
   a. Look at example function `test_cv`
   b. Add test `test_cv_skip_train`, with parameters:
   c. `Logloss:skip_train~true,AUC` (`skip_train` by default) and check that there is no metric results on train.

## 8. add options `{input,output}_borders_file` to fit function in CatBoost* in python-package

During training and prediction, CatBoost splits the range of values of each floating point feature into intervals, and uses these intervals instead of the true values.
Using the same feature borders during training and prediction improves prediction accuracy.
Currently python users can neither save feature borders after training, nor load them during prediction; while this is possible with the command-line CatBoost tool.

Your task is to close this gap.

You need to add `input_borders_file` and `output_borders_file` parameters to classes CatBoost* in the CatBoost python package, and pass them via cython to the C++ implementation.

## 9. Per feature one-hot encoding

Your task is to improve CatBoost command-line tool.
Currently CatBoost uses 'one hot' encoding for all categorical features that have a small number of different values specied via option --one-hot-max-size.

You need to implement option --one-hot to list explicitly indexes of categorical features which need one-hot encoding.
1. Add --one-hot option near `--one-hot-max-size`
  * Look in folder catboost/private/libs/options .
  * If both are given, throw TCatBoostException.
2. Find which code handles `--one-hot-max-size` during training, and change this code so that it looks at the features listed after --one-hot
  * Look in `catboost/private/libs/algo/greedy_tensor_search.cpp`.
  * If some feature listed after `--one-hot` has too many values, throw `TCatBoostException`.

## 10. sklearn check classifier

<https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html>

Current `CatBoostClassifier` and `CatBoostRegressor` don't pass this check.
(except for Add `CatBoostClassifier` `predict_log_proba` and `decision_function` methods to support better sklearn API, they are in other issue).

## 11. Implement "Generalized Cross Entropy Loss for Noise-Robust Classifications".

The proposed loss function in the article works well with training neural networks on noisy data.
We want to implement it and check how it is suitable for gradient boosting.

Steps of implementation:
- Work with paper.
   a. Find formula of Lq loss.
   b. Calculate its first, second [and third] derivatives.
- Download epsilon dataset:
   a. train part: <https://proxy.sandbox.yandex-team.ru/785711439> to train.tsv
   b. test part: <https://storage.mds.yandex.net/get-devtools-opensource/250854/epsilon.tar.gz> to test.tsv
- In catboost/private/libs/algo/error_functions.h
   a. Add new class `TNoiseRobustLogloss` inherited from `IDerCalcer`.
   b. Implement constructor and functions
      `ouble CalcDer(double approx, float target)`, `CalcDer2`, `CalcDer3`
- Add function to common set of loss functions:
   a. Add value `TNoiseRobustLogloss` to enum `ELossFunction` in `private/libs/options/enums.h`
   b. Add enum value to `GetAllObjectives`, `IsForCrossEntropyOptimization` functions in `private/libs/options/enum_helpers.cpp`
   c. Add object initialization to `BuildError` function in `private/libs/algo/tensor_search_helpers.cpp`

     ```cpp
     case ELossFunction::NoiseRobustLogloss:
         ...
         return MakeHolder<TNoiseRobustLogloss>(q);
     ```

- Test it:
   a. Compile the code via "ya make -r" in catboost/app
   b. Run usual logloss on epsilon dataset for 1000 iterations.
   c. Run noise robust loss on epsilon dataset with different q.
   d. Compare results
- Optional:
   a. Add random noise to labels and rerun experiments from point 4.

## 12. Model calculation: possibility to write predictions to stdout

Currently, in CLI CatBoost binary it is possible to output predictions only to file, when sometimes it would be more useful to output predictions to stdout.
To implement this functionality we suppose to add special input path scheme with two possible paths:  `stream://stdout` and `stream://stderr` .
In function  CalcModelSingleHost in file
catboost/private/libs/app_helpers/mode_calc_helpers.cpp
add output path schema check and use  `TFileOutput(Duplicate(1))` to create `IOutputStream` compatible stdout stream wrapper.

## 13. get borders from model in python

Float feature borders information is inside `TFloatFeature` structure vector `TObliviousTrees::FloatFeatures` in catboost/libs/model/model.h
We store pointer to `TFullModel` inside our `_CatBoost` cython wrapper class in `catboost/python-package/catboost/_catboost.pyx` so first you need to implement something like `_CatBoost._get_float_features_info()` wich can return simple list of dicts, or, better, python class that mimics `TFloatFeature` and holds all it's fields. Then add pure python method in `catboost/python-package/catboost/core.py` with docstring describing function interface.

## 14. Improvements in documentation

<https://tech.yandex.com/catboost/doc/dg/concepts/about-docpage/>

We know, our docs are not perfect. Please help us make them better. Unfortunatelly now source code of the docs is not on github. So we suggest to write md file and add it to catboost/docs folder.

## 15. Plot model decision tree in CatBoost Python API

<https://github.com/catboost/catboost/issues/355>

This task could be done by implementing special method in `TObliviousTrees` class in `catboost/libs/model/model.h` that will allow to export single tree structure and leafs information.
Then you need to add method-wrapper to `_CatBoost` class in `catboost/python_package/catboost/_catboost.pyx` and place plotting code in  `catboost/python_package/catboost/core.py`

## 16. python predict on single object

Your task is to improve Python CatBoost module.
Currently CatBoost classifiers take a list of objects and return the list of predictions for these objects.
You need to extend the prediction methods so that they accept single object and return the prediction for this object.
CatBoost python module is here: <https://github.com/catboost/catboost/tree/master/catboost/python-package>
CatBoost classes: `CatBoost`, `CatBoostClassifier`, and `CatBoostRegressor`
Prediction methods contain `predict` in their names.

There is an abandoned pull request for this task: <https://github.com/catboost/catboost/pull/559>.

## 17. Add new metrics

Add some metric or loss that you think is missing in CatBoost

## 18. Model calculation is not able to read features from stdin

You need to implement special `ILineDataReader` successor class for scheme and path `stream://stdin` in `catboost/private/libs/data_util/line_data_reader.cpp` and properly register `stream://` scheme for dsv parsing.

## 19. Add CatBoostClassifier `predict_log_proba` and `decision_function` methods to support sklearn API better

## 20. Example of Kaggle GPU kernel in tutorials

Use some large dataset, for example Epsilon (in catboost.datasets).
In kernel run CPU and GPU and compare timings.
Put link to kernel in tutorials repo.

## 21. rename Custom to UserDefined

catboost/private/libs/options/enums.h#L96
Custom -> PythonUserDefinedPerObject

## 22. Support passing feature names in cat_features

CatBoost python package allows specifying categorical features as indices of features columns when creating features dataset from python objects. Sometimes it is also convinient to specify categorical features by column names if they are available (when dataset is passed as `pandas.DataFrame`).
The task is to allow `cat_features` parameter of `Pool` class' constructor and `CatBoost`, `CatBoostClassifier`, `CatBoostRegressor` classes' constructors and `fit` methods to be a sequence of strings.
Then, when `Pool` data is initialized (either by user calling `Pool` constructor explicitly or by creation of `Pool` objects inside the implementation of `fit` methods) it should be checked that passed data is of type `pandas.DataFrame` and categorical features' indices have to be calculated by matching `pandas.DataFrame`'s column labels with strings in `cat_features` parameter, then these calculated categorical features' indices should be used as they are used currently in `Pool` `_init*` functions implementation. In other cases passing `cat_features` as strings should be an error (there is no way to get feature names from just feature values matrix).

Issue: <https://github.com/catboost/catboost/issues/608>

## 23. `use_weights_in_quantization` training parameter

CatBoost quantizes (also called discretization, binning and sometimes binarization) floating point features' values into bins. For some quantization algorithms like `MinEntropy` and `MaxSumLog` it is possible to use objects' (samples') weights but this feature is not currently implemented in CatBoost, but already implemented in quantization library called by CatBoost internally.
The task is to create a training parameter `use_weights_in_quantization` (and add it to CLI parameters and to python and R packages methods that accept training paramters) and if enabled call an appropriate quantization library method.
The case when specified quantization algorithm does not support weights but `use_weights_in_quantization` is enabled should be an error (in the form `CB_ENSURE(false, "Weights are not supported for quantization algorithm ...")`).
It also requires refactoring of quantization library to expose the quantization function that accepts weights as a parameter.

Reference:
* About Quantization in CatBoost: <https://tech.yandex.com/catboost/doc/dg/concepts/binarization-docpage/>
* Quantization options handling: <https://github.com/catboost/catboost/blob/master/catboost/private/libs/options/binarization_options.h>
* Quantization in CatBoost is called from <https://github.com/catboost/catboost/blob/e7d668e5e1fd2f549640fc80dc97598f260e3c4e/catboost/libs/data/quantization.cpp#L179-L183>
* Quantization library is here: <https://github.com/catboost/catboost/tree/master/library/cpp/grid_creator>
* Quantization function that accepts weights is here: <https://github.com/catboost/catboost/blob/e7d668e5e1fd2f549640fc80dc97598f260e3c4e/library/cpp/grid_creator/binarization.cpp#L640>


## 24. Add `SampleId` as a main name for sample id column in a column description file.

CatBoost has [a dsv-based file format](https://tech.yandex.com/catboost/doc/dg/concepts/input-data_values-file-docpage/) with columns' details that could be specified in a separate ['column descriptions' file](https://tech.yandex.com/catboost/doc/dg/concepts/input-data_column-descfile-docpage/).

One of the possible column types is now called `DocId`. It can contain a custom alphanumeric object identifier. The name `DocId` is used for historical reasons because it has been used for cases when samples in a dataset are documents for web search engine ranking. It is more approriate to have a more generally applicable name for this column (while keeping current `DocId` identifier for compatibility).

The task is to rename `DocId` column name to `SampleId` in a column description file (but retain current `DocId` as a synomym for compatibility).

Reference:

* Column types are stored here: <https://github.com/catboost/catboost/blob/5a294552a68367b1ffbbfb2f9e4e805080058e23/catboost/libs/column_description/column.h#L8-L22>. Function `TryFromString` to convert enum `EColumn` to `TString` is generated automatically because of `GENERATE_ENUM_SERIALIZATION` setting in `ya.make` [here](https://github.com/catboost/catboost/blob/5a294552a68367b1ffbbfb2f9e4e805080058e23/catboost/libs/column_description/ya.make#L16)
* Column description file parsing is implemented here: <https://github.com/catboost/catboost/blob/5a294552a68367b1ffbbfb2f9e4e805080058e23/catboost/libs/column_description/cd_parser.cpp#L29>. See how current synonyms are handled [here](https://github.com/catboost/catboost/blob/5a294552a68367b1ffbbfb2f9e4e805080058e23/catboost/libs/column_description/cd_parser.cpp#L66-L71)
* Column name handling to process synonyms should also be added in `eval_result` output columns specification processing in [this file](https://github.com/catboost/catboost/blob/master/catboost/libs/eval_result/eval_result.cpp).


## 25. Pairwise metrics in `eval_metrics` mode.

## 26. Non-informative error when `X_train` and `X_valid` has different number of features

## 27. Improve dataset details in `CreateDataDataProviders`' error messages.

## 28. Allow to specify sample ids as string in pairs input files.

## 29. There should be a warning if number of different class labels is greater than two when binary classification problem is solved.

## 30. Print detailed dataset statistics at the start of training

Example (for binary classification):

```
Number of samples: 400000
Number of features: 2000
Number of positive: 199823, number of negative: 200177
```

## 31. `[python-package]` Implement `__eq__` for `CatBoost*` classes.

Should compare trained models if both objects contain trained models or raise exception otherwise.

## 32. Allow names for all columns in column description file.

CatBoost has [a dsv-based file format](https://tech.yandex.com/catboost/doc/dg/concepts/input-data_values-file-docpage/) with columns' details that could be specified in a separate ['column descriptions' file](https://tech.yandex.com/catboost/doc/dg/concepts/input-data_column-descfile-docpage/).

Names for columns are allowed only for features now but it can be useful sometimes to specify names for other columns as well. Especially for `Auxuliary` columns - some extra data in `eval_result` file might be useful.

The task is to allow names for all columns in column description file and allow to specify them as output columns for `eval_result`.

* Column description file parsing is implemented here: <https://github.com/catboost/catboost/blob/5a294552a68367b1ffbbfb2f9e4e805080058e23/catboost/libs/column_description/cd_parser.cpp#L29>.
* `eval_result` output columns specification processing](https://github.com/catboost/catboost/blob/master/catboost/libs/eval_result/eval_result.cpp).
