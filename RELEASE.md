# Release 0.23.1

## New functionality
* CatBoost model could be simply converted into ONNX object in Python with `catboost.utils.convert_to_onnx_object` method. Implemented by @monkey0head
* We now print metric options with metric names as metric description in error logs by default. This allows you to distinguish between metrics of the same type with different parameters. For example, if user sets weigheted average `TotalF1` metric CatBoost will print `TotalF1:average=Weighted` as corresponding metric column header in error logs. Implemented by @ivanychev
* Implemented PRAUC metric (issue  #737). Thanks @azikmsu
* It's now possible to write custom multiregression objective in Python. Thanks @azikmsu
* Supported nonsymmetric models export to PMML
* `class_weights` parameter accepts dictionary with class name to class weight mapping
* Added `_get_tags()` method for compatibility with sklearn (issue #1282). Implemented by @crazyleg
* Lot's of improvements in .Net CatBoost library: implemented IDisposable interface, splitted ML.NET compatible and basic prediction classes in separate libraries, added base UNIX compatibility, supported GPU model evaluation, fixed tests. Thanks @khanova
* In addition to first_feature_use_penalties presented in the previous release, we added new option per_object_feature_penalties which considers feature usage on each object individually. For more details refer the [tutorial](https://github.com/catboost/catboost/blob/master/catboost/tutorials/feature_penalties/feature_penalties.ipynb).

## Breaking changes
* From now on we require explicit `loss_function` param in python `cv` method.

## Bugfixes:
* Fixed deprecation warning on import (issue #1269)
* Fixed saved models logging_level/verbose parameters conflict (issue #696)
* Fixed kappa metric - in some cases there were integer overflow, switched accumulation types to double
* Fixed per float feature quantization settings defaults

## Educational materials
* Extended shap values [tutorial](https://github.com/catboost/tutorials/blob/master/model_analysis/shap_values_tutorial.ipynb) with summary plot examples. Thanks @azanovivan02


# Release 0.23

## New functionality

* It is possible now to train models on huge datasets that do not fit into CPU RAM.
This can be accomplished by storing only quantized data in memory (it is many times smaller). Use `catboost.utils.quantize` function to create quantized `Pool ` this way. See usage example in the issue #1116.
 Implemented by @noxwell.
* Python Pool class now has `save_quantization_borders` method that allows to save resulting borders into a [file](https://catboost.ai/docs/concepts/output-data_custom-borders.html) and use it for quantization of other datasets. Quantization can be a bottleneck of training, especially on GPU. Doing quantization once for several trainings can significantly reduce running time. It is recommended for large dataset to perform quantization first, save quantization borders, use them to quantize validation dataset, and then use quantized training and validation datasets for further training.
Use saved borders when quantizing other Pools by specifying `input_borders` parameter of the `quantize` method.
Implemented by @noxwell.
* Text features are supported on CPU
* It is now possible to set `border_count` > 255 for GPU training. This might be useful if you have a "golden feature", see [docs](https://catboost.ai/docs/concepts/parameter-tuning.html#golden-features).
* Feature weights are implemented.
Specify weights for specific features by index or name like `feature_weights="FeatureName1:1.5,FeatureName2:0.5"`.
Scores for splits with this features will be multiplied by corresponding weights.
Implemented by @Taube03.
* Feature penalties can be used for cost efficient gradient boosting.
Penalties are specified in a similar fashion to feature weights, using parameter `first_feature_use_penalties`.
This parameter penalized the first usage of a feature. This should be used in case if the calculation of the feature is costly.
The penalty value (or the cost of using a feature) is subtracted from scores of the splits of this feature if feature has not been used in the model.
After the feature has been used once, it is considered free to proceed using this feature, so no substruction is done.
There is also a common multiplier for all `first_feature_use_penalties`, it can be specified by `penalties_coefficient` parameter.
Implemented by @Taube03 (issue #1155)
* `recordCount` attribute is added to PMML models (issue #1026).

## New losses and metrics

* New ranking objective 'StochasticRank', details in [paper](https://arxiv.org/abs/2003.02122).
* `Tweedie` loss is supported now. It can be a good solution for right-skewed target with many zero values, see [tutorial](https://github.com/catboost/tutorials/blob/master/regression/tweedie.ipynb).
When using `CatBoostRegressor.predict` function, default `prediction_type` for this loss will be equal to `Exponent`. Implemented by @ilya-pchelintsev (issue #577)
* Classification metrics now support a new parameter `proba_border`. With this parameter you can set decision boundary for treating prediction as negative or positive. Implemented by @ivanychev.
* Metric `TotalF1` supports a new parameter `average` with possible value `weighted`, `micro`, `macro`. Implemented by @ilya-pchelintsev.
* It is possible now to specify a custom multi-label metric in python. Note that it is only possible to calculate this metric and use it as `eval_metric`. It is not possible to used it as an optimization objective.
To write a multi-label metric, you need to define a python class which inherits from `MultiLabelCustomMetric` class. Implemented by @azikmsu.

## Improvements of grid and randomized search

* `class_weights` parameter is now supported in grid/randomized search. Implemented by @vazgenk.
* Invalid option configurations are automatically skipped during grid/randomized search. Implemented by @borzunov.
* `get_best_score` returns train/validation best score after grid/randomized search (in case of refit=False). Implemented by @rednevaler.

## Improvements of model analysis tools

* Computation of SHAP interaction values for CatBoost models. You can pass type=EFstrType.ShapInteractionValues to `CatBoost.get_feature_importance` to get a matrix of SHAP values for every prediction.
By default, SHAP interaction values are calculated for all features. You may specify features of interest using the `interaction_indices` argument.
Implemented by @IvanKozlov98.
* SHAP values can be calculated approximately now which is much faster than default mode. To use this mode specify `shap_calc_type` parameter of  `CatBoost.get_feature_importance` function as  `"Approximate"`. Implemented by @LordProtoss (issue #1146).
* `PredictionDiff` model analysis method can now be used with models that contain non symmetric trees. Implemented by @felixandrer.

## New educational materials

* A [tutorial](https://github.com/catboost/tutorials/blob/master/regression/tweedie.ipynb) on tweedie regression
* A [tutorial](https://github.com/catboost/tutorials/blob/master/regression/poisson.ipynb) on poisson regression
* A detailed [tutorial](https://github.com/catboost/tutorials/blob/master/metrics/AUC_tutorial.ipynb) on different types of AUC metric, which explains how different types of AUC can be used for binary classification, multiclassification and ranking tasks.

## Breaking changes

* When using `CatBoostRegressor.predict` function for models trained with `Poisson` loss, default `prediction_type` will be equal to `Exponent` (issue #1184). Implemented by @garkavem.

This release also contains bug fixes and performance improvements, including a major speedup for sparse data on GPU.

# Release 0.22

## New features:
- The main feature of the release is the support of non symmetric trees for training on CPU.
Using non symmetric trees might be useful if one-hot encoding is present, or data has little noise.
To try non symmetric trees change [``grow_policy`` parameter](https://catboost.ai/docs/concepts/parameter-tuning.html#tree-growing-policy).
Starting from this release non symmetric trees are supported for both CPU and GPU training.
- The next big feature improves catboost text features support.
Now tokenization is done during training, you don't have to do lowercasing, digit extraction and other tokenization on your own, catboost does it for you.
- Auto learning-rate is now supported in CPU MultiClass mode.
- CatBoost class supports ``to_regressor`` and ``to_classifier`` methods.

The release also contains a list of bug fixes.

# Release 0.21

## New features:
- The main feature of this release is the Stochastic Gradient Langevin Boosting (SGLB) mode that can improve quality of your models with non-convex loss functions. To use it specify ``langevin`` option and tune ``diffusion_temperature`` and ``model_shrink_rate``. See [the corresponding paper](https://arxiv.org/abs/2001.07248) for details.

## Improvements:

- Automatic learning rate is applied by default not only for ``Logloss`` objective, but also for ``RMSE`` (on CPU and GPU) and ``MultiClass`` (on GPU).
- Class labels type information is stored in the model. Now estimators in python package return values of proper type in ``classes_`` attribute and for prediction functions with ``prediction_type=Class``. #305, #999, #1017.
  Note: Class labels loaded from datasets in [CatBoost dsv format](https://catboost.ai/docs/concepts/input-data_values-file.html) always have string type now.

## Bug fixes:
- Fixed huge memory consumption for text features. #1107
- Fixed crash on GPU on big datasets with groups (hundred million+ groups).
- Fixed class labels consistency check and merging in model sums (now class names in binary classification are properly checked and added to the result as well)
- Fix for confusion matrix (PR #1152), thanks to @dmsivkov.
- Fixed shap values calculation when ``boost_from_average=True``. #1125
- Fixed use-after-free in fstr PredictionValuesChange with specified dataset
- Target border and class weights are now taken from model when necessary for feature strength, metrics evaluation, roc_curve, object importances and calc_feature_statistics calculations.
- Fixed that L2 regularization was not applied for non symmetric trees for binary classification on GPU.
- [R-package] Fixed the bug that ``catboost.get_feature_importance`` did not work after model is loaded #1064
- [R-package] Fixed the bug that ``catboost.train`` did not work when called with the single dataset parameter. #1162
- Fixed L2 score calculation on CPU

##Other:

- Starting from this release Java applier is released simultaneously with other components and has the same version.

##Compatibility:

- Models trained with this release require applier from this release or later to work correctly.

# Release 0.20.2

## New features:
- String class labels are now supported for binary classification
- [CLI only] Timestamp column for the datasets can be provided in separate files.
- [CLI only] Timesplit feature evaluation.
- Process groups of any size in block processing.


## Bug fixes:
- ``classes_count`` and ``class_weight`` params can be now used with user-defined loss functions. #1119
- Form correct metric descriptions on GPU if ``use_weights`` gets value by default. #1106
- Correct ``model.classes_`` attribute for binary classification (proper labels instead of always ``0`` and ``1``). #984
- Fix ``model.classes_`` attribute when classes_count parameter was specified.
- Proper error message when categorical features specified for MultiRMSE training. #1112
- Block processing: It is valid for all groups in a single block to have weights equal to 0
- fix empty asymmetric tree index calculation. #1104

# Release 0.20.1

## New features:
- Have `leaf_estimation_method=Exact` the default for MAPE loss
- Add `CatBoostClassifier.predict_log_proba()`, PR #1095

## Bug fixes:
- Fix usability of read-only numpy arrays, #1101
- Fix python3 compatibility for `get_feature_importance`, PR #1090
- Fix loading model from snapshot for `boost_from_average` mode

# Release 0.20

New submodule for text processing!
It contains two classes to help you make text features ready for training:
- [Tokenizer](https://github.com/catboost/catboost/blob/afb8331a638de280ba2aee3831ac9df631e254a0/library/text_processing/tokenizer/tokenizer.pxi#L77) -- use this class to split text into tokens (automatic lowercase and punctuation removal)
- [Dictionary](https://github.com/catboost/catboost/tree/master/library/text_processing/dictionary) -- with this class you create a dictionary which maps tokens to numeric identifiers. You then use these identifiers as new features.

## New features:
- Enabled `boost_from_average` for `MAPE` loss function

## Bug fixes:
- Fixed `Pool` creation from `pandas.DataFrame` with discontinuous columns, #1079
- Fixed `standalone_evaluator`, PR #1083

## Speedups:
- Huge speedup of preprocessing in python-package for datasets with many samples (>10 mln)

# Release 0.19.1

## New features:
- With this release we support `Text` features for *classification on GPU*. To specify text columns use `text_features` parameter. Achieve better quality by using text information of your dataset. See more in [Learning CatBoost with text features](https://github.com/catboost/tutorials/blob/master/text_features/text_features_in_catboost.ipynb)
- `MultiRMSE` loss function is now available on CPU. Labels for the multi regression mode should be specified in separate `Label` columns
- MonoForest framework for model analysis, based on our NeurIPS 2019 [paper](https://papers.nips.cc/paper/9530-monoforest-framework-for-tree-ensemble-analysis). Learn more in [MonoForest tutorial](https://github.com/catboost/tutorials/tree/master/model_analysis/monoforest_tutorial.ipynb)
- `boost_from_average` is now `True` by default for `Quantile` and `MAE` loss functions, which improves the resulting quality

## Speedups:
- Huge reduction of preprocessing time for datasets loaded from files and for datasets with many samples (> 10 million), which was a bottleneck for GPU training
- 3x speedup for small datasets


# Release 0.18.1

## New features:
- Now `datasets.msrank()` returns _full_ msrank dataset. Previously, it returned the first 10k samples.
We have added `msrank_10k()` dataset implementing the past behaviour.

## Bug fixes:
- `get_object_importance()` now respects parameter `top_size`, #1045 by @ibuda

# Release 0.18

- The main feature of the release is huge speedup on small datasets. We now use MVS sampling for CPU regression and binary classification training by default, together with `Plain` boosting scheme for both small and large datasets. This change not only gives the huge speedup but also provides quality improvement!
- The `boost_from_average` parameter is available in `CatBoostClassifier` and `CatBoostRegressor`
- We have added new formats for describing monotonic constraints. For example, `"(1,0,0,-1)"` or `"0:1,3:-1"` or `"FeatureName0:1,FeatureName3:-1"` are all valid specifications. With Python and `params-file` json, lists and dictionaries can also be used

## Bugs fixed:
- Error in `Multiclass` classifier training, #1040
- Unhandled exception when saving quantized pool, #1021
- Python 3.7: `RuntimeError` raised in `StagedPredictIterator`, #848

# Release 0.17.5

## Bugs fixed:
- `System of linear equations is not positive definite` when training MultiClass on Windows, #1022

# Release 0.17.4

## Improvements:
- Massive 2x speedup for `MultiClass` with many classes
- Updated MVS implementation. See _Minimal Variance Sampling in Stochastic Gradient Boosting_ by Bulat Ibragimov and Gleb Gusev at [NeurIPS 2019](https://neurips.cc/Conferences/2019)
- Added `sum_models` in R-package, #1007

## Bugs fixed:
- Multi model initialization in python, #995
- Mishandling of 255 borders in training on GPU, #1010

# Release 0.17.3

## Improvements:
- New visualization for parameter tuning. Use `plot=True` parameter in `grid_search` and `randomized_search` methods to show plots in jupyter notebook
- Switched to jemalloc allocator instead of LFalloc in CLI and model interfaces to fix some problems on Windows 7 machines, #881
- Calculation of binary class AUC is faster up to 1.3x
- Added [tutorial](https://github.com/catboost/tutorials/blob/master/convert_onnx_model/tutorial_convert_onnx_models.ipynb) on using fast CatBoost applier with LightGBM models

## Bugs fixed:
- Shap values for `MultiClass` objective don't give constant 0 value for the last class in case of GPU training.
  Shap values for `MultiClass` objective are now calculated in the following way. First, predictions are normalized so that the average of all predictions is zero in each tree. The normalized predictions produce the same probabilities as the non-normalized ones. Then the shap values are calculated for every class separately. Note that since the shap values are calculated on the normalized predictions, their sum for every class is equal to the normalized prediction
- Fixed bug in rangking tutorial, #955
- Allow string value for `per_float_feature_quantization` parameter, #996

# Release 0.17.2

## Improvements:
- For metric MAE on CPU default value of `leaf-estimation-method` is now `Exact`
- Speed up `LossFunctionChange` feature strength computation

## Bugs fixed:
- Broken label converter in grid search for multiclassification, #993
- Incorrect prediction with monotonic constraint, #994
- Invalid value of `eval_metric` in output of `get_all_params()`, #940
- Train AUC is not computed because hint `skip_train~false` is ignored, #970

# Release 0.17.1

## Bugs fixed:
- Incorrect estimation of total RAM size on Windows and Mac OS, #989
- Failure when dataset is a `numpy.ndarray` with `order='F'`
- Disable `boost_from_average` when baseline is specified

## Improvements:
- Polymorphic raw features storage (2x---25x faster data preparation for numeric features in non-float32 columns as either `pandas.DataFrame` or `numpy.ndarray` with `order='F'`).
- Support AUC metric for `CrossEntropy` loss on CPU
- Added `datasets.rotten_tomatoes()`, a textual dataset
- Usability of `monotone_constraints`, #950

## Speedups:
- Optimized computation of `CrossEntropy` metric on CPUs with SSE3

# Release 0.17

## New features:
- Sparse data support
- We've implemented and set to default `boost_from_average` in RMSE mode. It gives a boost in quality especially for a small number of iterations.

## Improvements:
- Quantile regression on CPU
- default parameters for Poisson regression

## Speedups:
- A number of speedups for training on CPU
- Huge speedups for loading datasets with categorical features represented as `pandas.Categorical`.
Hint: use `pandas.Categorical` instead of object to speed up loading up to 200x.

# Release 0.16.5

## Breaking changes:
- All metrics except for AUC metric now use weights by default.

## New features:
- Added `boost_from_average` parameter for RMSE training on CPU which might give a boost in quality.
- Added conversion from ONNX to CatBoost. Now you can convert XGBoost or LightGBM model to ONNX, then convert it to CatBoost and use our fast applier. Use `model.load_model(model_path, format="onnx")` for that.

## Speed ups:
- Training is  \~15% faster for datasets with categorical features.

## Bug fixes:
- R language: `get_features_importance` with `ShapValues` for `MultiClass`,  #868
- NormalizedGini was not calculated,  #962
- Bug in leaf calculation which could result in slightly worse quality if you use weights in binary classification mode
- Fixed `__builtins__` import in Python3 in PR #957, thanks to @AbhinavanT


# Release 0.16.4

## Bug fixes:
- Versions 0.16.* had a bug in python applier with categorical features for applying on more than 128 documents.

## New features:
- It is now possible to use pairwise modes for datasets without groups

## Improvements:
- 1.8x Evaluation speed on asymmetrical trees

# Release 0.16.3

## Breaking changes:
- Renamed column `Feature Index` to `Feature Id` in prettified output of python method `get_feature_importance()`, because it supports feature names now
- Renamed option `per_float_feature_binarization` (`--per-float-feature-binarization`) to `per_float_feature_quantization` (`--per-float-feature-quantization`)
- Removed parameter `inverted` from python `cv` method. Added `type` parameter instead, which can be set to `Inverted`
- Method `get_features()` now works only for datasets without categorical features

## New features
- A new multiclass version of AUC metric, called `AUC Mu`, which was proposed by Ross S. Kleiman on NeurIPS 2019, [link](http://proceedings.mlr.press/v97/kleiman19a/kleiman19a.pdf)
- Added time series cv
- Added `MeanWeightedTarget` in `fstat`
- Added `utils.get_confusion_matrix()`
- Now feature importance can be calculated for non-symmetric trees


# Release 0.16.2

## Breaking changes:
- Removed `get_group_id()` and `get_features()` methods of `Pool` class

## New model analysis tools:
- Added `PredictionDiff` type of `get_feature_importance()` method, which is a new method for model analysis. The method shows how the features influenced the fact that among two samples one has a higher prediction. It allows to debug ranking models: you find a pair of samples ranked incorrectly and you look at what features have caused that.
- Added `plot_predictions()` method

## New features:
- `model.set_feature_names()` method in Python
- Added stratified split to parameter search methods
- Support `catboost.load_model()` from CPU snapshots for numerical-only datasets
- `CatBoostClassifier.score()` now supports `y` as `DataFrame`
- Added `sampling_frequency`, `per_float_feature_binarization`, `monotone_constraints` parameters to `CatBoostClassifier` and `CatBoostRegresssor`

## Speedups:
- 2x speedup of multi-classification mode

## Bugfixes:
- Fixed `score()` for multiclassification, #924
- Fixed `get_all_params()` function,  #926

## Other improvements:
- Clear error messages when a model cannot be saved


# Release 0.16.1

## Breaking changes:
- parameter `fold_count` is now called `cv` in [`grid_search()`](https://catboost.ai/docs/concepts/python-reference_catboost_grid_search.html) and [`randomized_search`](https://catboost.ai/docs/concepts/python-reference_catboost_randomized_search.html)
- cv results are now returned from `grid_search()` and `randomized_search()` in `res['cv_results']` field

## New features:
- R-language function `catboost.save_model()` now supports PMML, ONNX and other formats
- Parameter `monotone_constraints` in python API allows specifying numerical features that the prediction shall depend on monotonically

## Bug fixes:
- Fixed `eval_metric` calculation for training with weights (in release 0.16 evaluation of a metric that was equal to an optimized loss did not use weights by default, so overfitting detector worked incorrectly)

## Improvements:
- Added option `verbose` to `grid_search()` and `randomized_search()`
- Added [tutorial](https://github.com/catboost/tutorials/blob/master/hyperparameters_tuning/hyperparameters_tuning.ipynb) on `grid_search()` and `randomized_search()`


# Release 0.16

## Breaking changes:
- `MultiClass` loss has now the same sign as Logloss. It had the other sign before and was maximized, now it is minimized.
- `CatBoostRegressor.score` now returns the value of $R^2$ metric instead of RMSE to be more consistent with the behavior of scikit-learn regressors.
- Changed metric parameter `use_weights` default value to false (except for ranking metrics)

## New features:
- It is now possible to apply model on GPU
- We have published two new realworld datasets with monotonic constraints, `catboost.datasets.monotonic1()` and `catboost.datasets.monotonic2()`. Before that  there was only `california_housing` dataset in open-source with monotonic constraints. Now you can use these two to benchmark algorithms with monotonic constraints.
- We've added several new metrics to catboost, including `DCG`, `FairLoss`, `HammingLoss`, `NormalizedGini` and `FilteredNDCG`
- Introduced efficient `GridSearch` and `RandomSearch` implementations.
- `get_all_params()` Python function returns the values of all training parameters, both user-defined and default.
- Added more synonyms for training parameters to be more compatible with other GBDT libraries.

## Speedups:
- AUC metric is computationally very expensive. We've implemented parallelized calculation of this metric, now it can be calculated on every iteration (or every k-th iteration) about 4x faster.

## Educational materials:
- We've improved our command-line tutorial, now it has examples of files and more information.

## Fixes:
- Automatic `Logloss` or `MultiClass` loss function deduction for `CatBoostClassifier.fit` now also works if the training dataset is specified as `Pool` or filename string.
- And some other fixes


# Release 0.15.2

## Breaking changes:
- Function `get_feature_statistics` is replaced by `calc_feature_statistics`
- Scoring function `Correlation` is renamed to `Cosine`
- Parameter `efb_max_conflict_fraction` is renamed to `sparse_features_conflict_fraction`

## New features:
- Models can be saved in PMML format now.
> **Note:**  PMML does not have full categorical features support, so to have the model in PMML format for datasets with categorical features you need to use set `one_hot_max_size` parameter to some large value, so that all categorical features are one-hot encoded
- Feature names can be used to specify ignored features

## Bug fixes, including:
- Fixed restarting of CV on GPU for datasets without categorical features
- Fixed learning continuation errors with changed dataset (PR #879) and with model loaded from file (#884)
- Fixed NativeLib for JDK 9+ (PR #857)


# Release 0.15.1

## Bug fixes
- restored parameter `fstr_type` in Python and R interfaces


# Release 0.15

## Breaking changes
- cv is now stratified by default for `Logloss`, `MultiClass` and `MultiClassOneVsAll`.
- We have removed `border` parameter of `Logloss` metric. You need to use `target_border` as a separate training parameter now.
- `CatBoostClassifier` now runs `MultiClass` if more than 2 different values are present in training dataset labels.
- `model.best_score_["validation_0"]` is replaced with `model.best_score_["validation"]` if a single validation dataset is present.
- `get_object_importance` function parameter `ostr_type` is renamed to `type` in Python and R.

## Model analysis
- Tree visualisation by [@karina-usmanova](https://github.com/karina-usmanova).
- New feature analysis: plotting information about how a feature was used in the model by [@alexrogozin12](https://github.com/alexrogozin12).
- Added `plot` parameter to `get_roc_curve`, `get_fpr_curve` and `get_fnr_curve` functions from `catboost.utils`.
- Supported prettified format for all types of feature importances.

## New ways of doing predictions
- Rust applier by [@shuternay](https://github.com/shuternay).
- DotNet applier by [@17minutes](https://github.com/17minutes).
- One-hot encoding for categorical features in CatBoost CoreML model by Kseniya Valchuk and Ekaterina Pogodina.


## New objectives
- Expectile Regression by [@david-waterworth](https://github.com/david-waterworth).
- Huber loss by [@atsky](https://github.com/atsky).

## Speedups
- Speed up of shap values calculation for single object or for small number of objects by [@Lokutrus](https://github.com/Lokutrus).
- Cheap preprocessing and no fighting of overfitting if there is little amount of iterations (since you will not overfit anyway).

## New functionality
- Prediction of leaf indices.

## New educational materials
- Rust tutorial by [@shuternay](https://github.com/shuternay).
- C# tutorial.
- Leaf indices.
- Tree visualisation tutorial by [@karina-usmanova](https://github.com/karina-usmanova).
- Google Colab tutorial for regression in catboost by [@col14m](https://github.com/col14m).

And a set of fixes for your issues.


# Release 0.14.2

## New features
- Add `has_header` parameter to [`CatboostEvaluation`](https://github.com/catboost/catboost/blob/2f35e0366c0bb6c1b44be89fda0a02fe12f84513/catboost/python-package/catboost/eval/catboost_evaluation.py#L30) class.

## Breaking changes
- Change output feature indices separator (`:` to `;`) in the `CatboostEvaluation` class.

# Release 0.14.1

## Breaking changes
- Changed default value for `--counter-calc-method` option to `SkipTest`

## New features:
- Add guid to trained models. You can access it in Python using [`get_metadata`](https://catboost.ai/docs/concepts/python-reference_catboost_metadata.html) function, for example `print catboost_model.get_metadata()['model_guid']`

## Bug fixes and other changes:
- Compatibility with glibc 2.12
- Improved embedded documentation
- Improved warning and error messages

# Release 0.14.0

## New features:

- GPU training now supports several tree learning strategies, selectable with `grow_policy` parameter. Possible values:
  - `SymmetricTree` -- The tree is built level by level until `max_depth` is reached. On each iteration, all leaves from the last tree level will be split with the same condition. The resulting tree structure will always be symmetric.
  - `Depthwise` -- The tree is built level by level until `max_depth` is reached. On each iteration, all non-terminal leaves from the last tree level will be split. Each leaf is split by condition with the best loss improvement.
  - `Lossguide` -- The tree is built leaf by leaf until `max_leaves` limit is reached. On each iteration, non-terminal leaf with best loss improvement will be split.
  > **Note:** grow policies `Depthwise` and `Lossguide` currently support only training and prediction modes. They do not support model analysis (like feature importances and SHAP values) and saving to different model formats like CoreML, ONNX, and JSON.
  - The new grow policies support several new parameters:
    `max_leaves` -- Maximum leaf count in the resulting tree, default 31. Used only for `Lossguide` grow policy. __Warning:__ It is not recommended to set this parameter greater than 64, as this can significantly slow down training.
    `min_data_in_leaf` -- Minimum number of training samples per leaf, default 1. CatBoost will not search for new splits in leaves with sample count less than  `min_data_in_leaf`. This option is available for `Lossguide` and `Depthwise` grow policies only.
  > **Note:** the new types of trees will be at least 10x slower in prediction than default symmetric trees.

- GPU training also supports several score functions, that might give your model a boost in quality. Use parameter `score_function` to experiment with them.

- Now you can use quantization with more than 255 borders and `one_hot_max_size` > 255 in CPU training.

## New features in Python package:
- It is now possible to use `save_borders()` function to write borders to a file after training.
- Functions `predict`, `predict_proba`, `staged_predict`, and `staged_predict_proba` now support applying a model to a single object, in addition to usual data matrices.

## Speedups:
- Impressive speedups for sparse datsets. Will depend on the dataset, but will be at least 2--3 times for sparse data.

## Breaking changes:
- Python-package class attributes don't raise exceptions now. Attributes return `None` if not initialized.
- Starting from 0.13 we have new feature importances for ranking modes. The new algorithm for feature importances shows how much features contribute to the optimized loss function. They are also signed as opposed to feature importances for not ranking modes which are non negative. This importances are expensive to calculate, thus we decided to not calculate them by default during training starting from 0.14. You need to calculate them after training.

# Release 0.13.1

## Changes:
- Fixed a bug in shap values that was introduced in v0.13

# Release 0.13

## Speedups:
- Impressive speedup of CPU training for datasets with predominantly binary features (up to 5-6x).
- Speedup prediction and shap values array casting on large pools (issue [#684](https://github.com/catboost/catboost/issues/684)).

## New features:
- We've introduced a new type of feature importances - `LossFunctionChange`.
  This type of feature importances works well in all the modes, but is especially good for ranking. It is more expensive to calculate, thus we have not made it default. But you can look at it by selecting the type of feature importance.
- Now we support online statistics for categorical features in `QuerySoftMax` mode on GPU.
- We now support feature names in `cat_features`, PR [#679](https://github.com/catboost/catboost/pull/679) by [@infected-mushroom](https://github.com/infected-mushroom) - thanks a lot [@infected-mushroom](https://github.com/infected-mushroom)!
- We've intoduced new sampling_type `MVS`, which speeds up CPU training if you use it.
- Added `classes_` attribute in python.
- Added support for input/output borders files in python package. Thank you [@necnec](https://github.com/necnec) for your PR [#656](https://github.com/catboost/catboost/pull/656)!
- One more new option for working with categorical features is `ctr_target_border_count`.
  This option can be used if your initial target values are not binary and you do regression or ranking. It is equal to 1 by default, but you can try increasing it.
- Added new option `sampling_unit` that allows to switch sampling from individual objects to entire groups.
- More strings are interpreted as missing values for numerical features (mostly similar to pandas' [read_csv](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)).
- Allow `skip_train` property for loss functions in cv method. Contributed by GitHub user [@RakitinDen](https://github.com/RakitinDen), PR [#662](https://github.com/catboost/catboost/pull/662), many thanks.
- We've improved classification mode on CPU, there will be less cases when the training diverges.
  You can also try to experiment with new `leaf_estimation_backtracking` parameter.
- Added new compare method for visualization, PR [#652](https://github.com/catboost/catboost/pull/652). Thanks [@Drakon5999](https://github.com/Drakon5999) for your contribution!
- Implemented `__eq__` method for `CatBoost*` python classes (PR [#654](https://github.com/catboost/catboost/pull/654)). Thanks [@daskol](https://github.com/daskol) for your contribution!
- It is now possible to output evaluation results directly to `stdout` or `stderr` in command-line CatBoost in [`calc` mode](https://catboost.ai/docs/concepts/cli-reference_calc-model.html) by specifying `stream://stdout` or `stream://stderr` in `--output-path` parameter argument. (PR [#646](https://github.com/catboost/catboost/pull/646)). Thanks [@towelenee](https://github.com/towelenee) for your contribution!
- New loss function - [Huber](https://en.wikipedia.org/wiki/Huber_loss). Can be used as both an objective and a metric for regression. (PR [#649](https://github.com/catboost/catboost/pull/649)). Thanks [@atsky](https://github.com/atsky) for your contribution!

## Changes:
- Changed defaults for `one_hot_max_size` training parameter for groupwise loss function training.
- `SampleId` is the new main name for former `DocId` column in input data format (`DocId` is still supported for compatibility). Contributed by GitHub user [@daskol](https://github.com/daskol), PR [#655](https://github.com/catboost/catboost/pull/655), many thanks.
- Improved CLI interface for cross-validation: replaced `-X/-Y` options with `--cv`, PR [#644](https://github.com/catboost/catboost/pull/644). Thanks [@tswr](https://github.com/tswr) for your pr!
- `eval_metrics` : `eval_period` is now clipped by total number of trees in the specified interval. PR [#653](https://github.com/catboost/catboost/pull/653). Thanks [@AntPon](https://github.com/AntPon) for your contribution!

## R package:
- Thanks to [@ws171913](https://github.com/ws171913) we made necessary changes to prepare catboost for CRAN integration, PR [#715](https://github.com/catboost/catboost/pull/715). This is in progress now.
- R interface for cross-validation contributed by GitHub user [@brsoyanvn](https://github.com/brsoyanvn), PR [#561](https://github.com/catboost/catboost/pull/561) -- many thanks [@brsoyanvn](https://github.com/brsoyanvn)!

## Educational materials:
- We've added new tutorial for [GPU training on Google Colaboratory](https://github.com/catboost/tutorials/blob/master/tools/google_colaboratory_cpu_vs_gpu_tutorial.ipynb).

We have also done a list of fixes and data check improvements.
Thanks [@brazhenko](https://github.com/brazhenko), [@Danyago98](https://github.com/Danyago98), [@infected-mushroom](https://github.com/infected-mushroom) for your contributions.

# Release 0.12.2
## Changes:
* Fixed loading of `epsilon` dataset into memory
* Fixed multiclass learning on GPU for >255 classes
* Improved error handling
* Some other minor fixes

# Release 0.12.1.1
## Changes:
* Fixed Python compatibility issue in dataset downloading
* Added `sampling_type` parameter for `YetiRankPairwise` loss

# Release 0.12.1
## Changes:
* Support saving models in ONNX format (only for models without categorical features).
* Added new dataset to our `catboost.datasets()` -- dataset [epsilon](catboost/benchmarks/model_evaluation_speed), a large dense dataset for binary classification.
* Speedup of Python `cv` on GPU.
* Fixed creation of `Pool` from `pandas.DataFrame` with `pandas.Categorical` columns.

# Release 0.12.0
## Breaking changes:
* Class weights are now taken into account by `eval_metrics()`,
  `get_feature_importance()`, and `get_object_importance()`.
  In previous versions the weights were ignored.
* Parameter `random-strength` for pairwise training (`PairLogitPairwise`,
  `QueryCrossEntropy`, `YetiRankPairwise`) is not supported anymore.
* Simultaneous use of `MultiClass` and `MultiClassOneVsAll` metrics is now
  deprecated.

## New functionality:
* `cv` method is now supported on GPU.
* String labels for classes are supported in Python.
  In multiclassification the string class names are inferred from the data.
  In binary classification for using string labels you should employ `class_names`
  parameter and specify which class is negative (0) and which is positive (1).
  You can also use `class_names` in multiclassification mode to pass all
  possible class names to the fit function.
* Borders can now be saved and reused.
  To save the feature quantization information obtained during training data
  preprocessing into a text file use cli option `--output-borders-file`.
  To use the borders for training use cli option `--input-borders-file`.
  This functionanlity is now supported on CPU and GPU (it was GPU-only in previous versions).
  File format for the borders is described [here](https://tech.yandex.com/catboost/doc/dg/concepts/input-data_custom-borders-docpage).
* CLI option `--eval-file` is now supported on GPU.

## Quality improvement:
* Some cases in binary classification are fixed where training could diverge

## Optimizations:
* A great speedup of the Python applier (10x)
* Reduced memory consumption in Python `cv` function (times fold count)

## Benchmarks and tutorials:
* Added [speed benchmarks](catboost/benchmarks/gpu_vs_cpu_training_speed) for CPU and GPU on a variety of different datasets.
* Added [benchmarks](catboost/benchmarks/ranking) of different ranking modes. In [this tutorial](catboost/tutorials/ranking/ranking_tutorial.ipynb) we compare
  different ranking modes in CatBoost, XGBoost and LightGBM.
* Added [tutorial](catboost/tutorials/apply_model/catboost4j_prediction_tutorial.ipynb) for applying model in Java.
* Added [benchmarks](catboost/benchmarks/shap_speed) of SHAP values calculation for CatBoost, XGBoost and LightGBM.
  The benchmarks also contain explanation of complexity of this calculation
  in all the libraries.

We also made a list of stability improvements
and stricter checks of input data and parameters.

And we are so grateful to our community members @canorbal and @neer201
for their contribution in this release. Thank you.


# Release 0.11.2
## Changes:
* Pure GPU implementation of NDCG metric
* Enabled LQ loss function
* Fixed NDCG metric on CPU
* Added `model_sum` mode to command line interface
* Added SHAP values benchmark (#566)
* Fixed `random_strength` for `Plain` boosting (#448)
* Enabled passing a test pool to caret training (#544)
* Fixed a bug in exporting the model as python code (#556)
* Fixed label mapper for multiclassification custom labels (#523)
* Fixed hash type of categorical features (#558)
* Fixed handling of cross-validation fold count options in python package (#568)


# Release 0.11.1
## Changes:
* Accelerated formula evaluation by ~15%
* Improved model application interface
* Improved compilation time for building GPU version
* Better handling of stray commas in list arguments
* Added a benchmark that employs Rossman Store Sales dataset to compare quality of GBDT packages
* Added references to Catboost papers in R-package CITATION file
* Fixed a build issue in compilation for GPU
* Fixed a bug in model applicator
* Fixed model conversion, #533
* Returned pre 0.11 behaviour for `best_score_` and `evals_result_` (issue #539)
* Make valid RECORD in wheel (issue #534)

# Release 0.11.0
## Changes:
* Changed default border count for float feature binarization to 254 on CPU to achieve better quality
* Fixed random seed to `0` by default
* Support model with more than 254 feature borders or one hot values when doing predictions
* Added model summation support in python: use `catboost.sum_models()` to sum models with provided weights.
* Added json model tutorial [json_model_tutorial.ipynb](https://github.com/catboost/catboost/blob/master/catboost/tutorials/apply_model/json_model_tutorial.ipynb)

# Release 0.10.4.1
## Changes:
- Bugfix for #518

# Release 0.10.4
## Breaking changes:
In python 3 some functions returned dictionaries with keys of type `bytes` - particularly eval_metrics and get_best_score. These are fixed to have keys of type `str`.
## Changes:
- New metric NumErrors:greater_than=value
- New metric and objective L_q:q=value
- model.score(X, y) - can now work with Pool and labels from Pool

# Release 0.10.3
## Changes:
* Added EvalResult output after GPU catboost training
* Supported prediction type option on GPU
* Added `get_evals_result()` method and `evals_result_` property to model in python wrapper to allow user access metric values
* Supported string labels for GPU training in cmdline mode
* Many improvements in JNI wrapper
* Updated NDCG metric: speeded up and added NDCG with exponentiation in numerator as a new NDCG mode
* CatBoost doesn't drop unused features from model after training
* Write training finish time and catboost build info to model metadata
* Fix automatic pairs generation for GPU PairLogitPairwise target

# Release 0.10.2
### Main changes:
* Fixed Python 3 support in `catboost.FeaturesData`
* 40% speedup QuerySoftMax CPU training

# Release 0.10.1
## Improvements
* 2x Speedup pairwise loss functions
* For all the people struggling with occasional NaNs in test datasets - now we only write warnings about it
## Bugfixes
* We set up default loss_function in `CatBoostClassifier` and `CatBoostRegressor`
* Catboost write `Warning` and `Error` logs to stderr

# Release 0.10.0
## Breaking changes
### R package
- In R package we have changed parameter name `target` to `label` in method [`save_pool()`](https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost-save_pool-docpage/)
### Python package
- We don't support Python 3.4 anymore
- CatBoostClassifier and CatBoostRegressor [`get_params()`](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_catboostclassifier_get_params-docpage/) method now returns only the params that were explicitly set when constructing the object. That means that CatBoostClassifier and CatBoostRegressor get_params() will not contain 'loss_function' if it was not specified.
This also means that this code:
```(python)
model1 = CatBoostClassifier()
params = model1.get_params()
model2 = CatBoost(params)
```
will create model2 with default loss_function RMSE, not with Logloss.
This breaking change is done to support sklearn interface, so that sklearn GridSearchCV can work.
- We've removed several attributes and changed them to functions. This was needed to avoid sklearn warnings:
`is_fitted_` => [`is_fitted()`](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_catboostclassifier_is_fitted-docpage/)
`metadata_` => [`get_metadata()`](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_catboostclassifier_metadata-docpage/)
- We removed file with model from constructor of estimator. This was also done to avoid sklearn warnings.
## Educational materials
- We added [tutorial](https://github.com/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb) for our ranking modes.
- We published our [slides](https://github.com/catboost/catboost/tree/master/slides), you are very welcome to use them.
## Improvements
### All
- Now it is possible to save model in json format.
- We have added Java interface for CatBoost model
- We now have static linkage with CUDA, so you don't have to install any particular version of CUDA to get catboost working on GPU.
- We implemented both multiclass modes on GPU, it is very fast.
- It is possible now to use multiclass with string labels, they will be inferred from data
- Added `use_weights` parameter to [metrics](https://tech.yandex.com/catboost/doc/dg/concepts/loss-functions-docpage/). By default all metrics, except for AUC use weights, but you can disable it. To calculate metric value without weights, you need to set this parameter to false. Example: Accuracy:use_weights=false. This can be done only for custom_metrics or eval_metric, not for the objective function. Objective function always uses weights if they are present in the dataset.
- We now use snapshot time intervals. It will work much faster if you save snapshot every 5 or 10 minutes instead of saving it on every iteration.
- Reduced memory consumption by ranking modes.
- Added automatic feature importance evaluation after completion of GPU training.
- Allow inexistent indexes in ignored features list
- Added [new metrics](https://tech.yandex.com/catboost/doc/dg/concepts/loss-functions-docpage/): `LogLikelihoodOfPrediction`, `RecallAt:top=k`, `PrecisionAt:top=k` and `MAP:top=k`.
- Improved quality for multiclass with weighted datasets.
- Pairwise modes now support automatic pairs generation (see [tutorial](https://github.com/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb) for that).
- Metric `QueryAverage` is renamed to a more clear `AverageGain`. This is a very important ranking metric. It shows average target value in top k documents of a group.
Introduced parameter `best_model_min_trees` - the minimal number of trees the best model should have.
### Python
- We now support sklearn GridSearchCV: you can pass categorical feature indices when constructing estimator. And then use it in GridSearchCV.
- We added new method to utils - building of ROC curve: [`get_roc_curve`](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_utils_get_roc_curve-docpage/).
- Added [`get_gpu_device_count()`](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_utils_get_gpu_device_count-docpage/) method to python package. This is a way to check if your CUDA devices are available.
- We implemented automatical selection of decision-boundary using ROC curve. You can select best classification boundary given the maximum FPR or FNR that you allow to the model. Take a look on [`catboost.select_threshold(self, data=None, curve=None, FPR=None, FNR=None, thread_count=-1)`](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_utils_select_threshold-docpage/). You can also calculate FPR and FNR for each boundary value.
- We have added pool slicing: [`pool.slice(doc_indices)`](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_pool_slice-docpage/)
- Allow GroupId and SubgroupId specified as strings.
### R package
- GPU support in R package. You need to use parameter `task_type='GPU'` to enable GPU training.
- Models in R can be saved/restored by means of R: save/load or saveRDS/readRDS
## Speedups
- New way of loading data in Python using [FeaturesData structure](https://tech.yandex.com/catboost/doc/dg/concepts/python-features-data__desc-docpage/). Using FeaturesData will speed up both loading data for training and for prediction. It is especially important for prediction, because it gives around 10 to 20 times python prediction speedup.
- Training multiclass on CPU ~ 60% speedup
- Training of ranking modes on CPU ~ 50% speedup
- Training of ranking modes on GPU ~ 50% speedup for datasets with many features and not very many objects
- Speedups of metric calculation on GPU. Example of speedup on our internal dataset: training with - AUC eval metric with test dataset with 2kk objects is speeded up 7sec => 0.2 seconds per iteration.
- Speedup of all modes on CPU training.

We also did a lot of stability improvements, and improved usability of the library, added new parameter synonyms and improved input data validations.

Thanks a lot to all people who created issues on github. And thanks a lot to our contributor @pukhlyakova who implemented many new useful metrics!

# Release 0.9.1.1
## Bugfixes
- Fixed #403 bug in cuda train submodule (training crashed without evaluation set)
- Fixed exception propagation on pool parsing stage
- Add support of string `GroupId` and `SubgroupId` in python-package
- Print real class names instead of their labels in eval output

# Release 0.9
## Breaking Changes
- We removed calc_feature_importance parameter from Python and R.
Now feature importance calculation is almost free, so we always calculate feature importances. Previously you could disable it if it was slowing down your training.
- We removed Doc type for feature importances. Use Shap instead.
- We moved thread_count parameter in Python get_feature_importance method to the end.

## Ranking
In this release we added several very powerfull ranking objectives:
- PairLogitPairwise
- YetiRankPairwise
- QueryCrossEntropy (GPU only)

Other ranking improvements:
- We have made improvements to our existing ranking objectives QuerySoftMax and PairLogit.
- We have added group weights support.

## Accuracy improvements
- Improvement for datasets with weights
- Now we automatically calculate a good learning rate for you in the start of training, you don't have to specify it. After the training has finished, you can look on the training curve on evaluation dataset and make ajustments to the selected learning rate, but it will already be a good value.

## Speedups:
- Several speedups for GPU training.
- 1.5x speedup for applying the model.
- Speed up multi classificaton training.
- 2x speedup for AUC calculation in eval_metrics.
- Several speedups for eval_metrics for other metrics.
- 100x speed up for Shap values calculation.
- Speedup for feature importance calculation. It used to be a bottleneck for GPU training previously, now it's not.
- We added possibility to not calculate metric on train dataset using `MetricName:hints=skip_train~false` (it might speed up your training if metric calculation is a bottle neck, for example, if you calculate many metrics or if you calculate metrics on GPU).
- We added possibility to calculate metrics only periodically, not on all iterations. Use metric_period for that.
(previously it only disabled verbose output on each iteration).
- Now we disable by default calculation of expensive metrics on train dataset. We don't calculate AUC and PFound metrics on train dataset by default. You can also disable calculation of other metrics on train dataset using `MetricName:hints=skip_train~true`. If you want to calculate AUC or PFound on train dataset you can use `MetricName:hints=skip_train~false`.
- Now if you want to calculate metrics using eval_metrics or during training you can use metric_period to skip some iterations. It will speed up eval_metrics and it might speed up training, especially GPU training.
Note that the most expensive metric calculation is AUC calculation, for this metric and large datasets it makes sense to use metric_period.
If you only want to see less verbose output, and still want to see metric values on every iteration written in file, you can use `verbose=n` parameter
- Parallelization of calculation of most of the metrics during training

## Improved GPU experience
- It is possible now to calculate and visualise custom_metric during training on GPU.
Now you can use our Jupyter visualization, CatBoost viewer or TensorBoard the same way you used it for CPU training. It might be a bottleneck, so if it slows down your training use `metric_period=something` and `MetricName:hints=skip_train~false`
- We switched to CUDA 9.1. Starting from this release CUDA 8.0 will not be supported
- Support for external borders on GPU for cmdline

## Improved tools for model analysis
- We added support of feature combinations to our Shap values implementation.
- Added Shap values for MultiClass and added an example of it's usage to our [Shap tutorial](https://github.com/catboost/tutorials/blob/master/model_analysis/shap_values_tutorial.ipynb).
- Added pretified parameter to get_feature_importance(). With `pretified=True` the function will return list of features with names sorted in descending order by their importance.
- Improved interfaces for eval-feature functionality
- Shap values support in R-package

## New features
- It is possible now to save any metainformation to the model.
- Empty values support
- Better support of sklearn
- feature_names_ for CatBoost class
- Added silent parameter
- Better stdout
- Better diagnostic for invalid inputs
- Better documentation
- Added a flag to allow constant labels

## New metrics
We added many new metrics that can be used for visualization, overfitting detection, selecting of best iteration of training or for cross-validation:
- BierScore
- HingeLoss
- HammingLoss
- ZeroOneLoss
- MSLE
- MAE
- BalancedAccuracy
- BalancedErrorRate
- Kappa
- Wkappa
- QueryCrossEntropy
- NDCG

## New ways to apply the model
- Saving model as C++ code
- Saving model with categorical features as Python code

## New ways to build the code
Added make files for binary with CUDA and for Python package

## Tutorials
We created a new [repo with tutorials](https://github.com/catboost/tutorials/), now you don't have to clone the whole catboost repo to run Jupyter notebook with a tutorial.

## Bugfixes
We have also a set of bugfixes and we are gratefull to everyone who has filled a bugreport, helping us making the library better.

## Thanks to our Contributors
This release contains contributions from CatBoost team.
We want to especially mention @pukhlyakova who implemented lots of useful metrics.

# Release 0.8.1
## Bug Fixes and Other Changes
- New model method `get_cat_feature_indices()` in Python wrapper.
- Minor fixes and stability improvements.

# Release 0.8
## Breaking changes
- We fixed bug in CatBoost. Pool initialization from `numpy.array` and `pandas.dataframe` with string values that can cause slight inconsistence while using trained model from older versions. Around 1% of cat feature hashes were treated incorrectly. If you expirience quality drop after update you should consider retraining your model.

## Major Features And Improvements
- Algorithm for finding most influential training samples for a given object from the 'Finding Influential Training Samples for Gradient Boosted Decision Trees' [paper](https://arxiv.org/pdf/1802.06640.pdf) is implemented. This mode for every object from input pool calculates scores for every object from train pool. A positive score means that the given train object has made a negative contribution to the given test object prediction. And vice versa for negative scores. The higher score modulo - the higher contribution.
See `get_object_importance` model method in Python package and `ostr` mode in cli-version. Tutorial for Python is available [here](https://github.com/catboost/tutorials/blob/master/model_analysis/object_importance_tutorial.ipynb).
More details and examples will be published in documentation soon.
- We have implemented new way of exploring feature importance - Shap values from [paper](https://arxiv.org/pdf/1706.06060.pdf). This allows to understand which features are most influent for a given object. You can also get more insite about your model, see details in a [tutorial](https://github.com/catboost/tutorials/blob/master/model_analysis/shap_values_tutorial.ipynb).
- Save model as code functionality published. For now you could save model as Python code with categorical features and as C++ code w/o categorical features.

## Bug Fixes and Other Changes
- Fix `_catboost` reinitialization issues #268 and #269.
- Python module `catboost.util` extended with `create_cd`. It creates column description file.
- Now it's possible to load titanic and amazon (Kaggle Amazon Employee Access Challenge) datasets from Python code. Use `catboost.datasets`.
- GPU parameter `use_cpu_ram_for_cat_features` renamed to `gpu_cat_features_storage` with posible values `CpuPinnedMemory` and `GpuRam`. Default is `GpuRam`.

## Thanks to our Contributors
This release contains contributions from CatBoost team.

As usual we are grateful to all who filed issues or helped resolve them, asked and answered questions.

# Release 0.7.2
## Major Features And Improvements
- GPU: New `DocParallel` mode for tasks without categorical features and or with categorical features and `—max-ctr-complextiy 1`. Provides best performance for pools with big number of documents.
- GPU: Distributed training on several GPU host via MPI. See instruction how to build binary [here](https://tech.yandex.com/catboost/doc/dg/concepts/cli-installation-docpage/#multi-node-installation).
- GPU: Up to 30% learning speed-up for Maxwell and later GPUs with binarization level > 32

## Bug Fixes and Other Changes
- Hotfixes for GPU version of python wrapper.

# Release 0.7.1
## Major Features And Improvements
- Python wrapper: added methods to download datasets titanic and amazon, to make it easier to try the library (`catboost.datasets`).
- Python wrapper: added method to write column desctiption file (`catboost.utils.create_cd`).
- Made improvements to visualization.
- Support non-numeric values in `GroupId` column.
- [Tutorials](https://github.com/catboost/tutorials/blob/master/README.md) section updated.

## Bug Fixes and Other Changes
- Fixed problems with eval_metrics (issue #285)
- Other fixes

# Release 0.7
## Breaking changes
- Changed parameter order in [`train()`](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_train-docpage/) function to be consistant with other GBDT libraries.
- `use_best_model` is set to True by default if `eval_set` labels are present.

## Major Features And Improvements
- New ranking mode [`YetiRank`](https://tech.yandex.com/catboost/doc/dg/concepts/loss-functions-docpage/#loss-functions__ranking) optimizes `NDGC` and `PFound`.
- New visualisation for `eval_metrics` and `cv` in Jupyter notebook.
- Improved per document feature importance.
- Supported `verbose`=`int`: if `verbose` > 1, `metric_period` is set to this value.
- Supported type(`eval_set`) = list in python. Currently supporting only single `eval_set`.
- Binary classification leaf estimation defaults are changed for weighted datasets so that training converges for any weights.
- Add `model_size_reg` parameter to control model size. Fix `ctr_leaf_count_limit` parameter, also to control model size.
- Beta version of distributed CPU training with only float features support.
- Add `subgroupId` to [Python](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_pool-docpage/)/[R-packages](https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost-load_pool-docpage/).
- Add groupwise metrics support in `eval_metrics`.

## Thanks to our Contributors
This release contains contributions from CatBoost team.

We are grateful to all who filed issues or helped resolve them, asked and answered questions.


# Release 0.6.3
## Breaking changes
- `boosting_type` parameter value `Dynamic` is renamed to `Ordered`.
- Data visualisation functionality in Jupyter Notebook requires ipywidgets 7.x+ now.
- `query_id` parameter renamed to `group_id` in Python and R wrappers.
- cv returns pandas.DataFrame by default if Pandas installed. See new parameter [`as_pandas`](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_cv-docpage/).

## Major Features And Improvements
- CatBoost build with make file. Now it’s possible to build command-line CPU version of CatBoost under Linux with [make file](https://tech.yandex.com/catboost/doc/dg/concepts/cli-installation-docpage/#make-install).
- In column description column name `Target` is changed to `Label`. It will still work with previous name, but it is recommended to use the new one.
- `eval-metrics` mode added into cmdline version. Metrics can be calculated for a given dataset using a previously [trained model](https://tech.yandex.com/catboost/doc/dg/concepts/cli-reference_eval-metrics-docpage/).
- New classification metric `CtrFactor` is [added](https://tech.yandex.com/catboost/doc/dg/concepts/loss-functions-docpage/).
- Load CatBoost model from memory. You can load your CatBoost model from file or initialize it from buffer [in memory](https://github.com/catboost/catboost/blob/master/catboost/CatboostModelAPI.md).
- Now you can run `fit` function using file with dataset: `fit(train_path, eval_set=eval_path, column_description=cd_file)`. This will reduce memory consumption by up to two times.
- 12% speedup for training.

## Bug Fixes and Other Changes
- JSON output data format is [changed](https://tech.yandex.com/catboost/doc/dg/concepts/output-data_training-log-docpage/).
- Python whl binaries with CUDA 9.1 support for Linux OS published into the release assets.
- Added `bootstrap_type` parameter to `CatBoostClassifier` and `Regressor` (issue #263).

## Thanks to our Contributors
This release contains contributions from newbfg and CatBoost team.

We are grateful to all who filed issues or helped resolve them, asked and answered questions.


# Release 0.6.2
## Major Features And Improvements
- **BETA** version of distributed mulit-host GPU via MPI training
- Added possibility to import coreml model with oblivious trees. Makes possible to migrate pre-flatbuffers model (with float features only) to current format (issue #235)
- Added QuerySoftMax loss function

## Bug Fixes and Other Changes
- Fixed GPU models bug on pools with both categorical and float features (issue #241)
- Use all available cores by default
- Fixed not querywise loss for pool with `QueryId`
- Default float features binarization method set to `GreedyLogSum`


# Release 0.6.1.1
## Bug Fixes and Other Changes
- Hotfix for critical bug in Python and R wrappers (issue #238)
- Added stratified data split in CV
- Fix `is_classification` check and CV for Logloss


# Release 0.6.1
## Bug Fixes and Other Changes
- Fixed critical bugs in formula evaluation code (issue #236)
- Added scale_pos_weight parameter

# Release 0.6
## Speedups
- 25% speedup of the model applier
- 43% speedup for training on large datasets.
- 15% speedup for `QueryRMSE` and calculation of querywise metrics.
- Large speedups when using binary categorical features.
- Significant (x200 on 5k trees and 50k lines dataset) speedup for plot and stage predict calculations in cmdline.
- Compilation time speedup.

## Major Features And Improvements
- Industry fastest [applier implementation](https://tech.yandex.com/catboost/doc/dg/concepts/c-plus-plus-api-docpage/#c-plus-plus-api).
- Introducing new parameter [`boosting-type`](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_parameters-list-docpage/) to switch between standard boosting scheme and dynamic boosting, described in paper ["Dynamic boosting"](https://arxiv.org/abs/1706.09516).
- Adding new [bootstrap types](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_parameters-list-docpage/) `bootstrap_type`, `subsample`. Using `Bernoulli` bootstrap type with `subsample < 1` might increase the training speed.
- Better logging for cross-validation, added [parameter](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_cv-docpage/) `logging_level` and `metric_period` (should be set in training parameters) to cv.
- Added a separate `train` [function](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_train-docpage/) that receives the parameters and returns a trained model.
- Ranking mode `QueryRMSE` now supports default settings for dynamic boosting.
- R-package pre-build binaries are included into release.
- We added many synonyms to our parameter names, now it is more convenient to try CatBoost if you are used to some other library.

## Bug Fixes and Other Changes
- Fix for CPU `QueryRMSE` with weights.
- Adding several missing parameters into wrappers.
- Fix for data split in querywise modes.
- Better logging.
- From this release we'll provide pre-build R-binaries
- More parallelisation.
- Memory usage improvements.
- And some other bug fixes.

## Thanks to our Contributors
This release contains contributions from CatBoost team.

We are grateful to all who filed issues or helped resolve them, asked and answered questions.

# Release 0.5.2

## Major Features And Improvements
- We've made single document formula applier 4 times faster!
- `model.shrink` function added in [Python](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_catboost_shrink-docpage/) and R wrappers.
- Added new [training parameter](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_parameters-list-docpage/) `metric_period` that controls output frequency.
- Added new ranking [metric](https://tech.yandex.com/catboost/doc/dg/concepts/loss-functions-docpage/) `QueryAverage`.
- This version contains an easy way to implement new user metrics in C++. How-to example [is provided](https://github.com/catboost/tutorials/blob/master/custom_loss/custom_metric_tutorial.md).

## Bug Fixes and Other Changes
- Stability improvements and bug fixes

As usual we are grateful to all who filed issues, asked and answered questions.

# Release 0.5

## Breaking Changes
Cmdline:
- Training parameter `gradient-iterations` renamed to `leaf-estimation-iterations`.
- `border` option removed. If you want to specify border for binary classification mode you need to specify it in the following way: `loss-function Logloss:Border=0.5`
- CTR parameters are changed:
   - Removed `priors`, `per-feature-priors`, `ctr-binarization`;
   - Added `simple-ctr`, `combintations-ctr`, `per-feature-ctr`;
   More details will be published in our documentation.

Python:
- Training parameter `gradient_iterations` renamed to `leaf_estimation_iterations`.
- `border` option removed. If you want to specify border for binary classification mode you need to specify it in the following way: `loss_function='Logloss:Border=0.5'`
- CTR parameters are changed:
   - Removed `priors`, `per_feature_priors`, `ctr_binarization`;
   - Added `simple_ctr`, `combintations_ctr`, `per_feature_ctr`;
   More details will be published in our documentation.

## Major Features And Improvements
- In Python we added a new method `eval_metrics`: now it's possible for a given model to calculate specified metric values for each iteration on specified dataset.
- One command-line binary for CPU and GPU: in CatBoost you can switch between CPU and GPU training by changing single parameter value `task-type CPU` or `GPU` (task_type 'CPU', 'GPU' in python bindings). Windows build still contains two binaries.
- We have speed up the training up to 30% for datasets with a lot of objects.
- Up to 10% speed-up of GPU implementation on Pascal cards

## Bug Fixes and Other Changes
- Stability improvements and bug fixes

As usual we are grateful to all who filed issues, asked and answered questions.

# Release 0.4

## Breaking Changes
FlatBuffers model format: new CatBoost versions wouldn’t break model compatibility anymore.

## Major Features And Improvements
* Training speedups: we have speed up the training by 33%.
* Two new ranking modes are [available](https://tech.yandex.com/catboost/doc/dg/concepts/loss-functions-docpage/#ranking):
  * `PairLogit` - pairwise comparison of objects from the input dataset. Algorithm maximises probability correctly reorder all dataset pairs.
  * `QueryRMSE` - mix of regression and ranking. It’s trying to make best ranking for each dataset query by input labels.

## Bug Fixes and Other Changes
* **We have fixed a bug that caused quality degradation when using weights < 1.**
* `Verbose` flag is now deprecated, please use `logging_level` instead. You could set the following levels: `Silent`, `Verbose`, `Info`, `Debug`.
* And some other bugs.

## Thanks to our Contributors
This release contains contributions from: avidale, newbfg, KochetovNicolai and CatBoost team.

We are grateful to all who filed issues or helped resolve them, asked and answered questions.

# Release 0.3

## Major Features And Improvements
GPU CUDA support is available. CatBoost supports multi-GPU training. Our GPU implementation is 2 times faster then LightGBM and more then 20 times faster then XGBoost one. Check out the news with benchmarks on our [site](https://catboost.yandex/news#version_0_3).

## Bug Fixes and Other Changes
Stability improvements and bug fixes

## Thanks to our Contributors
This release contains contributions from: daskol and CatBoost team.

We are grateful to all who filed issues or helped resolve them, asked and answered questions.

# Release 0.2

## Breaking Changes
* R library interface significantly changed
* New model format: CatBoost v0.2 model binary not compatible with previous versions
* Cross-validation parameters changes: we changed overfitting detector parameters of CV in python so that it is same as those in training.
* CTR types: MeanValue => BinarizedTargetMeanValue

## Major Features And Improvements
* Training speedups: we have speed up the training by 20-30%.
* Accuracy improvement with categoricals: we have changed computation of statistics for categorical features, which leads to better quality.
* New type of overfitting detector: `Iter`. This type of detector was requested by our users. So now you can also stop training by a simple criterion: if after a fixed number of iterations there is no improvement of your evaluation function.
* TensorBoard support: this is another way of looking on the graphs of different error functions both during training and after training has finished. To look at the metrics you need to provide `train_dir` when training your model and then run `"tensorboard --logdir={train_dir}"`
* Jupyter notebook improvements: for our Python library users that experiment with Jupyter notebooks, we have improved our visualisation tool. Now it is possible to save image of the graph. We also have changed scrolling behaviour so that it is more convenient to scroll the notebook.
* NaN features support: we also have added simple but effective way of dealing with NaN features. If you have some NaNs in the train set, they will be changed to a value that is less than the minimum value or greater than the maximum value in the dataset (this is configurable), so that it is guaranteed that they are in their own bin, and a split would separates NaN values from all other values. By default, no NaNs are allowed, so you need to use option `nan_mode` for that. When applying a model, NaNs will be treated in the same way for the features where NaN values were seen in train. It is not allowed to have NaN values in test if no NaNs in train for this feature were provided.
* Snapshotting: we have added snapshotting to our Python and R libraries. So if you think that something can happen with your training, for example machine can reboot, you can use `snapshot_file` parameter - this way after you restart your training it will start from the last completed iteration.
* R library tutorial: we have added [tutorial](https://github.com/catboost/tutorials/blob/master/r_tutorial.ipynb)
* Logging customization: we have added `allow_writing_files` parameter. By default some files with logging and diagnostics are written on disc, but you can turn it off using by setting this flag to False.
* Multiclass mode improvements: we have added a new objective for multiclass mode - `MultiClassOneVsAll`. We also added `class_names` param - now you don't have to renumber your classes to be able to use multiclass. And we have added two new metrics for multiclass: `TotalF1` and `MCC` metrics.
You can use the metrics to look how its values are changing during training or to use overfitting detection or cutting the model by best value of a given metric.
* Any delimeters support: in addition to datasets in `tsv` format, CatBoost now supports files with any delimeters

## Bug Fixes and Other Changes
Stability improvements and bug fixes

## Thanks to our Contributors
This release contains contributions from: grayskripko, hadjipantelis and CatBoost team.

We are grateful to all who filed issues or helped resolve them, asked and answered questions.
