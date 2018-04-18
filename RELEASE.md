# Release 0.8
## Breaking changes
- We fixed bug in CatBoost. Pool initialization from `numpy.array` and `pandas.dataframe` with string values that can cause slight inconsistence while using trained model from older versions. Around 1% of cat feature hashes were treated incorrectly. If you expirience quality drop after update you should consider retraining your model.

## Major Features And Improvements
- Algorithm for finding most influential training samples for a given object from the 'Finding Influential Training Samples for Gradient Boosted Decision Trees' [paper](https://arxiv.org/pdf/1802.06640.pdf) is implemented. This mode for every object from input pool calculates scores for every object from train pool. A positive score means that the given train object has made a negative contribution to the given test object prediction. And vice versa for negative scores. The higher score modulo - the higher contribution.
See `get_object_importance` model method in Python package and `ostr` mode in cli-version. Tutorial for Python is available [here](https://github.com/catboost/catboost/blob/master/catboost/tutorials/advanced_tutorials/catboost_object_importance_tutorial.ipynb).
More details and examples will be published in documentation soon.
- We have implemented new way of exploring feature importance - Shap values from [paper](https://arxiv.org/pdf/1706.06060.pdf). This allows to understand which features are most influent for a given object. You can also get more insite about your model, see details in a [tutorial](https://github.com/catboost/catboost/blob/master/catboost/tutorials/advanced_tutorials/shap_values_tutorial.ipynb).
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
- [Tutorials](https://github.com/catboost/catboost/blob/master/catboost/tutorials/README.md) section updated.

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
- This version contains an easy way to implement new user metrics in C++. How-to example [is provided](https://github.com/catboost/catboost/blob/master/catboost/tutorials/catboost_custom_loss_tutorial.md).

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
* R library tutorial: we have added [tutorial](https://github.com/catboost/catboost/blob/master/catboost/tutorials/catboost_r_tutorial.ipynb)
* Logging customization: we have added `allow_writing_files` parameter. By default some files with logging and diagnostics are written on disc, but you can turn it off using by setting this flag to False.
* Multiclass mode improvements: we have added a new objective for multiclass mode - `MultiClassOneVsAll`. We also added `class_names` param - now you don't have to renumber your classes to be able to use multiclass. And we have added two new metrics for multiclass: `TotalF1` and `MCC` metrics.
You can use the metrics to look how its values are changing during training or to use overfitting detection or cutting the model by best value of a given metric.
* Any delimeters support: in addition to datasets in `tsv` format, CatBoost now supports files with any delimeters

## Bug Fixes and Other Changes
Stability improvements and bug fixes

## Thanks to our Contributors
This release contains contributions from: grayskripko, hadjipantelis and CatBoost team.

We are grateful to all who filed issues or helped resolve them, asked and answered questions.
