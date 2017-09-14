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
* Jupyter notebook improvements: for our Python library users that experiment with Jupyter notebooks, we have improved our visualisation tool. Now it is possible to save image of the graph. We also have changed scrolling behavoir so that it is more convenient to scroll the notebook.
* NaN features support: we also have added simple but effective way of dealing with NaN features. If you have some NaNs in the train set, they will be changed to a value that is less than the minimum value or greater than the maximum value in the dataset (this is configurable), so that it is guaranteed that they are in their own bin, and a split would separates NaN values from all other values. By default, no NaNs are allowed, so you need to use option `nan_mode` for that. When applying a model, NaNs will be treated in the same way for the features where NaN values were seen in train. It is not allowed to have NaN values in test if no NaNs in train for this feature were provided.
* Snapshotting: we have added snapshotting to our Python and R libraries. So if you think that something can happen with your training, for example machine can reboot, you can use `snapshot_file` parameter - this way after you restart your training it will start from the last completed iteration.
* R library tutorial: we have added [tutorial] (https://github.com/catboost/catboost/blob/master/catboost/tutorials/catboost_r_tutorial.ipynb)
* Logging customization: we have added `allow_writing_files` parameter. By default some files with logging and diagnostics are written on disc, but you can turn it off using by setting this flag to False.
* Multiclass mode improvements: we have added a new objective for multiclass mode - `MultiClassOneVsAll`. We also added `class_names` param - now you don't have to renumber your classes to be able to use multiclass. And we have added two new metrics for multiclass: `TotalF1` and `MCC` metrics.
You can use the metrics to look how its values are changing during training or to use overfitting detection or cutting the model by best value of a given metric.
* Any delimeters support: in addition to datasets in `tsv` format, CatBoost now supports files with any delimeters

## Bug Fixes and Other Changes
* Stability improvements and bug fixes

## Thanks to our Contributors
This release contains contributions from: grayskripko, hadjipantelis and CatBoost team.

We are grateful to all who filed issues or helped resolve them, asked and answered questions.