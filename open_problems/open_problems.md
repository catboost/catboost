1. `treat_object_as_categorical`
Currently you have to pass `cat_features` to CatBoost* init function or to fit function.
Many people ask for automatic detection of categorical features.
This flag would solve the problem.
It is suggested to add the flag to Pool init function, CatBoost* init functions and to fit function the same way `cat_features` parameter is added.

Tests for all these cases must be provided.

2. `allow_float_categories`
Categorical features are treated in the following way. We first convert them to strings, then calculate hash from the string, then use the hash value in the algorithm.
For this reason it is only allowed to use data types that can be converted to string in a unique way. Otherwise if you are training from python and applying from C++, you might get different results because of different string representation.
But if you are only working from python, then it can be safe to use float numbers if user has explicitly confirmed that this is what the user wants.

This flag should also be used in Pool and CatBoost* init functions and in fit function.

3. `allow_nan_categories`
This problem is very similar to #2, but now nan categories are allowed.
It is suggested that in this case nan value is always converted to "None" string before calculating hashes.

4. Skip invalid parameter configurations in `grid_search` and `randomized_search` methods.
The python code of running parameter search should chech if configuration is valid. If it is not valid it should be skipped and a warning msg should be printed.
In case of `randomized_search`, where `n_iter` is number of checked configurations, invalid configurations should not be count as checked ones.

5. Add `model.class_count_` property to CatBoostClassifier class.
It should return `len(model.class_names_)`

6. Add `feature_names_`, `cat_feature_names_`, `num_feature_names_`, `cat_feature_indices_` properties to CatBoost* classes.

7. Implement a new ranking metric ERR (Expected Reciprocal Rank) and its documentation

8. Add CatBoostClassifier `predict_log_proba` method

9. Better parameter checks:
if `leaf_estimation_iterations`:5 with RMSE, there should be warning and 1 iteration

10. tutorial on poisson regression using monotonic1 dataset.
Jupyter notebook should give text explanation of what is the task, examples when it might appear and how it is solved.

11. In python cv request `loss_function` to be set
Currently if no `loss_function` is passed, then RMSE is used by default.
This might be misleading in the following case.
A user creates CatBoostClassifier and calles `get_params()` from a not trained model. The resulting parameters don't contain the `loss_function` parameter, because the default `loss_function` for CatBoostClassifier depends on number of classes in train dataset. If there are 2 classes, it is Logloss, if there are more than 2 classes, it is MultiClass.

These parameters are passed to cv function. And it trains an RMSE model, because it is the default loss.

This is not expected behavoir. So it is better to check that the loss is present among parameters passed to cv method.

12. Implement CatBoostRanker class
Currently we only have CatBoostRegressor and CatBoostClassifier.
It would be nice to implement a class for ranking also.
The default loss function in this case will be YetiRank.

13. Implement a ColumnDescription class in Python that can be used instead of cd file https://catboost.ai/docs/concepts/input-data_column-descfile.html
when creating Pool from file.
The class should have init function, methods load and save, and Pool init method should be able to use object of this class instead of cd file during initialization.

14. Add `eval_metrics` method to R library. Currently it's only supported in Python package.

15. Add `baseline` parameter to `eval_metrics` function in Python.
Currently this function assumes that initial value for every sample is 0.
This might be not the case, if we are traing from some baseline.

16. Automatic `class_weights`/`scale_pos_weight` based on training dataset class appearance frequency.
Interface: `class_weights`='Auto'

17. Add CatBoost to https://github.com/apple/turicreate

18. Implement Tweedie Regression
