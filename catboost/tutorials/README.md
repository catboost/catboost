# CatBoost tutorials

## Basic

It's better to start CatBoost exploring from this basic tutorials.

### Python

* [Python Tutorial](python_tutorial.ipynb)
    * This tutorial shows some base cases of using CatBoost, such as model training, cross-validation and predicting, as well as some useful features like early stopping,  snapshot support, feature importances and parameters tuning.
* [Python Tutorial with task](python_tutorial_with_tasks.ipynb)
    * There are 17 questions in this tutorial. Try answering all of them, this will help you to learn how to use the library.

### R

* [R Tutorial](r_tutorial.ipynb)
    * This tutorial shows how to convert your data to CatBoost Pool, how to train a model and how to make cross validation and parameter tunning.

### Command line

* [Command Line Tutorial](cmdline_tutorial/cmdline_tutorial.md)
    * This tutorial shows how to train and apply model with the command line tool.

## Classification

* [Classification Tutorial](classification/classification_tutorial.ipynb)
    * Here is an example for CatBoost to solve binary classification and multi-classification problems.

## Ranking
* [Ranking Tutorial](ranking/ranking_tutorial.ipynb)
    * CatBoost is learning to rank on Microsoft dataset (msrank).

## Feature selection
* [Feature selection Tutorial](feature_selection/eval_tutorial.ipynb)
    * This tutorial shows how to make feature evaluation with CatBoost and explore learning rate.

## Model analysis

* [Object Importance Tutorial](model_analysis/object_importance_tutorial.ipynb)
    * This tutorial shows how to evaluate importances of the train objects for test objects, and how to detect broken train objects by using the importance scores.

* [SHAP Values Tutorial](model_analysis/shap_values_tutorial.ipynb)
    * This tutorial shows how to use [SHAP](https://github.com/slundberg/shap) python-package to get and visualize feature importances.

* [Export CatBoost Model in JSON format Tutorial](model_analysis/model_export_as_json_tutorial.ipynb)
    * This tutorial shows how to save catboost model in JSON format and apply it.

* [Visualization of CatBoost decision trees tutorial](model_analysis/visualize_decision_trees_tutorial.ipynb)
    * This tutorial shows how to visualize catboost decision trees.

* [Feature statistics tutorial](model_analysis/feature_statistics_tutorial.ipynb)
    * This tutorial shows how to calculate feature statistics for catboost model.

* [CatBoost PredictionDiff Feature Importance Tutorial](./prediction_diff_feature_importance_tutorial.ipynb)
    * This tutorials shows how to use PredictionDiff feature importances.

## Custom loss

* [Custom Metrics Tutorial](custom_loss/custom_metric_tutorial.md)
    * This tutorial shows how to add custom per-object metrics.

## Apply model

* [CatBoost CoreML Tutorial](apply_model/coreml/coreml_export_tutorial.ipynb)
    * Explore this tutorial to learn how to convert CatBoost model to CoreML format and use it on any iOS device.

* [Export CatBoost Model as C++ code Tutorial](apply_model/model_export_as_cpp_code_tutorial.md)
    * Catboost model could be saved as standalone C++ code.

* [Export CatBoost Model as Python code Tutorial](apply_model/model_export_as_python_code_tutorial.md)
    * Catboost model could be saved as standalone Python code.

* [Apply CatBoost model from Java](apply_model/java/train_model.ipynb)
    * Explore how to apply CatBoost model from Java application. If you just want to look at code snippets you can go directly to [CatBoost4jPredictionTutorial.java](apply_model/java/src/main/java/CatBoost4jPredictionTutorial.java)

* [Apply CatBoost model from Rust](apply_model/rust/train_model.ipynb)
    * Explore how to apply CatBoost model from Rust application. If you just want to look at code snippets you can go directly to [main.rs](apply_model/rust/src/main.rs)

* [Convert LightGBM to CatBoost to use CatBoost fast appliers](apply_model/fast_light_gbm_applier.ipynb)
    * Convert LightGBM to CatBoost, save resulting CatBoost model and use CatBoost C++, Python, C# or other applier, which in case of not symmetric trees will be around 7-10 faster than native LightGBM one.
    * Note that CatBoost applier with CatBoost models is even faster, because it uses specific fast symmetric trees.

## Tools

* [Gradient Boosting: CPU vs GPU](tools/google_colaboratory_cpu_vs_gpu_tutorial.ipynb)
    * This is a basic tutorial which shows how to run gradient boosting on CPU and GPU on Google Colaboratory.

* [Regression on Gradient Boosting: CPU vs GPU](tools/google_colaboratory_cpu_vs_gpu_regression_tutorial.ipynb)
    * This is a basic tutorial which shows how to run regression on gradient boosting on CPU and GPU on Google Colaboratory.

## Competition examples

* [Kaggle Paribas Competition Tutorial](competition_examples/kaggle_paribas.ipynb)
    * This tutorial shows how to get to a 9th place on Kaggle Paribas competition with only few lines of code and training a CatBoost model.

* [ML Boot Camp V Competition Tutorial](competition_examples/mlbootcamp_v_tutorial.ipynb)
    * This is an actual 7th place solution by Mikhail Pershin. Solution is very simple and is based on CatBoost.

* [CatBoost & TensorFlow Tutorial](competition_examples/quora_w2v.ipynb)
    * This tutorial shows how to use CatBoost together with TensorFlow on Kaggle Quora Question Pairs competition if you have text as input data.

## Events

* [PyData Moscow tutorial](events/pydata_moscow_oct_13_2018.ipynb)
    * Tutorial from PyData Moscow, October 13, 2018.

* [PyData NYC tutorial](events/pydata_nyc_oct_19_2018.ipynb)
    * Tutorial from PyData New York, October 19, 2018.

* [PyData LA tutorial](events/pydata_la_oct_21_2018.ipynb)
    * Tutorial from PyData Los Angeles, October 21, 2018.

* [PyData Moscow tutorial](events/datastart_moscow_apr_27_2019.ipynb)
    * Tutorial from PyData Moscow, April 27, 2019.

* [PyData London tutorial](events/2019_pydata_london/pydata_london_2019.ipynb)
    * Tutorial from PyData London, June 15, 2019.

* [PyData Boston tutorial](events/2019_odsc_east/odsc_east_2019.ipynb)
    * Tutorial from PyData Boston, April 30, 2019.

## Tutorials in Russian

* Find tutorials in Russian on the separate [page](ru/README.md).
