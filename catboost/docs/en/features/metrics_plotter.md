# Metrics Plotter for Jupyter

The CatBoost library has a tool for plotting arbitrary learning curves. This tool triggers interactive widget in the Jupyter Notebook / JupyterLab interface.

[Additional packages for data visualization support](../installation/python-installation-additional-data-visualization-packages.md) must be installed to plot charts in [Jupyter Notebook](http://jupyter.org/).

## Features {#features}

- Clickable plots with learning curves.
- Remaining learning time estimation based on a total iteration count.
- Tracking of an iteration number with the best metric value.

## Usage {#usage}

```python
from catboost import MetricsPlotter

train_metrics = ["MSE", "MAE", "R2"]
test_metrics = ["MSE", "R2"]
iter_count = 10

with MetricPlotter(train_metrics, test_metrics, iter_count) as plotter:
    for i in range(iter_count):

        # TRAIN CODE: <...>

        plotter.log(
            i, train=True,
            metrics={
                "MAE": value1,
                "MSE": value2,
                "R2": value3,
            }
        )

        # EVALUATION CODE: <...>

        plotter.log(
            i, train=False,
            metrics={
                "MSE": value1,
                "R2": value2,
            }
        )
```

{% note info %}

Metrics should be passed to class successively, in a natural way described above.

{% endnote%}

To save plot in the notebook, use the `Widget` → `Save Widget State` menu option in the Jupyter Notebook interface.

`MetricsPlotter` constructor arguments:

Arguments | Type | Description
----- |  ----- | -----
`train_metrics` | `list` of `str` or `list` of `dict` | List of train metrics to be tracked. </br> Each item in the list can be either string with a metric name or a dict with the fields `name` and `best_value`, where the latter can be one of the following: `Max`, `Min`, `Undefined`.
`test_metrics` | `list` of `str` or `list` of `dict`, optional (default=`None`) | List of test metrics to be tracked. </br> Has the same format as `train_metrics`. Equals to `train_metrics` if it is not defined.
`total_iterations` | `int`, optional (default=`None`) | Total number of iterations. Allows to remain time estimation.

## Callbacks {#callbacks}

CatBoost learning curves widget can work with the models from [XGBoost](https://xgboost.readthedocs.io/en/stable/) and [LightGBM](https://lightgbm.readthedocs.io/en/latest/) by using callbacks.

See the examples below:

### XGBoost {#XGBoost}

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from catboost import XGBPlottingCallback

X, y = load_breast_cancer(return_X_y=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

num_boost_round = 100
```

- In `train()` method:

```python
D_train = xgb.DMatrix(X_train, y_train)
D_valid = xgb.DMatrix(X_valid, y_valid)

xgb.train(
    {
        "objective": "binary:logistic",
        "eval_metric": ["rmse", "error"],
    },
    D_train,
    # include train sample here for correct widget work:
    evals=[(D_train, "Train"), (D_valid, "Valid")],
    verbose_eval=False,
    num_boost_round=num_boost_round,
    callbacks=[XGBPlottingCallback(num_boost_round)])
```

- In `fit()` method:

```python
clf = xgb.XGBModel(
    objective="binary:logistic",
    n_estimators=num_boost_round
)

clf.fit(
    X_train, y_train,
    # first sample is considered as train below:
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_metric=["logloss", "rmse"],
    verbose=False,
    callbacks=[XGBPlottingCallback(num_boost_round)]
)
```

### LightGBM

```python
import lightgbm as lgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from catboost import lgbm_plotting_callback

housing = fetch_california_housing()

X_train, X_test, Y_train, Y_test = train_test_split(housing.data, housing.target)

n_estimators = 500
```

- In `train()` method:

```python
feature_names = housing.feature_names

train_dataset = lgb.Dataset(X_train, Y_train, feature_name=feature_names)
test_dataset = lgb.Dataset(X_test, Y_test, feature_name=feature_names)

booster = lgb.train(
    {"objective": "regression", "verbosity": -1},
    train_set=train_dataset,
    # include train sample here for correct widget work:
    valid_sets=(train_dataset, test_dataset),
    num_boost_round=n_estimators,
    verbose_eval=False,
    callbacks=[lgbm_plotting_callback()],
)
```

- In `fit()` method:

```python
booster = lgb.LGBMModel(objective="regression", n_estimators=n_estimators)

booster.fit(
    X_train, Y_train,
    # include train sample here for correct widget work:
    eval_set=[(X_train, Y_train), (X_test, Y_test)],
    eval_metric=["rmse", "mape"],
    verbose=False,
    callbacks=[lgbm_plotting_callback()]
)
```
