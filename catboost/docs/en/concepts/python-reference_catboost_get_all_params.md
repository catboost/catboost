# get_all_params

{% include [get_all_params-python__get_all_params__desc__div](../_includes/work_src/reusage-python/python__get_all_params__desc__div.md) %}


Use the [get_params](python-reference_catboost_get_params.md) method to obtain only such parameters that are explicitly specified before the training

## {{ dl--invoke-format }} {#python__get_all_params__call-format}

```python
get_all_params()
```

## {{ dl--output-format }} {#python__get_all_params__output-format}

{{ python-type--dict }}

## {{ output--example }} {#example}

```python
from catboost import CatBoost

train_data = [[1, 4, 5, 6],
              [4, 5, 6, 7],
              [30, 40, 50, 60]]

eval_data = [[2, 4, 6, 8],
             [1, 4, 50, 60]]

train_labels = [10, 20, 30]

model = CatBoost()

model.fit(train_data,
          train_labels,
          verbose=False)

print(model.get_all_params())

```

The output of this example:

```bash
{u'bayesian_matrix_reg': 0.1000000015, u'leaf_estimation_backtracking': u'AnyImprovement', u'has_time': False, u'classes_count': 0, u'iterations': 1000, u'bagging_temperature': 1, u'task_type': u'CPU', u'sampling_frequency': u'PerTree', u'loss_function': u'RMSE', u'l2_leaf_reg': 3, u'border_count': 254, u'boosting_type': u'Ordered', u'bootstrap_type': u'Bayesian', u'rsm': 1, u'use_best_model': False, u'leaf_estimation_iterations': 1, u'random_seed': 0, u'eval_metric': u'RMSE', u'permutation_count': 4, u'learning_rate': 0.02999999933, u'fold_permutation_block': 0, u'best_model_min_trees': 1, u'approx_on_full_history': False, u'model_size_reg': 0.5, u'feature_border_type': u'GreedyLogSum', u'random_strength': 1, u'depth': 6, u'fold_len_multiplier': 2, u'nan_mode': u'Min', u'leaf_estimation_method': u'Newton'}
```

