# get_all_params

{% include [get_all_params-python__get_all_params__desc__div](../_includes/work_src/reusage-python/python__get_all_params__desc__div.md) %}


Use the [get_params](python-reference_catboostclassifier_get_params.md) method to obtain only such parameters that are explicitly specified before the training

## {{ dl--invoke-format }} {#python__get_all_params__call-format}

```python
get_all_params()
```

## {{ dl--output-format }} {#python__get_all_params__output-format}

{{ python-type--dict }}

## {{ output--example }} {#example}

```
from catboost import CatBoostClassifier, Pool

train_data = [[0, 3],
              [4, 1],
              [8, 1],
              [9, 1]]

train_labels = [0, 0, 1, 1]

eval_data = [[2, 1],
             [3, 1],
             [9, 0],
             [5, 3]]

eval_labels = [0, 1, 1, 0]

eval_dataset = Pool(eval_data,
                    eval_labels)

model = CatBoostClassifier(eval_metric='AUC')

model.fit(train_data,
          train_labels,
          eval_set=eval_dataset,
          verbose=False)

print(model.get_all_params())

```

The output of this example:
```bash
{u'bayesian_matrix_reg': 0.10000000149011612, u'leaf_estimation_backtracking': u'AnyImprovement', u'has_time': False, u'min_data_in_leaf': 1, u'model_shrink_rate': 0, u'classes_count': 0, u'max_leaves': 64, u'iterations': 1000, u'boosting_type': u'Plain', u'task_type': u'CPU', u'sampling_frequency': u'PerTree', u'loss_function': u'Logloss', u'l2_leaf_reg': 3, u'border_count': 254, u'bootstrap_type': u'MVS', u'rsm': 1, u'use_best_model': True, u'leaf_estimation_iterations': 10, u'random_seed': 0, u'eval_metric': u'AUC', u'permutation_count': 4, u'learning_rate': 0.008101999759674072, u'nan_mode': u'Min', u'score_function': u'Cosine', u'fold_permutation_block': 0, u'best_model_min_trees': 1, u'model_shrink_mode': u'Constant', u'approx_on_full_history': False, u'boost_from_average': False, u'model_size_reg': 0.5, u'grow_policy': u'SymmetricTree', u'random_strength': 1, u'subsample': 1, u'depth': 6, u'fold_len_multiplier': 2, u'feature_border_type': u'GreedyLogSum', u'leaf_estimation_method': u'Newton', u'class_names': [0, 1]}
```

