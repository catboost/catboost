# {{ dl--attributes }}

## tree_count_ {#tree_count_}

#### {{ dl--purpose }}

Return the number of trees in the model.

This number can differ from the value specified in the `--iterations` training parameter in the following cases:
- The training is stopped by the [overfitting detector](../concepts/overfitting-detector.md).
- The `--use-best-model` training parameter is set to <q>True</q>.

#### {{ python__params-table__title__type }}
{{ python-type--int }}


## feature_importances_ {#feature_importances_}

#### {{ dl--purpose }}

Return the calculated [feature importances](../concepts/fstr.md). The output data depends on the type of the model's loss function:
- Non-ranking loss functions — [{{ title__regular-feature-importance-PredictionValuesChange }}](../concepts/fstr.md#regular-feature-importance)
- Ranking loss functions — [{{ title__regular-feature-importance-LossFunctionChange }}](../concepts/fstr.md#regular-feature-importances__lossfunctionchange)

If the corresponding feature importance is not calculated the returned value is <q>None</q>.

Use the `` function to surely calculate the [{{ title__regular-feature-importance-LossFunctionChange }}](../concepts/fstr.md#regular-feature-importances__lossfunctionchange) feature importance.

#### {{ python__params-table__title__type }}
{{ python-type__np_ndarray }}


## random_seed_ {#random_seed_}

#### {{ dl--purpose }}

The random seed used for training.

#### {{ python__params-table__title__type }}

{{ python-type--int }}


## learning_rate_ {#learning_rate_}

#### {{ dl--purpose }}

The learning rate used for training.

#### {{ python__params-table__title__type }}
{{ python-type--float }}


## feature_names_ {#feature_names_}

#### {{ dl--purpose }}

The names of features in the dataset.

#### {{ python__params-table__title__type }}
{{ python-type--list }}


## evals_result_ {#eval_result_}

#### {{ dl--purpose }}

Return the values of metrics calculated during the training.

{% note info %}

Only the values of calculated metrics are output. The following metrics are not calculated by default for the training dataset and therefore these metrics are not output:

- {{ error-function__PFound }}
- {{ error-function__YetiRank }}
- {{ error-function__ndcg }}
- {{ error-function__YetiRankPairwise }}
- {{ error-function--AUC }}
- {{ error-function--NormalizedGini }}
- {{ error-function__FilteredDCG }}
- {{ error-function__dcg }}

Use the `hints=skip_train~false` parameter to enable the calculation. See the [Enable, disable and configure metrics calculation](../concepts/loss-functions.md#enable-disable-configure-metrics) section for more details.

{% endnote %}

#### {{ python__params-table__title__type }}

{{ python-type--dict }}

Output format:
```no-highlight
{pool_name: {metric_name_1-1: [value_1, value_2, .., value_N]}, .., {metric_name_1-M: [value_1, value_2, .., value_N]}}
```

For example:
```no-highlight
{'learn': {'Logloss': [0.6720840012056274, 0.6476800666988386, 0.6284055381249782], 'AUC': [1.0, 1.0, 1.0], 'CrossEntropy': [0.6720840012056274, 0.6476800666988386, 0.6284055381249782]}}
```


## best_score_ {#best_score}

#### {{ dl--purpose }}

{% include [sections-with-methods-desc-python__method__get_best_score__desc](../_includes/work_src/reusage/python__method__get_best_score__desc.md) %}


{% include [sections-with-methods-desc-only-the-calculated-metrics-are-output](../_includes/work_src/reusage/only-the-calculated-metrics-are-output.md) %}

#### {{ python__params-table__title__type }}

{{ python-type--dict }}

Output format:
```bash
{pool_name_1: {metric_1: value,..., metric_N: value}, ..., pool_name_M: {metric_1: value,..., metric_N: value}
```

For example:
```bash
{'validation': {'Logloss': 0.6085537606941837, 'AUC': 0.0}}
```


## best_iteration_ {#best_iteration}

#### {{ dl--purpose }}

{% include [sections-with-methods-desc-python__method__get_best_iteration__desc](../_includes/work_src/reusage/python__method__get_best_iteration__desc.md) %}


#### {{ python__params-table__title__type }}

{{ python-type--int }} or None if the validation dataset is not specified.

