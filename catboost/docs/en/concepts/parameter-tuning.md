# Parameter tuning

{{ product }} provides a flexible interface for parameter tuning and can be configured to suit different tasks.

This section contains some tips on the possible parameter settings.

## One-hot encoding {#one-hot-enc}

{% include [parameter-tuning-one-hot-encoding__note](../_includes/work_src/reusage-common-phrases/one-hot-encoding__note.md) %}


Sometimes when categorical features don't have a lot of values, one-hot encoding works well.

Usually one-hot encoding does not significantly improve the quality of the model. But if it is required, use the inbuilt parameters instead of preprocessing the dataset.


{% cut "{{ dl--parameters }}" %}

  {% if audience == "internal" %}

  {% include [parameter-tuning-one-hot-encoding-features__internal](../yandex_specific/_includes/one-hot-encoding-features__internal.md) %}

  {% endif %}

  {% if audience == "external" %}

  {% include [parameter-tuning-one-hot-encoding-features](../_includes/work_src/reusage-common-phrases/one-hot-encoding-features.md) %}

  {% endif %}

{% endcut %}


## Number of trees {#trees-number}

It is recommended to check that there is no obvious underfitting or overfitting before tuning any other parameters. In order to do this it is necessary to analyze the metric value on the validation dataset and select the appropriate number of iterations.

This can be done by setting the number of [iterations](../references/training-parameters/common.md#iterations) to a large value, using the [overfitting detector](../concepts/overfitting-detector.md) parameters and turning the [use best model](../references/training-parameters/common.md#use_best_model) options on. In this case the resulting model contains only the first `k` best iterations, where `k` is the iteration with the best loss value on the validation dataset.

Also, the metric for choosing the best model may differ from the one used for optimizing the objective value. For example, it is possible to set the optimized function to {{ error-function--Logit }} and use the {{ error-function--AUC }} function for the overfitting detector. To do so, use the [evaluation metric](../references/training-parameters/common.md#eval_metric) parameter.


{% cut "{{ dl--parameters }}" %}

**Command-line version parameters:** `-i`, `--iterations`

**Python parameters:** `--iterations`

**R parameters:** `--iterations`

#### Description

 The maximum number of trees that can be built when solving machine learning problems.

When using other parameters that limit the number of iterations, the final number of trees may be less than the number specified in this parameter.


**Command-line version parameters:** `--use-best-model`

**Python parameters:** `--use-best-model`

**R parameters:** `--use-best-model`

#### Description

 If this parameter is set, the number of trees that are saved in the resulting model is defined as follows:
1. Build the number of trees defined by the training parameters.
1. Use the validation dataset to identify the iteration with the optimal value of the metric specified in  `--eval-metric` (`--eval-metric`).

No trees are saved after this iteration.

This option requires a validation dataset to be provided.


**Command-line version parameters:** `--eval-metric`

**Python parameters:** `--eval-metric`

**R parameters:** `--eval-metric`

#### Description

 The metric used for overfitting detection (if enabled) and best model selection (if enabled). Some metrics support optional parameters (see the [Objectives and metrics](../concepts/loss-functions.md) section for details on each metric).

Format:
```
<Metric>[:<parameter 1>=<value>;..;<parameter N>=<value>]
```

[Supported metrics](../references/eval-metric__supported-metrics.md)

Examples:
```
R2
```

```
Quantile:alpha=0.3
```

**Command-line version parameters:** **Overfitting detection settings**

**Command-line version parameters:** `--od-type`

**Python parameters:** `od_type`

**R parameters:** `od_type`

#### Description

 The type of the overfitting detector to use.

Possible values:
- {{ fit--od-type-inctodec }}
- {{ fit--od-type-iter }}


**Command-line version parameters:** `--od-pval`

**Python parameters:** `od_pval`

**R parameters:** `od_pval`

#### Description

 The threshold for the {{ fit--od-type-inctodec }} [overfitting detector](../concepts/overfitting-detector.md) type. The training is stopped when the specified value is reached. Requires that a validation dataset was input.

For best results, it is recommended to set a value in the range $[10^{–10}; 10^{-2}]$.

The larger the value, the earlier overfitting is detected.

{% note alert %}

Do not use this parameter with the {{ fit--od-type-iter }} overfitting detector type.

{% endnote %}

**Command-line version parameters:** `--od-wait`

**Python parameters:** `od_wait`

**R parameters:** `od_wait`

#### Description

 The number of iterations to continue the training after the iteration with the optimal metric value.
The purpose of this parameter differs depending on the selected overfitting detector type:
- {{ fit--od-type-inctodec }} — Ignore the overfitting detector when the threshold is reached and continue learning for the specified number of iterations after the iteration with the optimal metric value.
- {{ fit--od-type-iter }} — Consider the model overfitted and stop training after the specified number of iterations since the iteration with the optimal metric value.

{% endcut %}


## Learning rate {#learning-rate}

This setting is used for reducing the gradient step. It affects the overall time of training: the smaller the value, the more iterations are required for training. Choose the value based on the performance expectations.

By default, the learning rate is defined automatically based on the dataset properties and the number of iterations. The automatically defined value should be close to the optimal one.

Possible ways of adjusting the learning rate depending on the overfitting results:
- There is no overfitting on the last iterations of training (the training does not converge) — increase the learning rate.
- Overfitting is detected — decrease the learning rate.

{% cut "{{ dl--parameters }}" %}

**Command-line version parameters:** `-w`, `--learning-rate`

**Python parameters:** `learning_rate`

**R parameters:** `learning_rate`

#### Description

 The learning rate. Used for reducing the gradient step.

{% endcut %}


## Tree depth {#tree-depth}

In most cases, the optimal depth ranges from 4 to 10. Values in the range from 6 to 10 are recommended.

{% note info %}

The maximum depth of the trees is limited to {{ fit__maxtree__pairwise }} for pairwise modes ({{ error-function__YetiRank }}, {{ error-function__PairLogitPairwise }} and {{ error-function__QueryCrossEntropy }}) when the training is performed on GPU.

{% endnote %}


{% cut "{{ dl--parameters }}" %}

**Command-line version parameters:** `-n`, `--depth`

**Python parameters:** `depth`

**R parameters:** `depth`

#### Description

 Depth of the trees. The range of supported values depends on the processing unit type and the type of the selected loss function:
- CPU — Any integer up to  {{ fit--maxtree }}.

- GPU — Any integer up to {{ fit__maxtree__pairwise }} pairwise modes ({{ error-function__YetiRank }}, {{ error-function__PairLogitPairwise }} and {{ error-function__QueryCrossEntropy }}) and up to   {{ fit--maxtree }} for all other loss functions.

{% endcut %}


## L2 regularization {#l2-reg}

Try different values for the regularizer to find the best possible.

{% cut "{{ dl--parameters }}" %}

**Command-line version parameters:** `--l2-leaf-reg`

**Python parameters:** `l2_leaf_reg`

**R parameters:** `l2_leaf_reg`

#### Description

 Coefficient at the L2 regularization term of the cost function.
Any positive value is allowed.

{% endcut %}


## Random strength {#rand-str}

Try setting different values for the `random_strength` parameter.

{% cut "{{ dl--parameters }}" %}

**Command-line version parameters:** `--random-strength`

**Python parameters:** `random_strength`

**R parameters:** `random_strength`

#### Description

 The amount of randomness to use for scoring splits when the tree structure is selected. Use this parameter to avoid overfitting the model.

The value of this parameter is used when selecting splits. On every iteration each possible split gets a score (for example, the score indicates how much adding this split will improve the loss function for the training dataset). The split with the highest score is selected.

The scores have no randomness. A normally distributed random variable is added to the score of the feature. It has a zero mean and a variance that decreases during the training. The value of this parameter is the multiplier of the variance.
{% note info %}

This parameter is not supported for the following loss functions:
- {{ error-function__QueryCrossEntropy }}
- {{ error-function__YetiRankPairwise }}
- {{ error-function__PairLogitPairwise }}

{% endnote %}

{% endcut %}


## Bagging temperature {#bagg-temp}

Try setting different values for the `bagging_temperature` parameter


{% cut "{{ dl--parameters }}" %}

**Command-line version parameters:** `--bagging-temperature`

**Python parameters:** `bagging_temperature`

**R parameters:** `bagging_temperature`

#### Description

 Defines the settings of the Bayesian bootstrap. It is used by default in classification and regression modes.

Use the Bayesian bootstrap to assign random weights to objects.

The weights are sampled from exponential distribution if the value of this parameter is set to <q>1</q>. All weights are equal to 1 if the value of this parameter is set to <q>0</q>.

Possible values are in the range $[0; \inf)$. The higher the value the more aggressive the bagging is.

This parameter can be used if the selected bootstrap type is {{ fit__bootstrap-type__Bayesian }}.

{% endcut %}


## Border count {#border-count}

The number of splits for numerical features.

{% include [reusage-default-values-border_count](../_includes/work_src/reusage-default-values/border_count.md) %}

The value of this parameter significantly impacts the speed of training on GPU. The smaller the value, the faster the training is performed (refer to the [Number of splits for numerical features](speed-up-training.md) section for details).

128 splits are enough for many datasets. However, try to set the value of this parameter to 254 when training on GPU if the best possible quality is required.

{% include [parameter-tuning-border-count__how-affects-the-speed-cpu](../_includes/work_src/reusage-common-phrases/border-count__how-affects-the-speed-cpu.md) %}


{% cut "{{ dl--parameters }}" %}

**Command-line version parameters:** `-x`, `--border-count`

**Python parameters:** `border_count`

_Alias:_`max_bin`

**R parameters:** `border_count`

#### Description

 Recommended values are up to 255. Larger values slow down the training.

{% include [reusage-cli__border-count__desc](../_includes/work_src/reusage/cli__border-count__desc.md) %}

{% endcut %}


## Internal dataset order {#internal-dataset-order}

Use this option if the objects in your dataset are given in the required order. In this case, random permutations are not performed during the [Transforming categorical features to numerical features](../concepts/algorithm-main-stages_cat-to-numberic.md) and [Choosing the tree structure](../concepts/algorithm-main-stages_choose-tree-structure.md) stages.


{% cut "{{ dl--parameters }}" %}

**Command-line version parameters:** `--has-time`

**Python parameters:** `--has-time`

**R parameters:** `--has-time`

#### Description

 Use the order of objects in the input data (do not perform random permutations during the [Transforming categorical features to numerical features](../concepts/algorithm-main-stages_cat-to-numberic.md) and [Choosing the tree structure](../concepts/algorithm-main-stages_choose-tree-structure.md) stages).

The {{ cd-file__col-type__Timestamp }} column type is used to determine the order of objects if specified in the [input data](../concepts/input-data.md).

{% endcut %}


## Tree growing policy {#tree-growing-policy}

By default, {{ product }} uses symmetric trees, which are built if the growing policy is set to {{ growing_policy__SymmetricTree }}.

Such trees are built level by level until the specified depth is reached. On each iteration, all leaves from the last tree level are split with the same condition. The resulting tree structure is always symmetric.

Symmetric trees have a very good prediction speed (roughly 10 times faster than non-symmetric trees) and give better quality in many cases.

However, in some cases, other tree growing strategies can give better results than growing symmetric trees.

Try to analyze the results obtained with different growing trees strategies.

Specifics: Symmetric trees, that are used by default, can be applied much faster (up to 10 times faster).


{% cut "{{ dl--parameters }}" %}

**Command-line version parameters:** `--grow-policy`

**Python parameters:** `grow_policy`

**R parameters:** `grow_policy`

#### Description

 The tree growing policy. Defines how to perform greedy tree construction.

Possible values:
- {{ growing_policy__SymmetricTree }} — A tree is built level by level until the specified depth is reached. On each iteration, all leaves from the last tree level are split with the same condition. The resulting tree structure is always symmetric.
- {{ growing_policy__Depthwise }} — A tree is built level by level until the specified depth is reached. On each iteration, all non-terminal leaves from the last tree level are split. Each leaf is split by condition with the best loss improvement.

    {% note info %}

    Models with this growing policy can not be analyzed using the {{ title__predictiondiff }} feature importance and can be exported only to {{ fit__model-format_json }} and {{ fitpython__model-format_cbm }}.

    {% endnote %}

- {{ growing_policy__Lossguide }} — A tree is built leaf by leaf until the specified maximum number of leaves is reached. On each iteration, non-terminal leaf with the best loss improvement is split.

    {% include [reusage-cli__grow-policy__note](../_includes/work_src/reusage/cli__grow-policy__note.md) %}


**Command-line version parameters:** `--min-data-in-leaf`

**Python parameters:** `min_data_in_leaf`

_Alias:_`min_child_samples`

**R parameters:** `min_data_in_leaf`

#### Description

 The minimum number of training samples in a leaf. {{ product }} does not search for new splits in leaves with samples count less than the specified value.
Can be used only with the {{ growing_policy__Lossguide }} and {{ growing_policy__Depthwise }} growing policies.


**Command-line version parameters:** `--max-leaves`

**Python parameters:** `max_leaves`

_Alias:_`num_leaves`

**R parameters:** `max_leaves`

#### Description

 The maximum number of leafs in the resulting tree. Can be used only with the {{ growing_policy__Lossguide }} growing policy.

{% note info %}

It is not recommended to use values greater than 64, since it can significantly slow down the training process.

{% endnote %}

{% endcut %}


## Golden features {#golden-features}

If the dataset has a feature, which is a strong predictor of the result, the pre-quantisation of this feature may decrease the information that the model can get from it. It is recommended to use an increased number of borders (1024) for this feature.

{% note info %}

An increased number of borders should not be set for all features. It is recommended to set it for one or two golden features.

{% endnote %}

{% list tabs %}

- Command-line

    Parameter | Description
    ----- | -----
    `--per-float-feature-quantization` | A semicolon separated list of quantization descriptions.<br> Format: <br> ```FeatureId[:border_count=BorderCount][:nan_mode=BorderType][:border_type=border_selection_method]```|


    Examples:

    {% include [reusage-cli__per-float-feature-quantization__desc_example](../_includes/work_src/reusage/cli__per-float-feature-quantization__desc_example.md) %}

- Python

    Parameter | Description
    ----- | -----
    `per_float_feature_quantization` | The quantization description for the specified feature or list of features.<br> Description format for a single feature: <br>```FeatureId[:border_count=BorderCount][:nan_mode=BorderType][:border_type=border_selection_method]```|

    Examples:

     {% include [python-python__per-float-feature-quantization__desc__example](../_includes/work_src/reusage/python__per-float-feature-quantization__desc__example.md) %}

- R

    Parameter | Description
    ----- | -----
    `per_float_feature_quantization` | The quantization description for the specified feature or list of features.<br> Description format for a single feature: <br>```FeatureId[:border_count=BorderCount][:nan_mode=BorderType][:border_type=border_selection_method]```|


    Examples:

    {% include [parameters-r__per_float_feature_quantization__desc-examples__div](../_includes/work_src/reusage/r__per_float_feature_quantization__desc-examples__div.md) %}


{% endlist %}

## Methods for hyperparameter search {#defining-optimal-parameter-values}

The {{ python-package }} provides Grid and Randomized search methods for searching optimal parameter values for training the model with the given dataset.

{% cut "{{ dl--parameters }}" %}


Class | Method | Description
----- | ----- | -----
[CatBoost](python-reference_catboost.md) | [grid_search](python-reference_catboost_grid_search.md) | A simple grid search over specified parameter values for a model.|
[CatBoost](python-reference_catboost.md) | [randomized_search](python-reference_catboost_randomized_search.md) | A simple randomized search on hyperparameters. |
[CatBoostClassifier](python-reference_catboostclassifier.md) | [grid_search](python-reference_catboostclassifier_grid_search.md) | A simple grid search over specified parameter values for a model.|
[CatBoostClassifier](python-reference_catboostclassifier.md) | [randomized_search](python-reference_catboostclassifier_randomized_search.md) | A simple randomized search on hyperparameters.
[CatBoostRegressor](python-reference_catboostregressor.md) | [grid_search](python-reference_catboostregressor_grid_search.md) | A simple grid search over specified parameter values for a model. |
[CatBoostRegressor](python-reference_catboostregressor.md) | [randomized_search](python-reference_catboostregressor_randomized_search.md) | A simple randomized search on hyperparameters. |

{% endcut %}

## Methods for hyperparameter search by optuna

Optuna is a famous hyperparameter optimization framework.
Optuna enables efficient hyperparameter optimization by adopting state-of-the-art algorithms for sampling hyperparameters and pruning efficiently unpromising trials.
Catboost supports to stop unpromising trial of hyperparameter by callbacking after iteration functionality. [Pull Request](https://github.com/catboost/catboost/pull/1697/files#diff-ccca44461ac6b094190f29fec157a227996e226ea483213680dd0a152cd412eaR9679)

The following is an optuna example that demonstrates a pruner for CatBoost. [Example](https://github.com/optuna/optuna-examples/blob/main/catboost/catboost_pruning.py)
