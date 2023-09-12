# Common parameters

## loss_function {#loss_function}

Command-line: `--loss-function`

_Alias:_ `objective`

#### Description

{% include [reusage-loss-function-short-desc](../../_includes/work_src/reusage/loss-function-short-desc.md) %}


{% include [reusage-loss-function-format](../../_includes/work_src/reusage/loss-function-format.md) %}


{% cut "Supported metrics" %}

- {{ error-function--RMSE }}
- {{ error-function--Logit }}
- {{ error-function--MAE }}
- {{ error-function--CrossEntropy }}
- {{ error-function--Quantile }}
- {{ error-function--LogLinQuantile }}
- {{ error-function__lq }}
- {{ error-function__MultiRMSE }}
- {{ error-function--MultiClass }}
- {{ error-function--MultiClassOneVsAll }}
- {{ error-function__MultiLogloss }}
- {{ error-function__MultiCrossEntropy }}
- {{ error-function--MAPE }}
- {{ error-function--Poisson }}
- {{ error-function__PairLogit }}
- {{ error-function__PairLogitPairwise }}
- {{ error-function__QueryRMSE }}
- {{ error-function__QuerySoftMax }}
- {{ error-function__Tweedie }}

- {{ error-function__YetiRank }}
- {{ error-function__YetiRankPairwise }}
- {{ error-function__StochasticFilter }}
- {{ error-function__StochasticRank }}

{% endcut %}

A custom python object can also be set as the value of this parameter (see an [example](../../concepts/python-usages-examples.md)).

{% include [reusage-loss-function--example](../../_includes/work_src/reusage/loss-function--example.md) %}

**Type**

- {{ python-type--string }}
- object

**Default value**

{% cut "Python package" %}

Depends on the class:

- [CatBoostClassifier](../../concepts/python-reference_catboostclassifier.md): {{ error-function--Logit }} if the `target_border` parameter value differs from None. Otherwise, the default loss function depends on the number of unique target values and is either set to {{ error-function--Logit }} or {{ error-function--MultiClass }}.
- [CatBoost](../../concepts/python-reference_catboost.md) and [CatBoostRegressor](../../concepts/python-reference_catboostregressor.md): {{ error-function--RMSE }}

{% endcut %}

{% cut "R package, Command-line" %}

RMSE

{% endcut %}

**Supported processing units**

 {{ cpu-gpu }}

## custom_metric {#custom_metric}

Command-line: `--custom-metric`

#### Description

{% include [reusage-custom-loss--basic](../../_includes/work_src/reusage/custom-loss--basic.md) %}

{% include [reusage-loss-function-format](../../_includes/work_src/reusage/loss-function-format.md) %}


[Supported metrics](../../references/custom-metric__supported-metrics.md)

{% cut "Examples" %}

- Calculate the value of {{ error-function--CrossEntropy }}:

    ```
    {{ error-function--CrossEntropy }}
    ```

- Calculate the value of {{ error-function--Quantile }} with the coefficient $\alpha = 0.1$
    ```
    {{ error-function--Quantile }}:alpha=0.1
    ```

- Calculate the values of {{ error-function--Logit }} and {{ error-function--AUC }}:
    ```python
    ['{{ error-function--Logit }}', '{{ error-function--AUC }}']


    ```
{% endcut %}

{% include [reusage-custom-loss--values-saved-to](../../_includes/work_src/reusage/custom-loss--values-saved-to.md) %}


Use the [visualization tools](../../features/visualization.md) to see a live chart with the dynamics of the specified metrics.

**Type**

- {{ python-type--string }}
- {{ python-type--list-of-strings }}

**Default value**

{% cut "Python package" %}

{{ fit--custom_loss }}

{% endcut %}


{% cut "R package" %}

{{ fit--custom_loss }}

{% endcut %}


{% cut "Command-line" %}

None (do not output additional metric values)

{% endcut %}

**Supported processing units**

 {{ cpu-gpu }}


## eval_metric {#eval_metric}

Command-line: `--eval-metric`

#### Description

{% include [reusage-eval-metric--basic](../../_includes/work_src/reusage/eval-metric--basic.md) %}

{% include [reusage-eval-metric--format](../../_includes/work_src/reusage/eval-metric--format.md) %}

[Supported metrics](../../references/eval-metric__supported-metrics.md)

A user-defined function can also be set as the value (see an [example](../../concepts/python-usages-examples.md)).

{% include [reusage-eval-metric--examples](../../_includes/work_src/reusage/eval-metric--examples.md) %}

**Type**

- {{ python-type--string }}
- {{ python-type__object }}

**Default value**

  {{ fit--eval-metric }}

**Supported processing units**

 {{ cpu-gpu }}

## iterations {#iterations}

Command-line: `-i`, `--iterations`

_Aliases:_ `num_boost_round`, `n_estimators`, `num_trees`

#### Description

The maximum number of trees that can be built when solving machine learning problems.

When using other parameters that limit the number of iterations, the final number of trees may be less than the number specified in this parameter.

**Type**

{{ python-type--int }}

**Default value**

 {{ fit--iterations }}

**Supported processing units**

 {{ cpu-gpu }}

## learning_rate {#learning_rate}

Command-line: `-w`, `--learning-rate`

_Alias:_ `eta`

#### Description

The learning rate.

Used for reducing the gradient step.

**Type**

{{ python-type--float }}

**Default value**

The default value is defined automatically for {{ error-function--Logit }}, {{ error-function--MultiClass }} & {{ error-function--RMSE }} loss functions depending on the number of iterations if none of parameters leaf_estimation_iterations, `--leaf-estimation-method`,l2_leaf_reg is set. In this case, the selected learning rate is printed to stdout and saved in the model.

In other cases, the default value is 0.03.

**Supported processing units**

 {{ cpu-gpu }}

## random_seed {#random_seed}

Command-line: `-r`, `--random-seed`

_Alias:_`random_state`

#### Description

The random seed used for training.

**Type**

{{ python-type--int }}

**Default value**

{% cut "Python package" %}

None ({{ fit--random_seed }})

{% endcut %}

{% cut "R package, Command-line" %}

{{ fit--random_seed }}

{% endcut %}

**Supported processing units**

 {{ cpu-gpu }}

## l2_leaf_reg {#l2_leaf_reg}

Command-line: `--l2-leaf-reg`, `l2-leaf-regularizer`

_Alias:_ `reg_lambda`

#### Description

Coefficient at the L2 regularization term of the cost function.

Any positive value is allowed.

**Type**

{{ python-type--float }}

**Default value**

 {{ fit--l2-leaf-reg }}

**Supported processing units**

 {{ cpu-gpu }}


## bootstrap_type {#bootstrap_type}

Command-line: `--bootstrap-type`

#### Description

[Bootstrap type](../../concepts/algorithm-main-stages_bootstrap-options.md). Defines the method for sampling the weights of objects.

Supported methods:

- {{ fit__bootstrap-type__Bayesian }}
- {{ fit__bootstrap-type__Bernoulli }}
- {{ fit__bootstrap-type__MVS }}
- {{ fit__bootstrap-type__Poisson }} (supported for GPU only)
- {{ fit__bootstrap-type__No }}

**Type**

{{ python-type--string }}

**Default value**

The default value depends on `objective`, `task_type`, `bagging_temperature` and `sampling_unit`:

- When the objective parameter is {{ error-function__QueryCrossEntropy }}, {{ error-function__YetiRankPairwise }}, {{ error-function__PairLogitPairwise }} and the bagging_temperature parameter is not set: {{ fit__bootstrap-type__Bernoulli }} with the subsample parameter set to 0.5.
- Neither {{ error-function--MultiClass }} nor {{ error-function--MultiClassOneVsAll }}, task_type = CPU and sampling_unit = Object: {{ fit__bootstrap-type__MVS }} with the subsample parameter set to 0.8.
- Otherwise: {{ fit__bootstrap-type__Bayesian }}.


**Supported processing units**

 {{ cpu-gpu }}


## bagging_temperature {#bagging_temperature}

Command-line: `--bagging-temperature`

#### Description

Defines the settings of the Bayesian bootstrap. It is used by default in classification and regression modes.

Use the Bayesian bootstrap to assign random weights to objects.

The weights are sampled from exponential distribution if the value of this parameter is set to <q>1</q>. All weights are equal to 1 if the value of this parameter is set to <q>0</q>.

Possible values are in the range $[0; \inf)$. The higher the value the more aggressive the bagging is.

This parameter can be used if the selected bootstrap type is {{ fit__bootstrap-type__Bayesian }}.

**Type**

{{ python-type--float }}

**Default value**

 {{ fit__bagging-temperature }}

**Supported processing units**

 {{ cpu-gpu }}

## subsample {#subsample}

Command-line: `--subsample`

#### Description

Sample rate for bagging.

This parameter can be used if one of the following bootstrap types is selected:

- {{ fit__bootstrap-type__Poisson }}
- {{ fit__bootstrap-type__Bernoulli }}
- {{ fit__bootstrap-type__MVS }}

**Type**

{{ python-type--float }}

**Default value**

{% include [reusage-default-values-subsample__default](../../_includes/work_src/reusage-default-values/subsample__default.md) %}

**Supported processing units**

 {{ cpu-gpu }}

## sampling_frequency {#sampling_frequency}

Command-line: `--sampling-frequency`

#### Description

Frequency to sample weights and objects when building trees.

Supported values:

- {{ fit__sampling-frequency__PerTree }} — Before constructing each new tree
- {{ fit__sampling-frequency__PerTreeLevel }} — Before choosing each new split of a tree

**Type**

{{ python-type--string }}

**Default value**

 {{ fit__sampling-frequency }}

**Supported processing units**

 {{ calcer_type__cpu }}


## sampling_unit {#sampling_unit}

Command-line: `--sampling-unit`

#### Description


The sampling scheme.

Possible values:
- {{ python__ESamplingUnit__type__Object }} — The weight $w_{i}$ of the i-th object $o_{i}$ is used for sampling the corresponding object.
- {{ python__ESamplingUnit__type__Group }} — The weight $w_{j}$ of the group $g_{j}$ is used for sampling each object $o_{i_{j}}$ from the group $g_{j}$.

**Type**

{{ data-type__String }}

**Default value**

 {{ python__ESamplingUnit__type__Object }}

**Supported processing units**

 {{ cpu-gpu }}


## mvs_reg {#mvs_reg}

Command-line: `--mvs-reg`

#### Description

{% include [reusage-cli__mvs-head-fraction__div](../../_includes/work_src/reusage/cli__mvs-head-fraction__div.md) %}

{% note info %}

This parameter is supported only for the {{ fit__bootstrap-type__MVS }} sampling method (the `bootstrap_type` parameter must be set to {{ fit__bootstrap-type__MVS }}).

{% endnote %}

**Type**

{{ python-type--float }}

**Default value**

 The value is {{ fit__mvs_head_fraction }}

**Supported processing units**

 {{ calcer_type__cpu }}

## random_strength {#random_strength}

Command-line: `--random-strength`

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

**Type**

{{ python-type--float }}

**Default value**

 {{ fit--random-strength }}

**Supported processing units**

 {{ calcer_type__cpu }}

## use_best_model {#use_best_model}

Command-line: `--use-best-model`

#### Description

If this parameter is set, the number of trees that are saved in the resulting model is defined as follows:
1. Build the number of trees defined by the training parameters.
1. Use the validation dataset to identify the iteration with the optimal value of the metric specified in  `--eval-metric` (`--eval-metric`).

No trees are saved after this iteration.

This option requires a validation dataset to be provided.

**Type**

{{ python-type--bool }}

**Default value**

True if a validation set is input (the eval_set parameter is defined) and at least one of the label values of objects in this set differs from the others. False otherwise.

**Supported processing units**

 {{ cpu-gpu }}

## best_model_min_trees {#best_model_min_trees}

Command-line: `--best-model-min-trees`

#### Description

{% include [reusage-clii__best-model-min-trees__short-desc](../../_includes/work_src/reusage/clii__best-model-min-trees__short-desc.md) %}

Should be used with the `--use-best-model` parameter.

**Type**

{{ python-type--int }}

**Default value**

{% cut "Python package, R package" %}

None ({{ fit__best-model-min-trees }})

{% endcut %}

{% cut "Command-line" %}

The minimal number of trees for the best model is not set

{% endcut %}

**Supported processing units**

 {{ cpu-gpu }}

## depth {#depth}

Command-line: `-n`, `--depth`

_Alias:_ `max_depth`

#### Description

Depth of the trees.

The range of supported values depends on the processing unit type and the type of the selected loss function:

- CPU — Any integer up to  {{ fit--maxtree }}.

- GPU — Any integer up to {{ fit__maxtree__pairwise }} for pairwise modes ({{ error-function__YetiRank }}, {{ error-function__PairLogitPairwise }}, and {{ error-function__QueryCrossEntropy }}), and up to {{ fit--maxtree }} for all other loss functions.

**Type**

{{ python-type--int }}

**Default value**

 {{ fit--depth }}

**Supported processing units**

 {{ cpu-gpu }}

## grow_policy {#grow_policy}

Command-line: `--grow-policy`

#### Description

The tree growing policy. Defines how to perform greedy tree construction.

Possible values:
- {{ growing_policy__SymmetricTree }} —A tree is built level by level until the specified depth is reached. On each iteration, all leaves from the last tree level are split with the same condition. The resulting tree structure is always symmetric.
- {{ growing_policy__Depthwise }} — A tree is built level by level until the specified depth is reached. On each iteration, all non-terminal leaves from the last tree level are split. Each leaf is split by condition with the best loss improvement.

  {% note info %}

  Models with this growing policy can not be analyzed using the {{ title__predictiondiff }} feature importance and can be exported only to {{ fit__model-format_json }} and {{ fitpython__model-format_cbm }}.

  {% endnote %}

- {{ growing_policy__Lossguide }} — A tree is built leaf by leaf until the specified maximum number of leaves is reached. On each iteration, non-terminal leaf with the best loss improvement is split.

  {% include [reusage-cli__grow-policy__note](../../_includes/work_src/reusage/cli__grow-policy__note.md) %}

**Type**

{{ python-type--string }}

**Default value**

 {{ growing_policy__default }}

**Supported processing units**

 {{ cpu-gpu }}

## min_data_in_leaf {#min_data_in_leaf}

Command-line: `--min-data-in-leaf`

_Alias:_ `min_child_samples`

#### Description

The minimum number of training samples in a leaf. {{ product }} does not search for new splits in leaves with samples count less than the specified value.
Can be used only with the {{ growing_policy__Lossguide }} and {{ growing_policy__Depthwise }} growing policies.

**Type**

{{ python-type--int }}

**Default value**

 {{ min-samples-in-leaf__default }}

**Supported processing units**

 {{ cpu-gpu }}


## max_leaves {#max_leaves}

Command-line: `--max-leaves`

_Alias:_`num_leaves`

#### Description

The maximum number of leafs in the resulting tree. Can be used only with the {{ growing_policy__Lossguide }} growing policy.

{% note info %}

It is not recommended to use values greater than 64, since it can significantly slow down the training process.

{% endnote %}

**Type**

{{ python-type--int }}

**Default value**

{{ max-leaves-count__default }}

**Supported processing units**

{{ cpu-gpu }}

## ignored_features {#ignored_features}

Command-line: `-I`, `--ignore-features`

#### Description

Feature indices to exclude from the training.

{% cut "Python package" %}

 {% include [ignored-features](../../_includes/work_src/reusage/ignored-features.md) %}

 {% include [cli__ignored_features__specifics](../../_includes/work_src/reusage/cli__ignored_features__specifics.md) %}

 For example, use the following construction if features indexed 1, 2, 7, 42, 43, 44, 45, should be ignored: `[1,2,7,42,43,44,45]`

{% endcut %}

{% cut "R package" %}

 {% include [cli__ignored_features__specifics](../../_includes/work_src/reusage/cli__ignored_features__specifics.md) %}

 For example, if training should exclude features with the identifiers 1, 2, 7, 42, 43, 44, 45, the value of this parameter should be set to c(1,2,7,42,43,44,45).

{% endcut %}

{% cut "Command-line" %}

 {% include [ignored-features](../../_includes/work_src/reusage/ignored-features.md) %}

 {% include [reusage-cli__ignored_features__specifics](../../_includes/work_src/reusage/cli__ignored_features__specifics.md) %}

 For example, if training should exclude features with the identifiers 1, 2, 7, 42, 43, 44, 45, use the following construction: `1:2:7:42-45`.

{% endcut %}

**Default value**

{% cut "Python package, R package" %}

 None

{% endcut %}

{% cut "Command-line" %}

 Omitted

{% endcut %}

**Supported processing units**

{{ cpu-gpu }}

## one_hot_max_size {#one_hot_max_size}

Command-line: `--one-hot-max-size`

#### Description

Use one-hot encoding for all categorical features with a number of different values less than or equal to the given parameter value. Ctrs are not calculated for such features.

See [details](../../features/categorical-features.md).

**Type**

{{ python-type--int }}

**Default value**

{% include [reusage-default-values-one-hot-max-size-default](../../_includes/work_src/reusage-default-values/one-hot-max-size-default.md) %}

**Supported processing units**

 {{ cpu-gpu }}


## has_time {#has_time}

Command-line: `--has-time`

#### Description

Use the order of objects in the input data (do not perform random permutations during the [Transforming categorical features to numerical features](../../concepts/algorithm-main-stages_cat-to-numberic.md) and [Choosing the tree structure](../../concepts/algorithm-main-stages_choose-tree-structure.md) stages).

The {{ cd-file__col-type__Timestamp }} column type is used to determine the order of objects if specified in the [input data](../../concepts/input-data.md).

**Type**

{{ python-type--bool }}

**Default value**

{{ fit--has_time }}

**Supported processing units**

{{ cpu-gpu }}

## rsm {#rsm}

Command-line: `--rsm`

_Alias:_`colsample_bylevel`

#### Description

Random subspace method. The percentage of features to use at each split selection, when features are selected over again at random.

The value must be in the range (0;1].

**Type**

{{ python-type--float }} (0;1]

**Default value**

None (set to 1)

**Supported processing units**

{{ calcer_type__cpu }}; {{ calcer_type__gpu }} for pairwise ranking


## nan_mode {#nan_mode}

Command-line: `--nan-mode`

#### Description

The method for  [processing missing values](../../concepts/algorithm-missing-values-processing.md) in the input dataset.

{% include [reusage-cmd__nan-mode__div](../../_includes/work_src/reusage/cmd__nan-mode__div.md) %}

**Type**

{{ python-type--string }}

**Default value**

 {{ fit--nan_mode }}

**Supported processing units**

 {{ cpu-gpu }}


## input_borders {#input_borders}

Command-line: `--input-borders-file`

#### Description

Load [Custom quantization borders and missing value modes](../../concepts/input-data_custom-borders.md) from a file (do not generate them).

Borders are automatically generated before training if this parameter is not set.

**Type**

{{ python-type--string }}

**Default value**

{% cut "Python package" %}

None

{% endcut %}

{% cut "Command-line" %}

The file is not loaded, the values are generated

{% endcut %}

**Supported processing units**

 {{ cpu-gpu }}


## output_borders {#output_borders}

Command-line: `--output-borders-file`

#### Description

Save quantization borders for the current dataset to a file.

Refer to the [file format description](../../concepts/output-data_custom-borders.md).

**Type**

{{ python-type--string }}

**Default value**

{% cut "Python package" %}

None

{% endcut %}

{% cut "Command-line" %}

The file is not saved

{% endcut %}

**Supported processing units**

 {{ cpu-gpu }}


## fold_permutation_block {#fold_permutation_block}

Command-line: `--fold-permutation-block`

#### Description


Objects in the dataset are grouped in blocks before the random permutations. This parameter defines the size of the blocks. The smaller is the value, the slower is the training. Large values may result in quality degradation.

**Type**

{{ python-type--int }}

**Default value**

{% cut "Python package" %}

1

{% endcut %}

{% cut "R package, Command-line" %}

Default value differs depending on the dataset size and ranges from 1 to 256 inclusively

{% endcut %}

**Supported processing units**

 {{ cpu-gpu }}

## leaf_estimation_method {#leaf_estimation_method}

Command-line: `--leaf-estimation-method`

#### Description

The method used to calculate the values in leaves.

Possible values:

- {{ fit__leaf_estimation_method__Newton }}
- {{ fit__leaf_estimation_method__Gradient }}
- {{ fit__leaf_estimation_method__Exact }}

**Type**

{{ python-type--string }}

**Default value**

Depends on the mode and the selected loss function:
- Regression with {{ error-function--Quantile }} or {{ error-function--MAE }} loss functions — One {{ fit__leaf_estimation_method__Exact }} iteration.
- Regression with any loss function but {{ error-function--Quantile }} or {{ error-function--MAE }} – One {{ fit__leaf_estimation_method__Gradient }} iteration.
- Classification mode – Ten {{ fit__leaf_estimation_method__Newton }} iterations.
- Multiclassification mode – One {{ fit__leaf_estimation_method__Newton }} iteration.

**Supported processing units**

 {{ cpu-gpu }}


## leaf_estimation_iterations {#leaf_estimation_iterations}

Command-line: `--leaf-estimation-iterations`

#### Description

{{ product }} might calculate leaf values using several gradient or newton steps instead of a single one.

This parameter regulates how many steps are done in every tree when calculating leaf values.

**Type**

{{ python-type--int }}

**Default value**

{% cut "Python package" %}

None ({{ fit--gradient_iterations }})

{% endcut %}

{% cut "R package, Command-line" %}

{{ fit--gradient_iterations }}

{% endcut %}

**Supported processing units**

{{ cpu-gpu }}


## leaf_estimation_backtracking {#leaf_estimation_backtracking}

Command-line: `--leaf-estimation-backtracking`

#### Description

When the value of the `leaf_estimation_iterations` parameter is greater than 1, {{ product }} makes several gradient or newton steps when calculating the resulting leaf values of a tree.

The behaviour differs depending on the value of this parameter:

- No — Every next step is a regular gradient or newton step: the gradient step is calculated and added to the leaf.
- Any other value —Backtracking is used.
  In this case, before adding a step, a condition is checked. If the condition is not met, then the step size is reduced (divided by 2), otherwise the step is added to the leaf.

When `leaf_estimation_iterations` for the Command-line version is set to `n`, the leaf estimation iterations are calculated as follows: each iteration is either an addition of the next step to the leaf value, or it's a scaling of the leaf value. Scaling counts as a separate iteration. Thus, it is possible that instead of having `n` gradient steps, the algorithm makes a single gradient step that is reduced `n` times, which means that it is divided by $2\cdot n$ times.

Possible values:

- {{ cli__leaf-estimation-backtracking__No }} — Do not use backtracking. Supported on {{ calcer_type__cpu }} and {{ calcer_type__gpu }}.
- {{ cli__leaf-estimation-backtracking__AnyImprovement }} — Reduce the descent step up to the point when the loss function value is smaller than it was on the previous step. The trial reduction factors are 2, 4, 8, and so on. Supported on {{ calcer_type__cpu }} and {{ calcer_type__gpu }}.
- {{ cli__leaf-estimation-backtracking__Armijo }} — Reduce the descent step until the Armijo condition is met. Supported only on {{ calcer_type__gpu }}.

**Type**

{{ python-type--string }}

**Default value**

 {{ cli__leaf-estimation-backtracking__default }}

**Supported processing units**

 Depends on the selected value

## fold_len_multiplier {#fold_len_multiplier}

Command-line: `--fold-len-multiplier`

#### Description

Coefficient for changing the length of folds.

The value must be greater than 1. The best validation result is achieved with minimum values.

With values close to 1 (for example, $1+\epsilon$), each iteration takes a quadratic amount of memory and time for the number of objects in the iteration. Thus, low values are possible only when there is a small number of objects.

**Type**

{{ python-type--float }}

**Default value**

{{ fit--fold-len-multiplier }}

**Supported processing units**

{{ cpu-gpu }}

## approx_on_full_history {#approx_on_full_history}

Command-line:`--approx-on-full-history`

#### Description


The principles for calculating the approximated values.

Possible values:
- <q>False</q> — Use only а fraction of the fold for calculating the approximated values. The size of the fraction is calculated as follows: $\frac{1}{{X}}$, where `X` is the specified coefficient for changing the length of folds. This mode is faster and in rare cases slightly less accurate
- <q>True</q> — Use all the preceding rows in the fold for calculating the approximated values. This mode is slower and in rare cases slightly more accurate.

**Type**

{{ python-type--bool }}

**Default value**

{% cut "Python package, Command-line" %}

{{ fit--approx_on_full_history }}

{% endcut %}

{% cut "R package" %}

True

{% endcut %}

**Supported processing units**

 {{ calcer_type__cpu }}


## class_weights {#class_weights}

Command-line: `--class-weights`

#### Description


{% include [reusage-class-weights__short-desc-intro](../../_includes/work_src/reusage/class-weights__short-desc-intro.md) %}


{% cut "Python package" %}

{% note info %}

{% include [reusage-imbalanced-datasets-description](../../_includes/work_src/reusage/imbalanced-datasets-description.md) %}

{% endnote %}

For example, `class_weights=[0.1, 4]`multiplies the weights of objects from class 0 by 0.1 and the weights of objects from class 1 by 4.

If class labels are not standard consecutive integers [0, 1 ... class_count-1], use the {{ python-type--dict }} or {{ python-type__collectionsOrderedDict }} type with label to weight mapping.

For example, `class_weights={'a': 1.0, 'b': 0.5, 'c': 2.0}` multiplies the weights of objects with class label `a` by 1.0, the weights of objects with class label `b` by 0.5 and the weights of objects with class label `c` by 2.0.

The dictionary form can also be used with standard consecutive integers class labels for additional readability. For example: `class_weights={0: 1.0, 1: 0.5, 2: 2.0}`.

{% note info %}

Class labels are extracted from dictionary keys for the following types of class_weights:

- {{ python-type--dict }}
- {{ python-type__collectionsOrderedDict }} (when the order of classes in the model is important)

The class_names parameter can be skipped when using these types.

{% endnote %}

{% note alert %}

Do not use this parameter with auto_class_weights and scale_pos_weight.

{% endnote %}

{% endcut %}

{% cut "R package" %}

For example, `class_weights <- c(0.1, 4)` multiplies the weights of objects from class 0 by 0.1 and the weights of objects from class 1 by 4.

{% note alert %}

Do not use this parameter with auto_class_weights.

{% endnote %}

{% endcut %}

{% cut "Command-line" %}

{% note info %}

The quantity of class weights must match the quantity of class names specified in the `--class-names` parameter and the number of classes specified in the `--classes-count parameter`.

{% include [reusage-imbalanced-datasets-description](../../_includes/work_src/reusage/imbalanced-datasets-description.md) %}

{% endnote %}

Format:

```
<value for class 1>,..,<values for class N>
```
For example:

```
0.85,1.2,1
```

{% note alert %}

Do not use this parameter with auto_class_weights.

{% endnote %}

{% endcut %}

**Type**

- {{ python-type--list }}
- {{ python-type--dict }}
- {{ python-type__collectionsOrderedDict }}

**Default value**

 {{ fit--class-weights }}

**Supported processing units**

 {{ cpu-gpu }}


## class_names {#class_names}

#### Description


Classes names. Allows to redefine the default values when using the {{ error-function--MultiClass }} and {{ error-function--Logit }} metrics.

If the upper limit for the numeric class label is specified, the number of classes names should match this value.

{% note warning %}

The quantity of classes names must match the quantity of classes weights specified in the `--class-weights` parameter and the number of classes specified in the `--classes-count` parameter.

{% endnote %}

Format:

```
<name for class 1>,..,<name for class N>
```

For example:

```
smartphone,touchphone,tablet
```

**Type**

{{ python-type--list-of-strings }}

**Default value**

 None

**Supported processing units**

 {{ cpu-gpu }}

## auto_class_weights {#auto_class_weights}

Command-line: `--auto-class-weights`

#### Description


{% include [reusage-cli__auto-class-weights__div](../../_includes/work_src/reusage/cli__auto-class-weights__div.md) %}

{% note alert %}

Do not use this parameter with `class_weights` and `scale_pos_weight`.

{% endnote %}

**Type**

{{ python-type--string }}

**Default value**

 {{ autoclass__weights__default }}

**Supported processing units**

 {{ cpu-gpu }}

## scale_pos_weight {#scale_pos_weight}

#### Description


The weight for class 1 in binary classification. The value is used as a multiplier for the weights of objects from class 1.

{% note info %}

For imbalanced datasets, the weight multiplier can be set to $\left(\frac{sum\_negative}{sum\_positive}\right)$

{% endnote %}

{% note alert %}

Do not use this parameter with `auto_class_weights` and `class_weights`.

{% endnote %}

**Type**

{{ python-type--float }}

**Default value**

 1.0

**Supported processing units**

 {{ cpu-gpu }}

## boosting_type {#boosting_type}

Command-line: `--boosting-type`

#### Description


Boosting scheme.

Possible values:
- {{ fit__boosting-type__ordered }} — Usually provides better quality on small datasets, but it may be slower than the Plain scheme.
- {{ fit__boosting-type__plain }} — The classic gradient boosting scheme.

**Type**

{{ python-type--string }}

**Default value**

{% cut "{{ fit__boosting-type }}" %}

- {{ calcer_type__cpu }}

  {{ fit__boosting-type__plain }}

-  {{ calcer_type__gpu }}

    - Any number of objects, {{ error-function--MultiClass }} or {{ error-function--MultiClassOneVsAll }} mode: {{ fit__boosting-type__plain }}
    - More than 50 thousand objects, any mode: {{ fit__boosting-type__plain }}
    - Less than or equal to 50 thousand objects, any mode but {{ error-function--MultiClass }} or {{ error-function--MultiClassOneVsAll }}: {{ fit__boosting-type__ordered }}

{% endcut %}

**Supported processing units**

 {{ cpu-gpu }}

Only the {{ fit__boosting-type__plain }} mode is supported for the {{ error-function--MultiClass }} loss on GPU


## boost_from_average {#boost_from_average}

Command-line: `--boost-from-average`

#### Description


Initialize approximate values by best constant value for the specified loss function. Sets the value of bias to the initial best constant value.

Available for the following loss functions:
- {{ error-function--RMSE }}
- {{ error-function--Logit }}
- {{ error-function--CrossEntropy }}
- {{ error-function--Quantile }}
- {{ error-function--MAE }}
- {{ error-function--MAPE }}

**Type**

{{ python-type--bool }}

**Default value**

{% include [reusage-default-values-boost-from-average](../../_includes/work_src/reusage-default-values/boost-from-average.md) %}

**Supported processing units**

 {{ cpu-gpu }}

## langevin {#langevin}

Command-line: `--langevin`

#### Description


Enables the Stochastic Gradient Langevin Boosting mode.

Refer to the [SGLB: Stochastic Gradient Langevin Boosting]({{ stochastic-gradient-langevin-boosting }}) paper for details.

**Type**

{{ python-type--bool }}

**Default value**

 False

**Supported processing units**

 {{ calcer_type__cpu }}

## diffusion_temperature {#diffusion_temperature}

Command-line: `--diffusion-temperature`

#### Description


The diffusion temperature of the Stochastic Gradient Langevin Boosting mode.

Only non-negative values are supported.

**Type**

{{ python-type--float }}

**Default value**

 10000

**Supported processing units**

 {{ calcer_type__cpu }}

## posterior_sampling {#posterior_sampling}

Command-line: `--posterior-sampling	`

#### Description


If this parameter is set several options are specified as follows and model parameters are checked to obtain uncertainty predictions with good theoretical properties.
Specifies options:

- `Langevin`: true,
- `DiffusionTemperature`: objects in learn pool count,
- `ModelShrinkRate`: 1 / (2. * objects in learn pool count).

**Type**

bool

**Default value**

 False

**Supported processing units**

 CPU only

## allow_const_label {#allow_const_label}

Command-line: `--allow-const-label`

#### Description


Use it to train models with datasets that have equal label values for all objects.

**Type**

{{ python-type--bool }}

**Default value**

 {{ fit__allow-const-label }}

**Supported processing units**

 {{ cpu-gpu }}

## score_function {#score_function}

Command-line: `--score-function`

#### Description


The [score type](../../concepts/algorithm-score-functions.md) used to select the next split during the tree construction.

Possible values:

- {{ scorefunction__Correlation }} (do not use this score type with the {{ growing_policy__Lossguide }} tree growing policy)
- {{ scorefunction__L2 }}
- {{ scorefunction__NewtonCorrelation }} (do not use this score type with the {{ growing_policy__Lossguide }} tree growing policy)
- {{ scorefunction__NewtonL2 }}

**Type**

{{ python-type--string }}

**Default value**

 {{ scorefunction__default }}

**Supported processing units**

The supported score functions vary depending on the processing unit type:

- {{ calcer_type__gpu }} — All score types

- {{ calcer_type__cpu }} — {{ scorefunction__Correlation }}, {{ scorefunction__L2 }}


## monotone_constraints {#monotone_constraints}

Command-line: `--monotone-constraints`

#### Description

{% include [reusage-cli__monotone-constraints__desc__div](../../_includes/work_src/reusage/cli__monotone-constraints__desc__div.md) %}

{% include [reusage-monotone-constraints__formats__intro](../../_includes/work_src/reusage/monotone-constraints__formats__intro.md) %}

- {% include [reusage-set-individual-constraints__div](../../_includes/work_src/reusage/set-individual-constraints__div.md) %}

  Zero constraints for features at the end of the list may be dropped.

  In `monotone_constraints = "(1,0,-1)"`an increasing constraint is set on the first feature and a decreasing one on the third. Constraints are disabled for all other features.

- {% include [reusage-monotone-constraints__formats__individually-for-required-features](../../_includes/work_src/reusage/monotone-constraints__formats__individually-for-required-features.md) %}

    These examples

    ```
    monotone-constraints = "2:1,4:-1"
    ```

    ```
    monotone-constraints = "Feature2:1,Feature4:-1"
    ```

    are identical, given that the name of the feature index 2 is <q>Feature2</q> and the name of the feature indexed 4 is <q>Feature4</q>.

- {% include [reusage-cli__monotone-constraints__arra-or-dict](../../_includes/work_src/reusage/cli__monotone-constraints__arra-or-dict.md) %}

  {% cut "Array examples" %}

    ```
    monotone_constraints = [1, 0, -1]
    ```

  {% endcut %}

  {% cut "These dictionary examples" %}

    ```python
    monotone_constraints = {"Feature2":1,"Feature4":-1}
    ```

    ```python
    monotone_constraints = {"2":1, "4":-1}
    ```

  {% endcut %}

  are identical, given that the name of the feature indexed 2 is <q>Feature2</q> and the name of the feature indexed 4 is <q>Feature4</q>.

**Type**

- {{ python-type--list-of-strings }}
- {{ python-type--string }}
- {{ python-type--dict }}
- {{ python-type--list }}

**Default value**

{% cut "Python package, R package" %}

None

{% endcut %}

{% cut "Command-line" %}

Ommited

{% endcut %}

**Supported processing units**

 {{ calcer_type__cpu }}

## feature_weights {#feature_weights}

Command-line: `--feature-weights`

#### Description


{% include [reusage-cli__feature-weight__desc__intro](../../_includes/work_src/reusage/cli__feature-weight__desc__intro.md) %}


- {% include [reusage-cli__feature-weight__desc__weight-for-each-feature](../../_includes/work_src/reusage/cli__feature-weight__desc__weight-for-each-feature.md) %}

  In this

  {% cut "example" %}

    ```
    feature_weights = "(0.1,1,3)"
    ```

  {% endcut %}

  the multiplication weight is set to 0.1, 1 and 3 for the first, second and third features respectively. The multiplication weight for all other features is set to 1.

- {% include [reusage-cli__feature-weight__formats__individually-for-required-features](../../_includes/work_src/reusage/cli__feature-weight__formats__individually-for-required-features.md) %}

  {% cut "These examples" %}

    ```
    feature_weights = "2:1.1,4:0.1"
    ```

    ```
    feature_weights = "Feature2:1.1,Feature4:0.1"
    ```

  {% endcut %}

  are identical, given that the name of the feature indexed 2 is <q>Feature2</q> and the name of the feature indexed 4 is <q>Feature4</q>.

- {% include [reusage-cli__feature-weights__arra-or-dict](../../_includes/work_src/reusage/cli__feature-weights__arra-or-dict.md) %}

  {% cut "Array examples" %}

    ```
    feature_weights = [0.1, 1, 3]
    ```

  {% endcut %}

  {% cut "These dictionary examples" %}

    ```python
    feature_weights = {"Feature2":1.1,"Feature4":0.3}
    ```

    ```python
    feature_weights = {"2":1.1, "4":0.3}
    ```

  {% endcut %}

  are identical, given that the name of the feature indexed 2 is <q>Feature2</q> and the name of the feature indexed 4 is <q>Feature4</q>.

**Type**

- {{ python-type--list }}
- {{ python-type__np_ndarray }}
- {{ python-type--string }}
- {{ python-type--dict }}

**Default value**

 1 for all features

**Supported processing units**

 {{ calcer_type__cpu }}

## first_feature_use_penalties {#first_feature_use_penalties}

Command-line: `--first-feature-use-penalties`

#### Description


{% include [reusage-cli__first-feature-use-penalties__intro](../../_includes/work_src/reusage/cli__first-feature-use-penalties__intro.md) %}

{% include [penalties-format-penalties_format](../../_includes/work_src/reusage-python/penalties_format.md) %}

**Type**

- {{ python-type--list }}
- {{ python-type__np_ndarray }}
- {{ python-type--string }}
- {{ python-type--dict }}

**Default value**

 0 for all features

**Supported processing units**

 {{ calcer_type__cpu }}

## fixed_binary_splits {#fixed_binary_splits}

Command-line: `--fixed-binary-splits`

#### Description


A list of indices of binary features to put at the top of each tree; ignored if `grow_policy` is `Symmetric`.

**Type**

{{ python-type--list }}

**Default value**

 None

**Supported processing units**

 {{ calcer_type__gpu }}

## penalties_coefficient {#penalties_coefficient}

Command-line: `--penalties-coefficient`

#### Description


A single-value common coefficient to multiply all penalties.

Non-negative values are supported.

**Type**

{{ python-type--float }}

**Default value**

 1

**Supported processing units**

 {{ calcer_type__cpu }}

## per_object_feature_penalties {#per_object_feature_penalties}

Command-line: `--per-object-feature-penalties`

#### Description


{% include [reusage-per-object-feature-penalties__intro](../../_includes/work_src/reusage/per-object-feature-penalties__intro.md) %}

{% cut "Python package" %}

{% include [penalties-format-penalties_format](../../_includes/work_src/reusage-python/penalties_format.md) %}

{% endcut %}

{% cut "R package" %}

{% include [penalties-format-penalties_format](../../_includes/work_src/reusage-r/r__penalties.md) %}

{% endcut %}

**Type**

- {{ python-type--list }}
- {{ python-type__np_ndarray }}
- {{ python-type--string }}
- {{ python-type--dict }}

**Default value**

 0 for all objects

**Supported processing units**

 {{ calcer_type__cpu }}

## model_shrink_rate {#model_shrink_rate}

Command-line: `--model-shrink-rate`

#### Description


The constant used to calculate the coefficient for multiplying the model on each iteration.
The actual model shrinkage coefficient calculated at each iteration depends on the value of the `--model-shrink-mode`for the Command-line version parameter. The resulting value of the coefficient should be always in the range (0, 1].

**Type**

{{ python-type--float }}

**Default value**

The default value depends on the values of the following parameters:

- `--model-shrink-mode` for the Command-line version

- `--monotone-constraints` for the Command-line version

**Supported processing units**

 {{ calcer_type__cpu }}

## model_shrink_mode {#model_shrink_mode}

Command-line: `model_shrink_mode`

#### Description

Determines how the actual model shrinkage coefficient is calculated at each iteration.

Possible values:

- {{ model_shrink_mode__Constant }}:

  $1 - model\_shrink\_rate \cdot learning\_rate {,}$
    - $model\_shrink\_rate$ is the value of the `--model-shrink-rate`for the Command-line version parameter.
    - $learning\_rate$ is the value of the `--learning-rate`for the Command-line version parameter

- {{ model_shrink_mode__Decreasing }}:

  $1 - \frac{model\_shrink\_rate}{i} {,}$
    - $model\_shrink\_rate$ is the value of the `--model-shrink-rate`for the Command-line version parameter.
    - $i$ is the identifier of the iteration.

**Type**

{{ python-type--string }}

**Default value**

 {{ model_shrink_mode__Constant }}

**Supported processing units**

 {{ calcer_type__cpu }}
