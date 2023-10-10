## Common parameters

### loss_function

_Alias:_`objective`

**Type**
- {{ python-type--string }}
- object

#### Description

{% include [reusage-loss-function-short-desc](loss-function-short-desc.md) %}


{% include [reusage-loss-function-format](loss-function-format.md) %}


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
- {{ error-function__LambdaMart }}

{% endcut %}

A custom python object can also be set as the value of this parameter (see an [example](../../../concepts/python-usages-examples.md)).

{% include [reusage-loss-function--example](loss-function--example.md) %}



**Default value**


{% cut "Depends on the class" %}

- [CatBoostClassifier](../../../concepts/python-reference_catboostclassifier.md): {{ error-function--Logit }} if the `target_border` parameter value differs from None. Otherwise, the default loss function depends on the number of unique target values and is either set to {{ error-function--Logit }} or {{ error-function--MultiClass }}.
- [CatBoost](../../../concepts/python-reference_catboost.md) and [CatBoostRegressor](../../../concepts/python-reference_catboostregressor.md): {{ error-function--RMSE }}

{% endcut %}


**Supported processing units**


 {{ cpu-gpu }}



### custom_metric


**Type**


- {{ python-type--string }}
- {{ python-type--list-of-strings }}


#### Description


{% include [reusage-custom-loss--basic](custom-loss--basic.md) %}


{% include [reusage-loss-function-format](loss-function-format.md) %}


[Supported metrics](../../../references/custom-metric__supported-metrics.md)

Examples:
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

{% include [reusage-custom-loss--values-saved-to](custom-loss--values-saved-to.md) %}


Use the [visualization tools](../../../features/visualization.md) to see a live chart with the dynamics of the specified metrics.


**Default value**

{{ fit--custom_loss }}

**Supported processing units**


 {{ cpu-gpu }}



### eval_metric


**Type**


- {{ python-type--string }}
- {{ python-type__object }}


#### Description


{% include [reusage-eval-metric--basic](eval-metric--basic.md) %}


{% include [reusage-eval-metric--format](eval-metric--format.md) %}


[Supported metrics](../../../references/eval-metric__supported-metrics.md)

A user-defined function can also be set as the value (see an [example](../../../concepts/python-usages-examples.md)).

{% include [reusage-eval-metric--examples](eval-metric--examples.md) %}



**Default value**

{{ fit--eval-metric }}

**Supported processing units**


 {{ cpu-gpu }}



### iterations


_Aliases:_
- `num_boost_round`
- `n_estimators`
- `num_trees`

**Type**

{{ python-type--int }}

#### Description


The maximum number of trees that can be built when solving machine learning problems.

When using other parameters that limit the number of iterations, the final number of trees may be less than the number specified in this parameter.


**Default value**

{{ fit--iterations }}

**Supported processing units**


 {{ cpu-gpu }}



### learning_rate


_Alias:_`eta`

**Type**

{{ python-type--float }}

#### Description


The learning rate.

Used for reducing the gradient step.


**Default value**


The default value is defined automatically for {{ error-function--Logit }}, {{ error-function--MultiClass }} & {{ error-function--RMSE }} loss functions depending on the number of iterations if none of these parameters

- `leaf_estimation_iterations`
- `--leaf-estimation-method`
- `l2_leaf_reg`

is set. In this case, the selected learning rate is printed to stdout and saved in the model.

In other cases, the default value is 0.03.


**Supported processing units**


 {{ cpu-gpu }}



### random_seed


_Alias:_`random_state`

**Type**

{{ python-type--int }}

#### Description


The random seed used for training.


**Default value**

None ({{ fit--random_seed }})

**Supported processing units**


 {{ cpu-gpu }}



### l2_leaf_reg


_Alias:_`reg_lambda`

**Type**

{{ python-type--float }}

#### Description

Coefficient at the L2 regularization term of the cost function.
Any positive value is allowed.


**Default value**

{{ fit--l2-leaf-reg }}

**Supported processing units**


 {{ cpu-gpu }}



### bootstrap_type


**Type**

{{ python-type--string }}

#### Description


[Bootstrap type](../../../concepts/algorithm-main-stages_bootstrap-options.md). Defines the method for sampling the weights of objects.

Supported methods:

- {{ fit__bootstrap-type__Bayesian }}
- {{ fit__bootstrap-type__Bernoulli }}
- {{ fit__bootstrap-type__MVS }}
- {{ fit__bootstrap-type__Poisson }} (supported for GPU only)
- {{ fit__bootstrap-type__No }}


**Default value**

The default value depends on `objective`, `task_type`, `bagging_temperature` and `sampling_unit`

- When the `objective` parameter is {{ error-function__QueryCrossEntropy }}, {{ error-function__YetiRankPairwise }}, {{ error-function__PairLogitPairwise }} and the `bagging_temperature` parameter is not set: {{ fit__bootstrap-type__Bernoulli }} with the `subsample` parameter set to 0.5
- Not {{ error-function--MultiClass }} and {{ error-function--MultiClassOneVsAll }}, `task_type` = CPU and `sampling_unit` = `Object`: {{ fit__bootstrap-type__MVS }} with the `subsample` parameter set to 0.8.
- Otherwise: {{ fit__bootstrap-type__Bayesian }}.



**Supported processing units**


 {{ cpu-gpu }}



### bagging_temperature


**Type**

{{ python-type--float }}

#### Description


Defines the settings of the Bayesian bootstrap. It is used by default in classification and regression modes.

Use the Bayesian bootstrap to assign random weights to objects.

The weights are sampled from exponential distribution if the value of this parameter is set to <q>1</q>. All weights are equal to 1 if the value of this parameter is set to <q>0</q>.

Possible values are in the range $[0; \inf)$. The higher the value the more aggressive the bagging is.

This parameter can be used if the selected bootstrap type is {{ fit__bootstrap-type__Bayesian }}.


**Default value**

{{ fit__bagging-temperature }}

**Supported processing units**


 {{ cpu-gpu }}



### subsample


**Type**

{{ python-type--float }}

#### Description


Sample rate for bagging.

This parameter can be used if one of the following bootstrap types is selected:
- {{ fit__bootstrap-type__Poisson }}
- {{ fit__bootstrap-type__Bernoulli }}
- {{ fit__bootstrap-type__MVS }}


**Default value**


{% include [reusage-default-values-subsample__default](../reusage-default-values/subsample__default.md) %}



**Supported processing units**


 {{ cpu-gpu }}



### sampling_frequency


**Type**

{{ python-type--string }}

#### Description


Frequency to sample weights and objects when building trees.

Supported values:

- {{ fit__sampling-frequency__PerTree }} — Before constructing each new tree
- {{ fit__sampling-frequency__PerTreeLevel }} — Before choosing each new split of a tree


**Default value**

{{ fit__sampling-frequency }}

**Supported processing units**


{{ calcer_type__cpu }}


### sampling_unit


**Type**

{{ data-type__String }}

#### Description


The sampling scheme.

Possible values:
- {{ python__ESamplingUnit__type__Object }} — The weight $w_{i}$ of the i-th object $o_{i}$ is used for sampling the corresponding object.
- {{ python__ESamplingUnit__type__Group }} — The weight $w_{j}$ of the group $g_{j}$ is used for sampling each object $o_{i_{j}}$ from the group $g_{j}$.


**Default value**

{{ python__ESamplingUnit__type__Object }}

**Supported processing units**


 {{ cpu-gpu }}



### mvs_reg


**Type**

{{ python-type--float }}

#### Description


{% include [reusage-cli__mvs-head-fraction__div](cli__mvs-head-fraction__div.md) %}


{% note info %}

This parameter is supported only for the {{ fit__bootstrap-type__MVS }} sampling method (the `bootstrap_type` parameter must be set to {{ fit__bootstrap-type__MVS }}).

{% endnote %}



**Default value**

The value is {{ fit__mvs_head_fraction }}

**Supported processing units**

{{ calcer_type__cpu }}

### random_strength


**Type**

{{ python-type--float }}

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


**Default value**

{{ fit--random-strength }}

**Supported processing units**

{{ calcer_type__cpu }}

### use_best_model


**Type**

{{ python-type--bool }}

#### Description


If this parameter is set, the number of trees that are saved in the resulting model is defined as follows:
1. Build the number of trees defined by the training parameters.
1. Use the validation dataset to identify the iteration with the optimal value of the metric specified in  `--eval-metric` (`--eval-metric`).

No trees are saved after this iteration.

This option requires a validation dataset to be provided.


**Default value**


True if a validation set is input (the `eval_set` parameter is defined) and at least one of the label values of objects in this set differs from the others. False otherwise.


**Supported processing units**


 {{ cpu-gpu }}



### best_model_min_trees


**Type**

{{ python-type--int }}

#### Description


{% include [reusage-clii__best-model-min-trees__short-desc](clii__best-model-min-trees__short-desc.md) %}


Should be used with the `--use-best-model` parameter.


**Default value**

None ({{ fit__best-model-min-trees }})

**Supported processing units**


 {{ cpu-gpu }}



### depth


_Alias:_`max_depth`

**Type**

{{ python-type--int }}

#### Description


Depth of the trees.

The range of supported values depends on the processing unit type and the type of the selected loss function:
- CPU — Any integer up to  {{ fit--maxtree }}.

- GPU — Any integer up to {{ fit__maxtree__pairwise }}pairwise modes ({{ error-function__YetiRank }}, {{ error-function__PairLogitPairwise }} and {{ error-function__QueryCrossEntropy }}) and up to   {{ fit--maxtree }} for all other loss functions.


**Default value**

{{ fit--depth }}

**Supported processing units**


 {{ cpu-gpu }}



### grow_policy


**Type**

{{ python-type--string }}

#### Description


The tree growing policy. Defines how to perform greedy tree construction.

Possible values:
- {{ growing_policy__SymmetricTree }} —A tree is built level by level until the specified depth is reached. On each iteration, all leaves from the last tree level are split with the same condition. The resulting tree structure is always symmetric.
- {{ growing_policy__Depthwise }} — A tree is built level by level until the specified depth is reached. On each iteration, all non-terminal leaves from the last tree level are split. Each leaf is split by condition with the best loss improvement.

    {% note info %}

    Models with this growing policy can not be analyzed using the {{ title__predictiondiff }} feature importance and can be exported only to {{ fit__model-format_json }} and {{ fitpython__model-format_cbm }}.

    {% endnote %}

- {{ growing_policy__Lossguide }} — A tree is built leaf by leaf until the specified maximum number of leaves is reached. On each iteration, non-terminal leaf with the best loss improvement is split.

    {% include [reusage-cli__grow-policy__note](cli__grow-policy__note.md) %}


**Default value**

{{ growing_policy__default }}

**Supported processing units**


 {{ cpu-gpu }}



### min_data_in_leaf


_Alias:_`min_child_samples`

**Type**

{{ python-type--int }}

#### Description

The minimum number of training samples in a leaf. {{ product }} does not search for new splits in leaves with samples count less than the specified value.
Can be used only with the {{ growing_policy__Lossguide }} and {{ growing_policy__Depthwise }} growing policies.


**Default value**

{{ min-samples-in-leaf__default }}

**Supported processing units**


 {{ cpu-gpu }}



### max_leaves


_Alias:_`num_leaves`

**Type**

{{ python-type--int }}

#### Description


The maximum number of leafs in the resulting tree. Can be used only with the {{ growing_policy__Lossguide }} growing policy.

{% note info %}

It is not recommended to use values greater than 64, since it can significantly slow down the training process.

{% endnote %}


**Default value**

{{ max-leaves-count__default }}

**Supported processing units**


 {{ cpu-gpu }}



### ignored_features


**Type**

{{ python-type--list }}

#### Description


{% include [reusage-ignored-feature__common-div](ignored-feature__common-div.md) %}


For example, use the following construction if features indexed 1, 2, 7, 42, 43, 44, 45, should be ignored:

```
[1,2,7,42,43,44,45]
```


**Default value**

None

**Supported processing units**


 {{ cpu-gpu }}



### one_hot_max_size


**Type**

{{ python-type--int }}

#### Description


Use one-hot encoding for all categorical features with a number of different values less than or equal to the given parameter value. Ctrs are not calculated for such features.

See [details](../../../features/categorical-features.md).


**Default value**


{% include [reusage-default-values-one-hot-max-size-default](../reusage-default-values/one-hot-max-size-default.md) %}



**Supported processing units**


 {{ cpu-gpu }}



### has_time


**Type**

{{ python-type--bool }}

#### Description


Use the order of objects in the input data (do not perform random permutations during the [Transforming categorical features to numerical features](../../../concepts/algorithm-main-stages_cat-to-numberic.md) and [Choosing the tree structure](../../../concepts/algorithm-main-stages_choose-tree-structure.md) stages).

The {{ cd-file__col-type__Timestamp }} column type is used to determine the order of objects if specified in the [input data](../../../concepts/input-data.md).


**Default value**

{{ fit--has_time }}

**Supported processing units**


 {{ cpu-gpu }}



### rsm


_Alias:_`colsample_bylevel`

**Type**

{{ python-type--float }} (0;1]

#### Description


Random subspace method. The percentage of features to use at each split selection, when features are selected over again at random.

The value must be in the range (0;1].


**Default value**

None (set to 1)

**Supported processing units**


{{ calcer_type__cpu }} and {{ calcer_type__gpu }} for pairwise ranking



### nan_mode


**Type**

{{ python-type--string }}

#### Description


The method for  [processing missing values](../../../concepts/algorithm-missing-values-processing.md) in the input dataset.

{% include [reusage-cmd__nan-mode__div](cmd__nan-mode__div.md) %}



**Default value**

{{ fit--nan_mode }}

**Supported processing units**


 {{ cpu-gpu }}



### input_borders


**Type**

{{ python-type--string }}

#### Description


Load [Custom quantization borders and missing value modes](../../../concepts/input-data_custom-borders.md) from a file (do not generate them).

Borders are automatically generated before training if this parameter is not set.


**Default value**

None

**Supported processing units**


 {{ cpu-gpu }}



### output_borders


**Type**

{{ python-type--string }}

#### Description


Save quantization borders for the current dataset to a file.

Refer to the [file format description](../../../concepts/output-data_custom-borders.md).


**Default value**

None

**Supported processing units**


 {{ cpu-gpu }}



### fold_permutation_block


**Type**

{{ python-type--int }}

#### Description


Objects in the dataset are grouped in blocks before the random permutations. This parameter defines the size of the blocks. The smaller is the value, the slower is the training. Large values may result in quality degradation.


**Default value**

1

**Supported processing units**


 {{ cpu-gpu }}



### leaf_estimation_method


**Type**

{{ python-type--string }}

#### Description


The method used to calculate the values in leaves.

Possible values:
- {{ fit__leaf_estimation_method__Newton }}
- {{ fit__leaf_estimation_method__Gradient }}
- {{ fit__leaf_estimation_method__Exact }}


**Default value**

Depends on the mode and the selected loss function:
- Regression with {{ error-function--Quantile }} or {{ error-function--MAE }} loss functions — One {{ fit__leaf_estimation_method__Exact }} iteration.
- Regression with any loss function but {{ error-function--Quantile }} or {{ error-function--MAE }} – One {{ fit__leaf_estimation_method__Gradient }} iteration.
- Classification mode – Ten {{ fit__leaf_estimation_method__Newton }} iterations.
- Multiclassification mode – One {{ fit__leaf_estimation_method__Newton }} iteration.


**Supported processing units**


- The {{ fit__leaf_estimation_method__Exact }} method is available only on {{ calcer_type__cpu }}
- All other methods are available on both {{ calcer_type__cpu }} and {{ calcer_type__gpu }}


### leaf_estimation_iterations


**Type**

{{ python-type--int }}

#### Description


{{ product }} might calculate leaf values using several gradient or newton steps instead of a single one.

This parameter regulates how many steps are done in every tree when calculating leaf values.


**Default value**

None ({{ fit--gradient_iterations }})

**Supported processing units**


 {{ cpu-gpu }}



### leaf_estimation_backtracking


**Type**

{{ python-type--string }}

#### Description


When the value of the `--leaf-estimation-iterations` for the Command-line version parameter is greater than 1, {{ product }} makes several gradient or newton steps when calculating the resulting leaf values of a tree.

The behaviour differs depending on the value of this parameter:

- No — Every next step is a regular gradient or newton step: the gradient step is calculated and added to the leaf.
- Any other value —Backtracking is used.
    In this case, before adding a step, a condition is checked. If the condition is not met, then the step size is reduced (divided by 2), otherwise the step is added to the leaf.

When `--leaf-estimation-iterations`for the Command-line version is set to `n`, the leaf estimation iterations are calculated as follows: each iteration is either an addition of the next step to the leaf value, or it's a scaling of the leaf value. Scaling counts as a separate iteration. Thus, it is possible that instead of having `n` gradient steps, the algorithm makes a single gradient step that is reduced `n` times, which means that it is divided by $2\cdot n$ times.

Possible values:
- {{ cli__leaf-estimation-backtracking__No }} — Do not use backtracking. Supported on {{ calcer_type__cpu }} and {{ calcer_type__gpu }}.
- {{ cli__leaf-estimation-backtracking__AnyImprovement }} — Reduce the descent step up to the point when the loss function value is smaller than it was on the previous step. The trial reduction factors are 2, 4, 8, and so on. Supported on {{ calcer_type__cpu }} and {{ calcer_type__gpu }}.
- {{ cli__leaf-estimation-backtracking__Armijo }} — Reduce the descent step until the Armijo condition is met. Supported only on {{ calcer_type__gpu }}.


**Default value**

{{ cli__leaf-estimation-backtracking__default }}

**Supported processing units**

Depends on the selected value

### fold_len_multiplier


**Type**

{{ python-type--float }}

#### Description


Coefficient for changing the length of folds.

The value must be greater than 1. The best validation result is achieved with minimum values.

With values close to 1 (for example, $1+\epsilon$), each iteration takes a quadratic amount of memory and time for the number of objects in the iteration. Thus, low values are possible only when there is a small number of objects.


**Default value**

{{ fit--fold-len-multiplier }}

**Supported processing units**


 {{ cpu-gpu }}



### approx_on_full_history


**Type**

{{ python-type--bool }}

#### Description


The principles for calculating the approximated values.

Possible values:
- <q>False</q> — Use only а fraction of the fold for calculating the approximated values. The size of the fraction is calculated as follows: $\frac{1}$, where `X` is the specified coefficient for changing the length of folds. This mode is faster and in rare cases slightly less accurate
- <q>True</q> — Use all the preceding rows in the fold for calculating the approximated values. This mode is slower and in rare cases slightly more accurate.


**Default value**

{{ fit--approx_on_full_history }}

**Supported processing units**


{{ calcer_type__cpu }}



### class_weights


**Type**


- {{ python-type--list }}
- {{ python-type--dict }}
- {{ python-type__collectionsOrderedDict }}


#### Description


{% include [reusage-class-weights__short-desc-intro](class-weights__short-desc-intro.md) %}


{% note info %}

{% include [reusage-imbalanced-datasets-description](imbalanced-datasets-description.md) %}

{% endnote %}


For example, `class_weights=[0.1, 4]`multiplies the weights of objects from class 0 by 0.1 and the weights of objects from class 1 by 4.

If class labels are not standard consecutive integers [0, 1 ... class_count-1], use the {{ python-type--dict }} or {{ python-type__collectionsOrderedDict }} type with label to weight mapping.

For example, `class_weights={'a': 1.0, 'b': 0.5, 'c': 2.0}` multiplies the weights of objects with class label `a` by 1.0, the weights of objects with class label `b` by 0.5 and the weights of objects with class label `c` by 2.0.

The dictionary form can also be used with standard consecutive integers class labels for additional readability. For example: `class_weights={0: 1.0, 1: 0.5, 2: 2.0}`.

{% note info %}

Class labels are extracted from dictionary keys for the following types of `class_weights`:

- {{ python-type--dict }}
- {{ python-type__collectionsOrderedDict }} (when the order of classes in the model is important)

The `class_names` parameter can be skipped when using these types.

{% endnote %}


{% note alert %}

Do not use this parameter with `auto_class_weights` and `scale_pos_weight`.

{% endnote %}



**Default value**

{{ fit--class-weights }}

**Supported processing units**


 {{ cpu-gpu }}



### class_names


**Type**

{{ python-type--list-of-strings }}

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


**Default value**


None


**Supported processing units**


 {{ cpu-gpu }}



### auto_class_weights


**Type**

{{ python-type--string }}

#### Description


{% include [reusage-cli__auto-class-weights__div](cli__auto-class-weights__div.md) %}


{% note alert %}

Do not use this parameter with `class_weights` and `scale_pos_weight`.

{% endnote %}



**Default value**

{{ autoclass__weights__default }}

**Supported processing units**


 {{ cpu-gpu }}



### scale_pos_weight


**Type**

{{ python-type--float }}

#### Description


The weight for class 1 in binary classification. The value is used as a multiplier for the weights of objects from class 1.

{% note info %}

For imbalanced datasets, the weight multiplier can be set to $\left(\frac{sum\_negative}{sum\_positive}\right)$

{% endnote %}

{% note alert %}

Do not use this parameter with `auto_class_weights` and `class_weights`.

{% endnote %}



**Default value**

1.0

**Supported processing units**


 {{ cpu-gpu }}



### boosting_type


**Type**

{{ python-type--string }}

#### Description


Boosting scheme.

Possible values:
- {{ fit__boosting-type__ordered }} — Usually provides better quality on small datasets, but it may be slower than the Plain scheme.
- {{ fit__boosting-type__plain }} — The classic gradient boosting scheme.


**Default value**


#### {{ calcer_type__cpu }}

{{ fit__boosting-type__plain }}

#### {{ calcer_type__gpu }}

- Any number of objects, {{ error-function--MultiClass }} or {{ error-function--MultiClassOneVsAll }} mode: {{ fit__boosting-type__plain }}
- More than 50 thousand objects, any mode: {{ fit__boosting-type__plain }}
- Less than or equal to 50 thousand objects, any mode but {{ error-function--MultiClass }} or {{ error-function--MultiClassOneVsAll }}: {{ fit__boosting-type__ordered }}




**Supported processing units**


 {{ cpu-gpu }}


Only the {{ fit__boosting-type__plain }} mode is supported for the {{ error-function--MultiClass }} loss on GPU


### boost_from_average


**Type**

{{ python-type--bool }}

#### Description


Initialize approximate values by best constant value for the specified loss function. Sets the value of bias to the initial best constant value.

Available for the following loss functions:
- {{ error-function--RMSE }}
- {{ error-function--Logit }}
- {{ error-function--CrossEntropy }}
- {{ error-function--Quantile }}
- {{ error-function--MAE }}
- {{ error-function--MAPE }}


**Default value**


{% include [reusage-default-values-boost-from-average](../reusage-default-values/boost-from-average.md) %}



**Supported processing units**


 {{ cpu-gpu }}



### langevin


**Type**

{{ python-type--bool }}

#### Description


Enables the Stochastic Gradient Langevin Boosting mode.

Refer to the [SGLB: Stochastic Gradient Langevin Boosting]({{ stochastic-gradient-langevin-boosting }}) paper for details.


**Default value**

False

**Supported processing units**

{{ calcer_type__cpu }}

### diffusion_temperature


**Type**

{{ python-type--float }}

#### Description


The diffusion temperature of the Stochastic Gradient Langevin Boosting mode.

Only non-negative values are supported.


**Default value**

10000

**Supported processing units**

{{ calcer_type__cpu }}

### posterior_sampling


**Type**

bool

#### Description

If this parameter is set several options are specified as follows and model parameters are checked to obtain uncertainty predictions with good theoretical properties.
Specifies options: `Langevin`: true, `DiffusionTemperature`: objects in learn pool count, `ModelShrinkRate`: 1 / (2. * objects in learn pool count)


**Default value**

False

**Supported processing units**

CPU only

### allow_const_label

#### Description

Use it to train models with datasets that have equal label values for all objects.

**Type**

{{ python-type--bool }}

**Default value**

{{ fit__allow-const-label }}

**Supported processing units**

 {{ cpu-gpu }}

### score_function

#### Description

The [score type](../../../concepts/algorithm-score-functions.md) used to select the next split during the tree construction.

Possible values:

- {{ scorefunction__Correlation }} (do not use this score type with the {{ growing_policy__Lossguide }} tree growing policy)
- {{ scorefunction__L2 }}
- {{ scorefunction__NewtonCorrelation }} (do not use this score type with the {{ growing_policy__Lossguide }} tree growing policy)
- {{ scorefunction__NewtonL2 }}

**Type**

- {{ python-type--list-of-strings }}
- {{ python-type--string }}
- {{ python-type--dict }}
- {{ python-type--list }}

**Default value**

{{ scorefunction__default }}

**Supported processing units**

The supported score functions vary depending on the processing unit type:

- {{ calcer_type__gpu }} — All score types

- {{ calcer_type__cpu }} — {{ scorefunction__Correlation }}, {{ scorefunction__L2 }}


### monotone_constraints

#### Description

{% include [reusage-cli__monotone-constraints__desc__div](cli__monotone-constraints__desc__div.md) %}

{% include [reusage-monotone-constraints__formats__intro](monotone-constraints__formats__intro.md) %}

- {% include [reusage-set-individual-constraints__div](set-individual-constraints__div.md) %}

    Zero constraints for features at the end of the list may be dropped.

    In

    ```
    monotone_constraints = "(1,0,-1)"
    ```

  an increasing constraint is set on the first feature and a decreasing one on the third. Constraints are disabled for all other features.

- {% include [reusage-monotone-constraints__formats__individually-for-required-features](monotone-constraints__formats__individually-for-required-features.md) %}

    These examples

    ```
    monotone-constraints = "2:1,4:-1"
    ```

    ```
    monotone-constraints = "Feature2:1,Feature4:-1"
    ```

    are identical, given that the name of the feature index 2 is <q>Feature2</q> and the name of the feature indexed 4 is <q>Feature4</q>.

- {% include [reusage-cli__monotone-constraints__arra-or-dict](cli__monotone-constraints__arra-or-dict.md) %}

    Array examples

    ```
    monotone_constraints = [1, 0, -1]
    ```

    These dictionary examples

    ```python
    monotone_constraints = {"Feature2":1,"Feature4":-1}
    ```

    ```python
    monotone_constraints = {"2":1, "4":-1}
    ```

    are identical, given that the name of the feature indexed 2 is <q>Feature2</q> and the name of the feature indexed 4 is <q>Feature4</q>.

**Type**

- {{ python-type--list-of-strings }}
- {{ python-type--string }}
- {{ python-type--dict }}
- {{ python-type--list }}

**Default value**

None

**Supported processing units**

{{ calcer_type__cpu }}

### feature_weights


**Type**


- {{ python-type--list }}
- {{ python-type__np_ndarray }}
- {{ python-type--string }}
- {{ python-type--dict }}


#### Description


{% include [reusage-cli__feature-weight__desc__intro](cli__feature-weight__desc__intro.md) %}


- {% include [reusage-cli__feature-weight__desc__weight-for-each-feature](cli__feature-weight__desc__weight-for-each-feature.md) %}

    In this example
    ```
    feature_weights = "(0.1,1,3)"
    ```

    the multiplication weight is set to 0.1, 1 and 3 for the first, second and third features respectively. The multiplication weight for all other features is set to 1.

- {% include [reusage-cli__feature-weight__formats__individually-for-required-features](cli__feature-weight__formats__individually-for-required-features.md) %}

 These examples

    ```
    feature_weights = "2:1.1,4:0.1"
    ```

    ```
    feature_weights = "Feature2:1.1,Feature4:0.1"
    ```

are identical, given that the name of the feature indexed 2 is <q>Feature2</q> and the name of the feature indexed 4 is <q>Feature4</q>.
- {% include [reusage-cli__feature-weights__arra-or-dict](cli__feature-weights__arra-or-dict.md) %}

Array examples

    ```
    feature_weights = [0.1, 1, 3]
    ```
These dictionary examples

    ```python
    feature_weights = {"Feature2":1.1,"Feature4":0.3}
    ```

    ```python
    feature_weights = {"2":1.1, "4":0.3}
    ```

are identical, given that the name of the feature indexed 2 is <q>Feature2</q> and the name of the feature indexed 4 is <q>Feature4</q>.



**Default value**

1 for all features

**Supported processing units**

{{ calcer_type__cpu }}

### first_feature_use_penalties


**Type**


- {{ python-type--list }}
- {{ python-type__np_ndarray }}
- {{ python-type--string }}
- {{ python-type--dict }}


#### Description


{% include [reusage-cli__first-feature-use-penalties__intro](cli__first-feature-use-penalties__intro.md) %}


{% include [penalties-format-penalties_format](../reusage-python/penalties_format.md) %}



**Default value**

0 for all features

**Supported processing units**

{{ calcer_type__cpu }}

## fixed_binary_splits

**Type**

{{ python-type--list }}

#### Description


A list of indices of binary features to put at the top of each tree; ignored if `grow_policy` is `Symmetric`.

**Default value**

 None

**Supported processing units**

 {{ calcer_type__gpu }}

### penalties_coefficient


**Type**

{{ python-type--float }}

#### Description


A single-value common coefficient to multiply all penalties.

Non-negative values are supported.


**Default value**

1

**Supported processing units**

{{ calcer_type__cpu }}

### per_object_feature_penalties


**Type**


- {{ python-type--list }}
- {{ python-type__np_ndarray }}
- {{ python-type--string }}
- {{ python-type--dict }}


#### Description


{% include [reusage-per-object-feature-penalties__intro](per-object-feature-penalties__intro.md) %}


{% include [penalties-format-penalties_format](../reusage-python/penalties_format.md) %}



**Default value**

0 for all objects

**Supported processing units**

{{ calcer_type__cpu }}

### model_shrink_rate


**Type**

{{ python-type--float }}

#### Description


The constant used to calculate the coefficient for multiplying the model on each iteration.
The actual model shrinkage coefficient calculated at each iteration depends on the value of the `--model-shrink-mode`for the Command-line version parameter. The resulting value of the coefficient should be always in the range (0, 1].

**Default value**


The default value depends on the values of the following parameters:
- `--model-shrink-mode`for the Command-line version

- `--monotone-constraints`for the Command-line version

**Supported processing units**

{{ calcer_type__cpu }}

### model_shrink_mode


**Type**

{{ python-type--string }}

#### Description


Determines how the actual model shrinkage coefficient is calculated at each iteration.

Possible values:
- {{ model_shrink_mode__Constant }}:

    $1 - model\_shrink\_rate \cdot learning\_rate$
    - $model\_shrink\_rate$ is the value of the `--model-shrink-rate`for the Command-line version parameter.
    - $learning\_rate$ is the value of the `--learning-rate`for the Command-line version parameter

- {{ model_shrink_mode__Decreasing }}:

    $1 - \frac{model\_shrink\_rate}{i}$

    - $model\_shrink\_rate$ is the value of the `--model-shrink-rate`for the Command-line version parameter.
    - $i$ is the identifier of the iteration.

**Default value**

{{ model_shrink_mode__Constant }}

**Supported processing units**

{{ calcer_type__cpu }}


## Text processing parameters

### tokenizers

#### Description

{% include [reusage-cli__tokenizers__desc__div](cli__tokenizers__desc__div.md) %}

```json
[{
'TokenizerId1': <value>,
'option_name_1': <value>,
..
'option_name_N': <value>,}]
```

- `TokenizerId` — The unique name of the tokenizer.
- `option_name` — One of the [supported tokenizer options](../../../references/tokenizer_options.md).

This parameter works with `dictionaries` and `feature_calcers` parameters.

{% include [reusage-tokenizer-dictionaries-feature-calcers__note_div](tokenizer-dictionaries-feature-calcers__note_div.md) %}

Usage example

```python
tokenizers = [{
	'tokenizerId': 'Space',
	'delimiter': ' ',
	'separator_type': 'ByDelimiter',
},{
	'tokenizerId': 'Sense',
	'separator_type': 'BySense',
}]
```

**Type**

{{ python-type__list-of-json }}

**Default value**

–

**Supported processing units**


{% include [reusage-python-gpu](../reusage-python/gpu.md) %}



### dictionaries

#### Description


{% include [reusage-cli__dictionaries__desc__div](cli__dictionaries__desc__div.md) %}


```
[{
'dictionaryId1': <value>,
'option_name_1': <value>,
..
'option_name_N': <value>,}]
```

- `DictionaryId` — The unique name of dictionary.
- `option_name` — One of the [supported dictionary options](../../../references/dictionaries_options.md).

{% note info %}

This parameter works with `tokenizers` and `feature_calcers` parameters.

{% include [reusage-tokenizer-dictionaries-feature-calcers__note_div](tokenizer-dictionaries-feature-calcers__note_div.md) %}

{% endnote %}


Usage example

```python
dictionaries = [{
	'dictionaryId': 'Unigram',
	'max_dictionary_size': '50000',
	'gram_count': '1',
},{
	'dictionaryId': 'Bigram',
	'max_dictionary_size': '50000',
	'gram_count': '2',
}]
```

**Type**

{{ python-type__list-of-json }}


**Default value**

–

**Supported processing units**


{% include [reusage-python-gpu](../reusage-python/gpu.md) %}



### feature_calcers

#### Description


{% include [reusage-cli__feature-calcers__desc__div](cli__feature-calcers__desc__div.md) %}


```json
['FeatureCalcerName[:option_name=option_value],
]
```

- `FeatureCalcerName` — The required [feature calcer](../../../references/text-processing__feature_calcers.md).

- `option_name` — Additional options for feature calcers. Refer to the [list of supported calcers](../../../references/text-processing__feature_calcers.md) for details on options available for each of them.


{% note info %}

This parameter works with `tokenizers` and `dictionaries` parameters.

{% include [reusage-tokenizer-dictionaries-feature-calcers__note_div](tokenizer-dictionaries-feature-calcers__note_div.md) %}

{% cut "Usage example" %}

```python
feature_calcers = [
	'BoW:top_tokens_count=1000',
	'NaiveBayes',
]
```

{% endcut %}

{% endnote %}

**Type**

{{ python-type--list-of-strings }}

**Default value**

–

**Supported processing units**


{% include [reusage-python-gpu](../reusage-python/gpu.md) %}



### text_processing

#### Description

{% include [reusage-cli__text-processing__div](cli__text-processing__div.md) %}


- `tokenizers`
- `dictionaries`
- `feature_calcers`

{% note alert %}

Do not use this parameter with the following ones:

- `tokenizers`
- `dictionaries`
- `feature_calcers`

{% endnote %}


**Type**

{{ python-type__json }}

**Default value**

[Default value](../../../references/text-processing__test-processing__default-value.md)

**Supported processing units**


{% include [reusage-python-gpu](../reusage-python/gpu.md) %}



## Overfitting detection settings

### early_stopping_rounds

{{ python-type--int }}Sets the overfitting detector type to {{ fit--od-type-iter }} and stops the training after the specified number of iterations since the iteration with the optimal metric value.False
 {{ cpu-gpu }}


### od_type

#### Description

The type of the overfitting detector to use.

Possible values:
- {{ fit--od-type-inctodec }}
- {{ fit--od-type-iter }}


**Type**

{{ python-type--string }}

**Default value**

{{ fit--od-type-inctodec }}

**Supported processing units**


 {{ cpu-gpu }}

### od_pval

#### Description
The threshold for the {{ fit--od-type-inctodec }} [overfitting detector](../../../concepts/overfitting-detector.md) type. The training is stopped when the specified value is reached. Requires that a validation dataset was input.

For best results, it is recommended to set a value in the range $[10^{–10}; 10^{-2}]$.

The larger the value, the earlier overfitting is detected.

{% note alert %}

Do not use this parameter with the {{ fit--od-type-iter }} overfitting detector type.

{% endnote %}



**Default value**

{{ fit--auto_stop_pval }}

**Supported processing units**


 {{ cpu-gpu }}



### od_wait


**Type**

{{ python-type--int }}

#### Description

The number of iterations to continue the training after the iteration with the optimal metric value.
The purpose of this parameter differs depending on the selected overfitting detector type:
- {{ fit--od-type-inctodec }} — Ignore the overfitting detector when the threshold is reached and continue learning for the specified number of iterations after the iteration with the optimal metric value.
- {{ fit--od-type-iter }} — Consider the model overfitted and stop training after the specified number of iterations since the iteration with the optimal metric value.


**Default value**

{{ fit--od-wait }}

**Supported processing units**


 {{ cpu-gpu }}



## Quantization settings

### target_border


**Type**

{{ python-type--float }}

#### Description


If set, defines the border for converting target values to 0 and 1.

Depending on the specified value:

- $target\_value \le border\_value$ the target is converted to 0
- $target\_value > border\_value$ the target is converted to 1


**Default value**

None

**Supported processing units**


 {{ cpu-gpu }}



### border_count


_Alias:_`max_bin`

**Type**

{{ python-type--int }}

#### Description


The number of splits for numerical features. Allowed values are integers from 1 to 65535 inclusively.


**Default value**


{% include [reusage-default-values-border_count](../reusage-default-values/border_count.md) %}



**Supported processing units**


 {{ cpu-gpu }}



### feature_border_type


**Type**

{{ python-type--string }}

#### Description


The [quantization mode](../../../concepts/quantization.md) for numerical features.

Possible values:
- Median
- Uniform
- UniformAndQuantiles
- MaxLogSum
- MinEntropy
- GreedyLogSum


**Default value**

{{ fit--feature_border_type }}

**Supported processing units**


 {{ cpu-gpu }}



### per_float_feature_quantization


**Type**


- {{ python-type--string }}
- {{ python-type--list-of-strings }}


#### Description


The quantization description for the specified feature or list of features.

Description format for a single feature:

```
FeatureId[:border_count=BorderCount][:nan_mode=BorderType][:border_type=border_selection_method]
```

Examples:

- ```
    per_float_feature_quantization='0:border_count=1024'
    ```

    In this example, the feature indexed 0 has 1024 borders.

- ```python
    per_float_feature_quantization=['0:border_count=1024', '1:border_count=1024']
    ```

    In this example, features indexed 0 and 1 have 1024 borders.



**Default value**

None

**Supported processing units**


 {{ cpu-gpu }}



## Multiclassification settings

### classes_count


**Type**

{{ python-type--int }}

#### Description


{% include [reusage-classes-count__main-desc](classes-count__main-desc.md) %}

{% include [reusage-classes-count__possible-values](classes-count__possible-values.md) %}



If this parameter is specified the labels for all classes in the input dataset should be smaller than the given value


**Default value**


None.

Calculation principles

`{{ fit--classes-count }}`




**Supported processing units**


 {{ cpu-gpu }}



## Performance settings

### thread_count


**Type**

{{ python-type--int }}

#### Description


The number of threads to use during the training.

- **For CPU**

    Optimizes the speed of execution. This parameter doesn't affect results.

- **For GPU**
    The given value is used for reading the data from the hard drive and does not affect the training.

    During the training one main thread and one thread for each GPU are used.


**Default value**

{{ fit__thread_count__wrappers }}

**Supported processing units**


 {{ cpu-gpu }}



### used_ram_limit


**Type**

{{ python-type--int }}

#### Description

Attempt to limit the amount of used CPU RAM.

{% note alert %}

- This option affects only the CTR calculation memory usage.
- In some cases it is impossible to limit the amount of CPU RAM used in accordance with the specified value.

{% endnote %}


Format:
```
<size><measure of information>
```

Supported measures of information (non case-sensitive):
- MB
- KB
- GB

For example:
```
2gb
```


**Default value**

{{ fit__used-ram-limit }}

**Supported processing units**


{{ calcer_type__cpu }}



### gpu_ram_part


**Type**

{{ python-type--float }}

#### Description


How much of the GPU RAM to use for training.


**Default value**

{{ fit__gpu__gpu-ram-part }}

**Supported processing units**


{% include [reusage-python-gpu](../reusage-python/gpu.md) %}



### pinned_memory_size


**Type**

{{ python-type--int }}

#### Description


How much pinned (page-locked) CPU RAM to use per GPU.

The value should be a positive integer or `inf`. Measure of information can be defined for integer values.

Format:

```
<size><measure of information>
```

Supported measures of information (non case-sensitive):
- MB
- KB
- GB

For example:

```
2gb
```


**Default value**

{{ fit__gpu__pinned-memory-size }}

**Supported processing units**


{% include [reusage-python-gpu](../reusage-python/gpu.md) %}



### gpu_cat_features_storage


**Type**

{{ python-type--string }}

#### Description


The method for storing the categorical features' values.

Possible values:

- {{ fit__gpu__gpu_cat_features_storage__value__CpuPinnedMemory }}
- {{ fit__gpu__gpu_cat_features_storage__value__GpuRam }}

{% note info %}

Use the {{ fit__gpu__gpu_cat_features_storage__value__CpuPinnedMemory }} value if feature combinations are used and the available GPU RAM is not sufficient.

{% endnote %}



**Default value**

None (set to {{ fit__gpu__use-cpu-ram-for-catfeatures }})

**Supported processing units**


{% include [reusage-python-gpu](../reusage-python/gpu.md) %}



### data_partition


**Type**

{{ python-type--string }}

#### Description


The method for splitting the input dataset between multiple workers.

Possible values:
- {{ fit__gpu__data-partition__mode__FeatureParallel }} — Split the input dataset by features and calculate the value of each of these features on a certain GPU.

    For example:

    - GPU0 is used to calculate the values of features indexed 0, 1, 2
    - GPU1 is used to calculate the values of features indexed 3, 4, 5, etc.

- {{ fit__gpu__data-partition__mode__DocParallel }} — Split the input dataset by objects and calculate all features for each of these objects on a certain GPU. It is recommended to use powers of two as the value for optimal performance.

    For example:
    - GPU0 is used to calculate all features for objects indexed `object_1`, `object_2`
    - GPU1 is used to calculate all features for objects indexed `object_3`, `object_4`, etc.


**Default value**

{{ fit__gpu__data-partition }}

**Supported processing units**


{% include [reusage-python-gpu](../reusage-python/gpu.md) %}



## Processing unit settings

### task_type


**Type**

{{ python-type--string }}

#### Description


The processing unit type to use for training.

Possible values:
- CPU
- GPU


**Default value**

{{ fit__python-r__calcer_type }}

**Supported processing units**


 {{ cpu-gpu }}



### devices


**Type**

{{ python-type--string }}

#### Description


IDs of the GPU devices to use for training (indices are zero-based).

Format

- `<unit ID>` for one device (for example, `3`)
- `<unit ID1>:<unit ID2>:..:<unit IDN>` for multiple devices (for example, `devices='0:1:3'`)
- `<unit ID1>-<unit IDN>` for a range of devices (for example, `devices='0-3'`)


**Default value**

{{ fit__python-r__device_config }}

**Supported processing units**


{% include [reusage-python-gpu](../reusage-python/gpu.md) %}



## Visualization settings

### name


**Type**

{{ python-type--string }}

#### Description

The experiment name to display in [visualization tools](../../../features/visualization.md).

**Default value**

{{ fit--name }}

**Supported processing units**


 {{ cpu-gpu }}



## Output settings

### logging_level


**Type**

{{ python-type--string }}

#### Description


The logging level to output to stdout.

Possible values:
- Silent — Do not output any logging information to stdout.

- Verbose — Output the following data to stdout:

    - optimized metric
    - elapsed time of training
    - remaining time of training

- Info — Output additional information and the number of trees.

- Debug — Output debugging information.


**Default value**

None (corresponds to the {{ fit--verbose }} logging level)

**Supported processing units**


 {{ cpu-gpu }}



### metric_period


**Type**

{{ python-type--int }}

#### Description


The frequency of iterations to calculate the values of [objectives and metrics](../../../concepts/loss-functions.md). The value should be a positive integer.

The usage of this parameter speeds up the training.

{% note info %}

It is recommended to increase the value of this parameter to maintain training speed if a GPU processing unit type is used.

{% endnote %}



**Default value**

{{ fit__metric-period }}

**Supported processing units**


 {{ cpu-gpu }}



### verbose


_Alias:_`verbose_eval`

**Type**


- {{ python-type--bool }}
- {{ python-type--int }}


#### Description


{% include [sections-with-methods-desc-python__feature-importances__verbose__short-description__list-intro](python__feature-importances__verbose__short-description__list-intro.md) %}


- {{ python-type--bool }} — Defines the logging level:

    - <q>True</q>  corresponds to the Verbose logging level
    - <q>False</q> corresponds to the Silent logging level

- {{ python-type--int }} — Use the Verbose logging level and set the logging period to the value of this parameter.

{% note alert %}

Do not use this parameter with the `logging_level` parameter.

{% endnote %}



**Default value**

{{ train_verbose_fr-of-iterations-to-output__default }}

**Supported processing units**


 {{ cpu-gpu }}



### train_final_model


**Type**


bool


#### Description


If specified, then the model with selected features will be trained after features selection.


**Default value**


True


**Supported processing units**


 {{ cpu-gpu }}



### train_dir


**Type**

{{ python-type--string }}

#### Description


The directory for storing the files generated during training.


**Default value**

{{ fit--train_dir }}

**Supported processing units**


 {{ cpu-gpu }}



### model_size_reg


**Type**

{{ python-type--float }}

#### Description


The model size regularization coefficient. The larger the value, the smaller the model size. Refer to the [Model size regularization coefficient](../../../references/model-size-reg.md) section for details.

Possible values are in the range $[0; \inf)$.

This regularization is needed only for models with categorical features (other models are small). Models with categorical features might weight tens of gigabytes or more if categorical features have a lot of values. If the value of the regularizer differs from zero, then the usage of categorical features or feature combinations with a lot of values has a penalty, so less of them are used in the resulting model.

Note that the resulting quality of the model can be affected. Set the value to 0 to turn off the model size optimization option.


**Default value**

None ({{ fit__model_size_reg }})

**Supported processing units**


 {{ cpu-gpu }}



### allow_writing_files


**Type**

{{ python-type--bool }}

#### Description


Allow to write analytical and snapshot files during training.

If set to <q>False</q>, the [snapshot](../../../features/snapshots.md) and [data visualization](../../../features/visualization.md) tools are unavailable.


**Default value**

{{ fit--allow-writing-files }}

**Supported processing units**


 {{ cpu-gpu }}



### save_snapshot


**Type**

{{ python-type--bool }}

#### Description


Enable snapshotting for [restoring the training progress after an interruption](../../../features/snapshots.md). If enabled, the default period for making snapshots is {{ fit__snapshot-interval__default }} seconds. Use the `snapshot_interval` parameter to change this period.

{% include [reusage-common-phrases-snapshot-not-working-for-cv](../reusage-common-phrases/snapshot-not-working-for-cv.md) %}



**Default value**

{{ fit--save_snapshot }}

**Supported processing units**


 {{ cpu-gpu }}



### snapshot_file


**Type**

{{ python-type--string }}

#### Description


The name of the file to save the training progress information in. This file is used for [recovering training after an interruption](../../../features/snapshots.md).

{% include [reusage-snapshot-filename-desc](snapshot-filename-desc.md) %}


{% include [reusage-common-phrases-snapshot-not-working-for-cv](../reusage-common-phrases/snapshot-not-working-for-cv.md) %}



**Default value**


experiment

{{ fit--snapshot-file-python }}




**Supported processing units**


 {{ cpu-gpu }}



### snapshot_interval


**Type**

{{ python-type--int }}

#### Description


The interval between saving snapshots in seconds.

The first snapshot is taken after the specified number of seconds since the start of training. Every subsequent snapshot is taken after the specified number of seconds since the previous one. The last snapshot is taken at the end of the training.

{% include [reusage-common-phrases-snapshot-not-working-for-cv](../reusage-common-phrases/snapshot-not-working-for-cv.md) %}



**Default value**

{{ fit__snapshot-interval__default }}

**Supported processing units**


 {{ cpu-gpu }}



### roc_file


**Type**

{{ python-type--string }}

#### Description


The name of the [output file](../../../concepts/output-data_roc-curve-points.md) to save the ROC curve points to. This parameter can only be set in [cross-validation](../../../concepts/python-reference_cv.md) mode if the {{ error-function--Logit }} loss function is selected. The ROC curve points are calculated for the test fold.

The output file is saved to the `catboost_info` directory.


**Default value**

None (the file is not saved)

**Supported processing units**


 {{ cpu-gpu }}



## CTR settings

### simple_ctr


**Type**

{{ python-type--string }}

#### Description


{% include [reusage-cli__simple-ctr__intro](cli__simple-ctr__intro.md) %}


Format:

```
['CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
 'CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
  ...]
```

{% include [reusage-cli__simple-ctr__components](cli__simple-ctr__components.md) %}


{% include [reusage-cli__simple-ctr__examples__p](cli__simple-ctr__examples__p.md) %}



**Supported processing units**


 {{ cpu-gpu }}



### combinations_ctr


**Type**

{{ python-type--string }}

#### Description


{% include [reusage-cli__combination-ctr__intro](cli__combination-ctr__intro.md) %}


```
['CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
 'CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
  ...]
```

{% include [reusage-cli__combination-ctr__components](cli__combination-ctr__components.md) %}



**Supported processing units**


 {{ cpu-gpu }}



### per_feature_ctr


**Type**

{{ python-type--string }}

#### Description


{% include [reusage-cli__per-feature-ctr__intro](cli__per-feature-ctr__intro.md) %}


```
['FeatureId:CtrType:[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
 'FeatureId:CtrType:[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
  ...]
```

{% include [reusage-cli__per-feature-ctr__components](cli__per-feature-ctr__components.md) %}



**Supported processing units**


 {{ cpu-gpu }}



### ctr_target_border_count


**Type**

{{ python-type--int }}

#### Description


{% include [reusage-cli__ctr-target-border-count__short-desc](cli__ctr-target-border-count__short-desc.md) %}


The value of the `{{ ctr-types__TargetBorderCount }}` component overrides this parameter if it is specified for one of the following parameters:

- `simple_ctr`
- `combinations_ctr`
- `per_feature_ctr`


**Default value**

{{ parameters__ctr-target-border-count__default }}

**Supported processing units**


 {{ cpu-gpu }}



### counter_calc_method


**Type**

{{ python-type--string }}

#### Description


The method for calculating the Counter CTR type.

Possible values:

- {{ counter-calculation-method--static }} — Objects from the validation dataset are not considered at all
- {{ counter-calculation-method--full }} — All objects from both learn and validation datasets are considered


**Default value**

None ({{ fit--counter-calc-method }} is used)

**Supported processing units**


 {{ cpu-gpu }}



### max_ctr_complexity


**Type**

{{ python-type--int }}

#### Description


The maximum number of features that can be combined.

Each resulting combination consists of one or more categorical features and can optionally contain binary features in the following form: <q>numeric feature > value</q>.


**Default value**


{% include [reusage-default-values-max_xtr_complexity](../reusage-default-values/max_xtr_complexity.md) %}



**Supported processing units**


 {{ cpu-gpu }}



### ctr_leaf_count_limit


**Type**

{{ python-type--int }}

#### Description


The maximum number of leaves with categorical features. If the quantity exceeds the specified value a part of leaves is discarded.

The leaves to be discarded are selected as follows:

1. The leaves are sorted by the frequency of the values.
1. The top `N` leaves are selected, where N is the value specified in the parameter.
1. All leaves starting from `N+1` are discarded.

This option reduces the resulting model size and the amount of memory required for training. Note that the resulting quality of the model can be affected.


**Default value**


None

{{ fit--ctr-leaf-count-limit }}


**Supported processing units**


{{ calcer_type__cpu }}



### store_all_simple_ctr


**Type**

{{ python-type--bool }}

#### Description


Ignore categorical features, which are not used in feature combinations, when choosing candidates for exclusion.

There is no point in using this parameter without the `--ctr-leaf-count-limit`for the Command-line version parameter.


**Default value**


None (set to False)

{{ fit--store-all-simple-ctr }}


**Supported processing units**


{{ calcer_type__cpu }}



### final_ctr_computation_mode


**Type**

{{ python-type--string }}

#### Description


Final CTR computation mode.

Possible values:

- {{ cli__fit__final-ctr-computation-mode__possible-values__Default }} — Compute final CTRs for learn and validation datasets.

- {{ cli__fit__final-ctr-computation-mode__possible-values__Skip }} — Do not compute final CTRs for learn and validation datasets. In this case, the resulting model can not be applied. This mode decreases the size of the resulting model. It can be useful for research purposes when only the metric values have to be calculated.


**Default value**

{{ cli__fit__final-ctr-computation-mode__default }}

**Supported processing units**


 {{ cpu-gpu }}
