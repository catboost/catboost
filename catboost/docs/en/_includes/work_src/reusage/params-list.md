## Common parameters

### loss_function


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

{% include [reusage-loss-function--example](loss-function--example.md) %}



**Default value**

{{ fit--loss_function }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### custom_loss


#### Description


{% include [reusage-custom-loss--basic](custom-loss--basic.md) %}


Format:
```
<Metric>[:<parameter 1>=<value>;..;<parameter N>=<value>]
```

[Supported metrics](../../../references/custom-metric__supported-metrics.md)

Examples:
- Calculate the value of {{ error-function--CrossEntropy }}:

    ```
    c('CrossEntropy')
    ```

    Or simply:
    ```
    'CrossEntropy'
    ```

- Calculate the values of {{ error-function--Logit }} and {{ error-function--AUC }}:

    ```
    c('{{ error-function--Logit }}', '{{ error-function--AUC }}')
    ```

- Calculate the value of {{ error-function--Quantile }} with the coefficient $\alpha = 0.1$
    ```
    c('{{ error-function--Quantile }}:alpha=0.1')
    ```

{% include [reusage-custom-loss--values-saved-to](custom-loss--values-saved-to.md) %}



**Default value**

{{ fit--custom_loss }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### eval_metric


#### Description


{% include [reusage-eval-metric--basic](eval-metric--basic.md) %}


{% include [reusage-eval-metric--format](eval-metric--format.md) %}


[Supported metrics](../../../references/eval-metric__supported-metrics.md)

```
Quantile:alpha=0.3
```


**Default value**

{{ fit--eval-metric }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### iterations


#### Description


The maximum number of trees that can be built when solving machine learning problems.

When using other parameters that limit the number of iterations, the final number of trees may be less than the number specified in this parameter.


**Default value**

{{ fit--iterations }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### learning_rate


#### Description


The learning rate.

Used for reducing the gradient step.


**Default value**


The default value is defined automatically for {{ error-function--Logit }}, {{ error-function--MultiClass }} & {{ error-function--RMSE }} loss functions depending on the number of iterations if none of
{% cut "these parameters" %}

- `leaf_estimation_iterations`
- `--leaf-estimation-method`
- `l2_leaf_reg`

{% endcut %}

 is set. In this case, the selected learning rate is printed to stdout and saved in the model.

In other cases, the default value is 0.03.


**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### random_seed


#### Description


The random seed used for training.


**Default value**

{{ fit--random_seed }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



###  l2_leaf_reg


#### Description

Coefficient at the L2 regularization term of the cost function.
Any positive value is allowed.


**Default value**

{{ fit--l2-leaf-reg }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### bootstrap_type


#### Description


[Bootstrap type](../../../concepts/algorithm-main-stages_bootstrap-options.md). Defines the method for sampling the weights of objects.

Supported methods:

- {{ fit__bootstrap-type__Bayesian }}
- {{ fit__bootstrap-type__Bernoulli }}
- {{ fit__bootstrap-type__MVS }}
- {{ fit__bootstrap-type__Poisson }} (supported for GPU only)
- {{ fit__bootstrap-type__No }}


**Default value**

The default value depends on the selected mode and processing unit type:

- {{ error-function__QueryCrossEntropy }}, {{ error-function__YetiRankPairwise }}: {{ fit__bootstrap-type__Bernoulli }} with the `subsample` parameter set to 0.5.
- {{ error-function--MultiClass }} and {{ error-function--MultiClassOneVsAll }}:  {{ fit__bootstrap-type__Bayesian }}.
- Other modes:
  - GPU: Bayesian
  - CPU: MVS with the subsample parameter set to 0.8.

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### bagging_temperature


#### Description


Defines the settings of the Bayesian bootstrap. It is used by default in classification and regression modes.

Use the Bayesian bootstrap to assign random weights to objects.

The weights are sampled from exponential distribution if the value of this parameter is set to <q>1</q>. All weights are equal to 1 if the value of this parameter is set to <q>0</q>.

Possible values are in the range $[0; \inf)$. The higher the value the more aggressive the bagging is.

This parameter can be used if the selected bootstrap type is {{ fit__bootstrap-type__Bayesian }}.


**Default value**

{{ fit__bagging-temperature }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### subsample


#### Description


Sample rate for bagging.

This parameter can be used if one of the following bootstrap types is selected:
- {{ fit__bootstrap-type__Poisson }}
- {{ fit__bootstrap-type__Bernoulli }}
- {{ fit__bootstrap-type__MVS }}


**Default value**


{% include [reusage-default-values-subsample__default](../reusage-default-values/subsample__default.md) %}



**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### sampling_frequency


#### Description


Frequency to sample weights and objects when building trees.

Supported values:

- {{ fit__sampling-frequency__PerTree }} — Before constructing each new tree
- {{ fit__sampling-frequency__PerTreeLevel }} — Before choosing each new split of a tree


**Default value**

{{ fit__sampling-frequency }}

**{{ cli__params-table__title__processing-units-type }}**


{{ calcer_type__cpu }}


### sampling_unit


#### Description


The sampling scheme.

Possible values:
- {{ python__ESamplingUnit__type__Object }} — The weight $w_{i}$ of the i-th object $o_{i}$ is used for sampling the corresponding object.
- {{ python__ESamplingUnit__type__Group }} — The weight $w_{j}$ of the group $g_{j}$ is used for sampling each object $o_{i_{j}}$ from the group $g_{j}$.


**Default value**

{{ python__ESamplingUnit__type__default }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### mvs_reg


#### Description


{% include [reusage-cli__mvs-head-fraction__div](cli__mvs-head-fraction__div.md) %}


{% include [python-python__mvs_reg__desc__note](python__mvs_reg__desc__note.md) %}



**Default value**

The value is {{ fit__mvs_head_fraction }}

**{{ cli__params-table__title__processing-units-type }}**

{{ calcer_type__cpu }}

### random_strength


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

**{{ cli__params-table__title__processing-units-type }}**

{{ calcer_type__cpu }}

### use_best_model


#### Description


If this parameter is set, the number of trees that are saved in the resulting model is defined as follows:
1. Build the number of trees defined by the training parameters.
1. Use the validation dataset to identify the iteration with the optimal value of the metric specified in  `--eval-metric` (`--eval-metric`).

No trees are saved after this iteration.

This option requires a validation dataset to be provided.


**Default value**


True if a validation set is input (the `train_pool` parameter is defined) and at least one of the label values of objects in this set differs from the others. False otherwise.


**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### best_model_min_trees


#### Description


{% include [reusage-clii__best-model-min-trees__short-desc](clii__best-model-min-trees__short-desc.md) %}


Should be used with the `--use-best-model` parameter.


**Default value**

None ({{ fit__best-model-min-trees }})

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### train_pool


#### Description


The validation set for the following processes:
- [overfitting detector](../../../concepts/overfitting-detector.md)
- best iteration selection
- monitoring metrics' changes


**Default value**

None

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### depth


#### Description


Depth of the trees.

The range of supported values depends on the processing unit type and the type of the selected loss function:
- CPU — Any integer up to  {{ fit--maxtree }}.

- GPU — Any integer up to {{ fit__maxtree__pairwise }}pairwise modes ({{ error-function__YetiRank }}, {{ error-function__PairLogitPairwise }} and {{ error-function__QueryCrossEntropy }}) and up to   {{ fit--maxtree }} for all other loss functions.


**Default value**

{{ fit--depth }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### grow_policy


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

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### min_data_in_leaf


#### Description

The minimum number of training samples in a leaf. {{ product }} does not search for new splits in leaves with samples count less than the specified value.
Can be used only with the {{ growing_policy__Lossguide }} and {{ growing_policy__Depthwise }} growing policies.


**Default value**

{{ min-samples-in-leaf__default }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### max_leaves


#### Description


The maximum number of leafs in the resulting tree. Can be used only with the {{ growing_policy__Lossguide }} growing policy.

{% note info %}

It is not recommended to use values greater than 64, since it can significantly slow down the training process.

{% endnote %}


**Default value**

{{ max-leaves-count__default }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### ignored_features


#### Description

Feature indices to exclude from the training.

{% include [reusage-cli__ignored_features__specifics](cli__ignored_features__specifics.md) %}

The identifiers of features to exclude should be enumerated at vector.

For example, if training should exclude features with the identifiers 1, 2, 7, 42, 43, 44, 45, the value of this parameter should be set to `c(1,2,7,42,43,44,45)`.

**Default value**

None

**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}

### one_hot_max_size


#### Description


Use one-hot encoding for all categorical features with a number of different values less than or equal to the given parameter value. Ctrs are not calculated for such features.

See [details](../../../features/categorical-features.md).


**Default value**


{% include [reusage-default-values-one-hot-max-size-default](../reusage-default-values/one-hot-max-size-default.md) %}


**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### has_time


#### Description


Use the order of objects in the input data (do not perform random permutations during the [Transforming categorical features to numerical features](../../../concepts/algorithm-main-stages_cat-to-numberic.md) and [Choosing the tree structure](../../../concepts/algorithm-main-stages_choose-tree-structure.md) stages).

The {{ cd-file__col-type__Timestamp }} column type is used to determine the order of objects if specified in the [input data](../../../concepts/input-data.md).


**Default value**

{{ fit--has_time-r }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### rsm


#### Description


Random subspace method. The percentage of features to use at each split selection, when features are selected over again at random.

The value must be in the range (0;1].


**Default value**

{{ fit--rsm }}

**{{ cli__params-table__title__processing-units-type }}**


{{ calcer_type__cpu }} and {{ calcer_type__gpu }} for pairwise ranking



### nan_mode


#### Description


The method for  [processing missing values](../../../concepts/algorithm-missing-values-processing.md) in the input dataset.

{% include [reusage-cmd__nan-mode__div](cmd__nan-mode__div.md) %}



**Default value**

{{ fit--nan_mode }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### fold_permutation_block


#### Description


Objects in the dataset are grouped in blocks before the random permutations. This parameter defines the size of the blocks. The smaller is the value, the slower is the training. Large values may result in quality degradation.


**Default value**

{{ fit--fold-permutation-block }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### leaf_estimation_method


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


**{{ cli__params-table__title__processing-units-type }}**


- The {{ fit__leaf_estimation_method__Exact }} method is available only on {{ calcer_type__cpu }}
- All other methods are available on both {{ calcer_type__cpu }} and {{ calcer_type__gpu }}


### leaf_estimation_iterations


#### Description


{{ product }} might calculate leaf values using several gradient or newton steps instead of a single one.

This parameter regulates how many steps are done in every tree when calculating leaf values.


**Default value**

{{ fit--gradient_iterations }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### leaf_estimation_backtracking


#### Description


When the value of the `--leaf-estimation-iterations`for the Command-line version  parameter is greater than 1, {{ product }} makes several gradient or newton steps when calculating the resulting leaf values of a tree.

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

**{{ cli__params-table__title__processing-units-type }}**

Depends on the selected value

### name


#### Description

The experiment name to display in [visualization tools](../../../features/visualization.md).

**Default value**

{{ fit--name }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### fold_len_multiplier


#### Description


Coefficient for changing the length of folds.

The value must be greater than 1. The best validation result is achieved with minimum values.

With values close to 1 (for example, $1+\epsilon$), each iteration takes a quadratic amount of memory and time for the number of objects in the iteration. Thus, low values are possible only when there is a small number of objects.


**Default value**

{{ fit--fold-len-multiplier }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### approx_on_full_history


#### Description


{% include [reusage-cli__approx-on-full-history__first-sentence](cli__approx-on-full-history__first-sentence.md) %}


Possible values:
- <q>TRUE</q> — Use all the preceding rows in the fold for calculating the approximated values. This mode is slower and in rare cases slightly more accurate.
- <q>FALSE</q> — Use only а fraction of the fold for calculating the approximated values. The size of the fraction is calculated as follows: $\frac{1}{{X}}$, where `X` is the specified coefficient for changing the length of folds. This mode is faster and in rare cases slightly less accurate


**Default value**

TRUE

**{{ cli__params-table__title__processing-units-type }}**


{{ calcer_type__cpu }}



### class_weights


#### Description


{% include [reusage-class-weights__short-desc-intro](class-weights__short-desc-intro.md) %}


For example, `class_weights <- c(0.1, 4)`multiplies the weights of objects from class 0 by 0.1 and the weights of objects from class 1 by 4.

{% note alert %}

Do not use this parameter with `auto_class_weights`.

{% endnote %}


**Default value**

{{ fit--class-weights }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### auto_class_weights


#### Description


{% include [reusage-cli__auto-class-weights__div](cli__auto-class-weights__div.md) %}


{% note alert %}

Do not use this parameter with `class_weights`.

{% endnote %}



**Default value**

{{ autoclass__weights__default }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### boosting_type


#### Description


Boosting scheme.

Possible values:
- {{ fit__boosting-type__ordered }} — Usually provides better quality on small datasets, but it may be slower than the Plain scheme.
- {{ fit__boosting-type__plain }} — The classic gradient boosting scheme.


**Default value**

Depends on the processing unit type, the number of objects in the training dataset and the selected learning mode:

- {{{ calcer_type__cpu }}:
    - {{ fit__boosting-type__plain }}
- {{ calcer_type__gpu }}:
    - Any number of objects, {{ error-function--MultiClass }} or {{ error-function--MultiClassOneVsAll }} mode: {{ fit__boosting-type__plain }}
    - More than 50 thousand objects, any mode: {{ fit__boosting-type__plain }}
    - Less than or equal to 50 thousand objects, any mode but {{ error-function--MultiClass }} or {{ error-function--MultiClassOneVsAll }}: {{ fit__boosting-type__ordered }}


**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}

Only the {{ fit__boosting-type__plain }} mode is supported for the {{ error-function--MultiClass }} loss on GPU


### boost_from_average


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



**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### langevin


#### Description


Enables the Stochastic Gradient Langevin Boosting mode.

Refer to the [SGLB: Stochastic Gradient Langevin Boosting]({{ stochastic-gradient-langevin-boosting }}) paper for details.


**Default value**

False

**{{ cli__params-table__title__processing-units-type }}**

{{ calcer_type__cpu }}

### diffusion_temperature


#### Description


The diffusion temperature of the Stochastic Gradient Langevin Boosting mode.

Only non-negative values are supported.


**Default value**

10000

**{{ cli__params-table__title__processing-units-type }}**

{{ calcer_type__cpu }}

### allow_const_label


#### Description


Use it to train models with datasets that have equal label values for all objects.


**Default value**

{{ fit__allow-const-label }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### score_function


#### Description


The [score type](../../../concepts/algorithm-score-functions.md) used to select the next split during the tree construction.

Possible values:

- {{ scorefunction__Correlation }} (do not use this score type with the {{ growing_policy__Lossguide }} tree growing policy)
- {{ scorefunction__L2 }}
- {{ scorefunction__NewtonCorrelation }} (do not use this score type with the {{ growing_policy__Lossguide }} tree growing policy)
- {{ scorefunction__NewtonL2 }}


**Default value**

{{ scorefunction__default }}

**{{ cli__params-table__title__processing-units-type }}**


The supported score functions vary depending on the processing unit type:

- {{ calcer_type__gpu }} — All score types

- {{ calcer_type__cpu }} — {{ scorefunction__Correlation }}, {{ scorefunction__L2 }}



### cat_features


#### Description


A vector of categorical features indices.

The indices are zero-based and can differ from the ones given in the [columns description](../../../concepts/input-data_column-descfile.md) file.


**Default value**

{{ fit__r__cat_features }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### monotone_constraints


#### Description


{% include [reusage-cli__monotone-constraints__desc__div](cli__monotone-constraints__desc__div.md) %}


{% include [reusage-monotone-constraints__formats__intro](monotone-constraints__formats__intro.md) %}


- {% include [reusage-set-individual-constraints__div](set-individual-constraints__div.md) %}

    In
    {% cut "this example" %}

    ```
    monotone_constraints = "(1,0,-1)"
    ```

    {% endcut %}

    an increasing constraint is set on the first feature and a decreasing one on the third. Constraints are disabled for all other features.

- {% include [python-python__set-constraints-individually-for-each-required-feature](python__set-constraints-individually-for-each-required-feature.md) %}



**Default value**

None

**{{ cli__params-table__title__processing-units-type }}**

{{ calcer_type__cpu }}

### feature_weights


#### Description


{% include [reusage-cli__feature-weight__desc__intro](cli__feature-weight__desc__intro.md) %}


- {% include [reusage-cli__feature-weight__desc__weight-for-each-feature](cli__feature-weight__desc__weight-for-each-feature.md) %}

    In this
    {% cut "example" %}

    ```
    feature_weights = "(0.1,1,3)"
    ```

    {% endcut %}

    the multiplication weight is set to 0.1, 1 and 3 for the first, second and third features respectively. The multiplication weight for all other features is set to 1.

- {% include [reusage-cli__feature-weight__formats__individually-for-required-features](cli__feature-weight__formats__individually-for-required-features.md) %}

    {% cut "These examples" %}

    ```
    feature_weights = "2:1.1,4:0.1"
    ```

    ```
    feature_weights = "Feature2:1.1,Feature4:0.1"
    ```

    {% endcut %}

    are identical, given that the name of the feature indexed 2 is <q>Feature2</q> and the name of the feature indexed 4 is <q>Feature4</q>.


**Default value**

1 for all features

**{{ cli__params-table__title__processing-units-type }}**

{{ calcer_type__cpu }}

### first_feature_use_penalties


#### Description


{% include [reusage-cli__first-feature-use-penalties__intro](cli__first-feature-use-penalties__intro.md) %}


{% include [reusage-r-r__penalties](../reusage-r/r__penalties.md) %}



**Default value**

0 for all features

**{{ cli__params-table__title__processing-units-type }}**

{{ calcer_type__cpu }}

## fixed_binary_splits

#### Description


A list of indices of binary features to put at the top of each tree; ignored if `grow_policy` is `Symmetric`.

**Default value**

 None

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__gpu }}

### penalties_coefficient


#### Description


A single-value common coefficient to multiply all penalties.

Non-negative values are supported.


**Default value**

1

**{{ cli__params-table__title__processing-units-type }}**

{{ calcer_type__cpu }}

### per_object_feature_penalties


#### Description


{% include [reusage-per-object-feature-penalties__intro](per-object-feature-penalties__intro.md) %}


{% include [reusage-r-r__penalties](../reusage-r/r__penalties.md) %}



**Default value**

0 for all objects

**{{ cli__params-table__title__processing-units-type }}**

{{ calcer_type__cpu }}

### model_shrink_rate


#### Description


The constant used to calculate the coefficient for multiplying the model on each iteration.
The actual model shrinkage coefficient calculated at each iteration depends on the value of the `--model-shrink-mode`for the Command-line version  parameter. The resulting value of the coefficient should be always in the range (0, 1].

**Default value**


The default value depends on the values of the following parameters:
- `--model-shrink-mode`for the Command-line version
- `--monotone-constraints`for the Command-line version

**{{ cli__params-table__title__processing-units-type }}**

{{ calcer_type__cpu }}

### model_shrink_mode

#### Description

Determines how the actual model shrinkage coefficient is calculated at each iteration.

Possible values:
- {{ model_shrink_mode__Constant }}:

    $1 - model\_shrink\_rate \cdot learning\_rate{,}$
    - $model\_shrink\_rate$ is the value of the `--model-shrink-rate`for the Command-line version parameter.
    - $learning\_rate$ is the value of the `--learning-rate` for the Command-line version parameter

- {{ model_shrink_mode__Decreasing }}:
    $1 - \frac{model\_shrink\_rate}{i}{,}$
    - $model\_shrink\_rate$ is the value of the `--model-shrink-rate`for the Command-line version parameter.
    - $i$ is the identifier of the iteration.

**Default value**

{{ model_shrink_mode__Constant }}

**{{ cli__params-table__title__processing-units-type }}**

{{ calcer_type__cpu }}

## Overfitting detection settings

### early_stopping_rounds

#### Description

Sets the overfitting detector type to {{ fit--od-type-iter }} and stops the training after the specified number of iterations since the iteration with the optimal metric value.

**Default value**

FALSE

**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}


### od_type


#### Description


The type of the overfitting detector to use.

Possible values:
- {{ fit--od-type-inctodec }}
- {{ fit--od-type-iter }}


**Default value**

{{ fit--od-type-inctodec }}

**{{ cli__params-table__title__processing-units-type }}**


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

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### od_wait


#### Description

The number of iterations to continue the training after the iteration with the optimal metric value.
The purpose of this parameter differs depending on the selected overfitting detector type:
- {{ fit--od-type-inctodec }} — Ignore the overfitting detector when the threshold is reached and continue learning for the specified number of iterations after the iteration with the optimal metric value.
- {{ fit--od-type-iter }} — Consider the model overfitted and stop training after the specified number of iterations since the iteration with the optimal metric value.


**Default value**

{{ fit--od-wait }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



## Quantization settings

### target_border


#### Description


If set, defines the border for converting target values to 0 and 1.

Depending on the specified value:

- $target\_value \le border\_value$ the target is converted to 0
- $target\_value > border\_value$ the target is converted to 1


**Default value**

None

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### border_count


#### Description


The number of splits for numerical features. Allowed values are integers from 1 to 65535 inclusively.


**Default value**


{% include [reusage-default-values-border_count](../reusage-default-values/border_count.md) %}



**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### feature_border_type


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

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### per_float_feature_quantization


#### Description


{% include [python-python__per-float-feature-quantization__desc-without-examples](python__per-float-feature-quantization__desc-without-examples.md) %}


Examples:

- ```r
    per_float_feature_quantization = '0:border_count=1024')
    ```

    In this example, the feature indexed 0 has 1024 borders.

- ```r
    per_float_feature_quantization = c('0:border_count=1024', '1:border_count=1024'
    ```

    In this example, features indexed 0 and 1 have 1024 borders.



**Default value**

None

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



## Multiclassification settings

### classes_count


#### Description


{% include [reusage-classes-count__main-desc](classes-count__main-desc.md) %}


{% include [reusage-classes-count__possible-values](classes-count__possible-values.md) %}


If this parameter is specified the labels for all classes in the input dataset should be smaller than the given value


**Default value**


{{ fit--classes-count }}


**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



## Performance settings

### thread_count


#### Description


{% include [reusage-thread-count-short-desc](thread-count-short-desc.md) %}


{% include [reusage-thread_count__cpu_cores__optimizes-the-speed-of-execution](thread_count__cpu_cores__optimizes-the-speed-of-execution.md) %}



**Default value**

{{ fit__thread_count__wrappers }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



## Processing units settings

### task_type


#### Description


The processing unit type to use for training.

Possible values:
- CPU
- GPU


**Default value**

{{ fit__python-r__calcer_type }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### devices


#### Description


IDs of the GPU devices to use for training (indices are zero-based).

Format

- `<unit ID>` for one device (for example, `3`)
- `<unit ID1>:<unit ID2>:..:<unit IDN>` for multiple devices (for example, `devices='0:1:3'`)
- `<unit ID1>-<unit IDN>` for a range of devices (for example, `devices='0-3'`)


**Default value**

-1 (all GPU devices are used if the corresponding processing unit type is selected)

**{{ cli__params-table__title__processing-units-type }}**


{% include [reusage-python-gpu](../reusage-python/gpu.md) %}



## Output settings

### logging_level


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

{{ fit--verbose }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### metric_period


#### Description


{% include [reusage-cli__metric-period__desc__start](cli__metric-period__desc__start.md) %}



**Default value**

{{ fit__metric-period }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### verbose


#### Description


{% include [reusage-cli__train__verbose__short-desc](cli__train__verbose__short-desc.md) %}


{% note alert %}

Do not use this parameter with the `--logging-level` parameter.

{% endnote %}



**Default value**

{{ train_verbose_fr-of-iterations-to-output__default }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### train_dir


#### Description


The directory for storing the files generated during training.


**Default value**

{{ fit--train_dir }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### model_size_reg


#### Description


The model size regularization coefficient. The larger the value, the smaller the model size. Refer to the [Model size regularization coefficient](../../../references/model-size-reg.md) section for details.

Possible values are in the range $[0; \inf)$.

This regularization is needed only for models with categorical features (other models are small). Models with categorical features might weight tens of gigabytes or more if categorical features have a lot of values. If the value of the regularizer differs from zero, then the usage of categorical features or feature combinations with a lot of values has a penalty, so less of them are used in the resulting model.

Note that the resulting quality of the model can be affected. Set the value to 0 to turn off the model size optimization option.


**Default value**

{{ fit__model_size_reg }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### allow_writing_files


#### Description


Allow to write analytical and snapshot files during training.

If set to <q>False</q>, the [snapshot](../../../features/snapshots.md) and [data visualization](../../../features/visualization.md) tools are unavailable.


**Default value**

TRUE

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### save_snapshot


#### Description


{% include [python-save-snapshot__python-desc__desc__div](save-snapshot__python-desc__desc__div.md) %}



**Default value**

{{ fit--save_snapshot }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### snapshot_file


#### Description


{% include [python-python__snapshot-file__desc__div](python__snapshot-file__desc__div.md) %}



**Default value**

{{ fit--snapshot-file }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### snapshot_interval


#### Description


{% include [python-python__snapshot_interval__desc__div](python__snapshot_interval__desc__div.md) %}



**Default value**

{{ fit__snapshot-interval__default }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



## CTR settings

### simple_ctr


#### Description


{% include [reusage-cli__simple-ctr__intro](cli__simple-ctr__intro.md) %}


{% include [ctr-params-ctr__desc__format](ctr__desc__format.md) %}


```
c(CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N])
```

{% include [reusage-cli__simple-ctr__components](cli__simple-ctr__components.md) %}


{% include [reusage-cli__simple-ctr__examples__p](cli__simple-ctr__examples__p.md) %}



**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### combinations_ctr


#### Description


{% include [reusage-cli__combination-ctr__intro](cli__combination-ctr__intro.md) %}


{% include [ctr-params-ctr__desc__format](ctr__desc__format.md) %}


```
c(CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N])
```

{% include [reusage-cli__combination-ctr__components](cli__combination-ctr__components.md) %}



**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### ctr_target_border_count


#### Description


{% include [reusage-cli__ctr-target-border-count__short-desc](cli__ctr-target-border-count__short-desc.md) %}


{% include [python-python__ctr_borde_count__prelist](python__ctr_borde_count__prelist.md) %}


- `simple_ctr`
- `combinations_ctr`


**Default value**

{{ parameters__ctr-target-border-count__default }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### counter_calc_method


#### Description


The method for calculating the Counter CTR type.

Possible values:
- {{ counter-calculation-method--static }} — Objects from the validation dataset are not considered at all
- {{ counter-calculation-method--full }} — All objects from both learn and validation datasets are considered


**Default value**

{{ fit--counter-calc-method }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### max_ctr_complexity


#### Description


The maximum number of features that can be combined.

Each resulting combination consists of one or more categorical features and can optionally contain binary features in the following form: <q>numeric feature > value</q>.


**Default value**


{% include [reusage-default-values-max_xtr_complexity](../reusage-default-values/max_xtr_complexity.md) %}



**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}



### ctr_leaf_count_limit


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


**{{ cli__params-table__title__processing-units-type }}**


{{ calcer_type__cpu }}



### store_all_simple_ctr


#### Description


Ignore categorical features, which are not used in feature combinations, when choosing candidates for exclusion.

There is no point in using this parameter without the `--ctr-leaf-count-limit`for the Command-line version  parameter.

**Default value**
False

{{ fit--store-all-simple-ctr }}

**{{ cli__params-table__title__processing-units-type }}**


{{ calcer_type__cpu }}



### final_ctr_computation_mode


#### Description


Final CTR computation mode.

Possible values:
- {{ cli__fit__final-ctr-computation-mode__possible-values__Default }} — Compute final CTRs for learn and validation datasets.
- {{ cli__fit__final-ctr-computation-mode__possible-values__Skip }} — Do not compute final CTRs for learn and validation datasets. In this case, the resulting model can not be applied. This mode decreases the size of the resulting model. It can be useful for research purposes when only the metric values have to be calculated.


**Default value**

{{ cli__fit__final-ctr-computation-mode__default }}

**{{ cli__params-table__title__processing-units-type }}**


{{ cpu-gpu }}
