## Input file settings

### -f, --learn-set

#### Description

The path to the input file{% if audience == "internal" %} or table{% endif %} that contains the dataset description.

{% include [files-internal-files-internal__desc__full](../../../_includes/work_src/reusage-formats/files-internal__desc__full.md) %}

**{{ cli__params-table__title__default }}**

{{ fit__learn-set }}


**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### -t, --test-set

#### Description

A comma-separated list of input files that contain the validation dataset description (the format must be the same as used in the training dataset).

{% if audience == "internal" %}

{% include [files-internal-format-for-multimple-files](../../../yandex_specific/_includes/reusage-formats/format-for-multimple-files.md) %}

{% include [files-internal-files-internal__desc__possbile-values](../../../yandex_specific/_includes/reusage-formats/files-internal__desc__possbile-values.md) %}

{% endif %}

**{{ cli__params-table__title__default }}**

 Omitted. If this parameter is omitted, the validation dataset isn't used.


**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}


{% note alert %}

Only a single validation dataset can be input if the training is performed on GPU (`--task-type` is set to GPU)

{% endnote %}



### --cd, --column-description

#### Description

The path to the input file {% if audience == "internal" %}or table{% endif %} that contains the [columns description](../../../concepts/input-data_column-descfile.md).

{% if audience == "internal" %}

{% include [internal__cd-internal-cd-desc](../../../yandex_specific/_includes/reusage-formats/internal-cd-desc.md) %}

{% endif %}

**{{ cli__params-table__title__default }}:**


If omitted, it is assumed that the first column in the file with the dataset description defines the label value, and the other columns are the values of numerical features.


**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --learn-pairs

#### Description

The path to the input file that contains the [pairs description](../../../concepts/input-data_pairs-description.md) for the training dataset.

This information is used for calculation and optimization of [](../../../concepts/loss-functions-ranking.md).


**{{ cli__params-table__title__default }}**

 Omitted.

{% include [loss-functions-pairwisemetrics_require_pairs_data](../reusage-common-phrases/pairwisemetrics_require_pairs_data.md) %}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --test-pairs

#### Description

The path to the input file that contains the [pairs description](../../../concepts/input-data_pairs-description.md) for the validation dataset.

{% include [reusage-learn_pairs__where_is_used](learn_pairs__where_is_used.md) %}


**{{ cli__params-table__title__default }}**

Omitted.

{% include [loss-functions-pairwisemetrics_require_pairs_data](../reusage-common-phrases/pairwisemetrics_require_pairs_data.md) %}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --learn-group-weights

#### Description

The path to the input file that contains the weights of groups. Refer to the [Group weights](../../../concepts/input-data_group-weights.md) section for format details.

{% include [reusage-input-data-group_weights__input-dataset-requirement](../reusage-input-data/group_weights__input-dataset-requirement.md) %}


{% include [reusage-input-data-group_weights__precedence-over-datasetdesc](../reusage-input-data/group_weights__precedence-over-datasetdesc.md) %}

**{{ cli__params-table__title__default }}:**

Omitted (group weights are either read from the dataset description or set to 1 for all groups if absent in the input dataset)

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --test-group-weights

#### Description

The path to the input file that contains the weights of groups for the validation dataset. Refer to the [Group weights](../../../concepts/input-data_group-weights.md) section for format details.

{% include [reusage-input-data-group_weights__input-dataset-requirement](../reusage-input-data/group_weights__input-dataset-requirement.md) %}


{% include [reusage-input-data-group_weights__precedence-over-datasetdesc](../reusage-input-data/group_weights__precedence-over-datasetdesc.md) %}


**{{ cli__params-table__title__default }}:**

Omitted (group weights are either read from the dataset description or set to 1 for all groups if absent in the input dataset)

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --learn-baseline

#### Description

The path to the input file that contains baseline values for the training dataset. Refer to the [Baseline ](../../../concepts/input-data_baseline.md) section for format details.


**{{ cli__params-table__title__default }}**

Omitted

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --test-baseline

#### Description

The path to the input file that contains baseline values for the validation dataset. Refer to the [Baseline ](../../../concepts/input-data_baseline.md) section for format details.


**{{ cli__params-table__title__default }}**

Omitted

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --delimiter

#### Description

The delimiter character used to separate the data in the dataset description input file.

Only single char delimiters are supported. If the specified value contains more than one character, only the first one is used.

{% include [libsvm-note-restriction-delimiter-separated-format](../reusage-formats/note-restriction-delimiter-separated-format.md) %}


**{{ cli__params-table__title__default }}**

{{ fit__delimiter }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --has-header

#### Description

Read the column names from the first line of the dataset description file if this parameter is set.

{% include [libsvm-note-restriction-delimiter-separated-format](../reusage-formats/note-restriction-delimiter-separated-format.md) %}

**{{ cli__params-table__title__default }}:**

False (the first line is supposed to have the same data as the rest of them)

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --params-file

#### Description

The path to the input JSON file that contains the training parameters, for example:

```
{
"learning_rate": 0.1,
"bootstrap_type": "No"
}
```

Names of training parameters are the same as for the [{{ python-package }}](../../../references/training-parameters/index.md) or the [{{ r-package }}](../../../concepts/r-reference_catboost-train.md#parameters-list).

If a parameter is specified in both the JSON file and the corresponding command-line parameter, the command-line value is used.

**{{ cli__params-table__title__default }}**

Omitted

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --nan-mode

#### Description

The method for [processing missing values](../../../concepts/algorithm-missing-values-processing.md) in the input dataset.

Possible values:

{% include [reusage-missing-values-mv-processing-methods](../reusage-missing-values/mv-processing-methods.md) %}

Using the  {{ fit__nan_mode__min }} or {{ fit__nan_mode__max }} value of this parameter guarantees that a split between missing values and other values is considered when selecting a new split in the tree.

{% note info %}

The method for processing missing values can be set individually for each feature in the [Custom quantization borders and missing value modes](../../../concepts/input-data_custom-borders.md) input file. Such values override the ones specified in this parameter.

{% endnote %}

**{{ cli__params-table__title__default }}**

{{ fit--nan_mode }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



## Training parameters

### --loss-function

#### Description

The [metric](../../../concepts/loss-functions.md) to use in training. The specified value also determines the machine learning problem to solve. Some metrics support optional parameters (see the [Objectives and metrics](../../../concepts/loss-functions.md) section for details on each metric).

Format:
```
<Metric>[:<parameter 1>=<value>;..;<parameter N>=<value>]
```

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

For example, use the following construction to calculate the value of {{ error-function--Quantile }} with the coefficient $\alpha = 0.1$:
```
{{ error-function--Quantile }}:alpha=0.1
```


**{{ cli__params-table__title__default }}**

{{ fit--loss_function }}


**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --custom-metric

#### Description

[Metric](../../../concepts/loss-functions.md) values to output during training. These functions are not optimized and are displayed for informational purposes only. Some metrics support optional parameters (see the [Objectives and metrics](../../../concepts/loss-functions.md) section for details on each metric).

Format:
```
<Metric 1>[:<parameter 1>=<value>;..;<parameter N>=<value>],<Metric 2>[:<parameter 1>=<value>;..;<parameter N>=<value>],..,<Metric N>[:<parameter 1>=<value>;..;<parameter N>=<value>]
```

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

Values of all custom metrics for learn and validation datasets are saved to the [Metric](../../../concepts/output-data_loss-function.md) output files (`learn_error.tsv` and `test_error.tsv` respectively). The directory for these files is specified in the `--train-dir` (`train_dir`) parameter.


**{{ cli__params-table__title__default }}**

None (do not output additional metric values)

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --eval-metric

#### Description

The metric used for overfitting detection (if enabled) and best model selection (if enabled). Some metrics support optional parameters (see the [Objectives and metrics](../../../concepts/loss-functions.md) section for details on each metric).

Format:
```
<Metric>[:<parameter 1>=<value>;..;<parameter N>=<value>]
```

[Supported metrics](../../../references/eval-metric__supported-metrics.md)

Examples:
```
R2
```

```
Quantile:alpha=0.3
```


**{{ cli__params-table__title__default }}**

{{ fit--eval-metric }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### -i, --iterations

#### Description

The maximum number of trees that can be built when solving machine learning problems.

When using other parameters that limit the number of iterations, the final number of trees may be less than the number specified in this parameter.


**{{ cli__params-table__title__default }}**

{{ fit--iterations }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### -w, --learning-rate

#### Description

The learning rate.

Used for reducing the gradient step.


**{{ cli__params-table__title__default }}:**

The default value is defined automatically for {{ error-function--Logit }}, {{ error-function--MultiClass }} & {{ error-function--RMSE }} loss functions depending on the number of iterations if none of these parameters (`leaf_estimation_iterations`, `--leaf-estimation-method`, `l2_leaf_reg`) is set.
In this case, the selected learning rate is printed to stdout and saved in the model.

In other cases, the default value is 0.03.

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}


### -r, --random-seed

#### Description

The random seed used for training.

**{{ cli__params-table__title__default }}**

{{ fit--random_seed }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --l2-leaf-reg, l2-leaf-regularizer

#### Description

Coefficient at the L2 regularization term of the cost function.
Any positive value is allowed.


**{{ cli__params-table__title__default }}**

{{ fit--l2-leaf-reg }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --bootstrap-type

#### Description

[Bootstrap type](../../../concepts/algorithm-main-stages_bootstrap-options.md). Defines the method for sampling the weights of objects.

Supported methods:

- {{ fit__bootstrap-type__Bayesian }}
- {{ fit__bootstrap-type__Bernoulli }}
- {{ fit__bootstrap-type__MVS }}
- {{ fit__bootstrap-type__Poisson }} (supported for GPU only)
- {{ fit__bootstrap-type__No }}


**{{ cli__params-table__title__default }}:**

- {{ error-function__QueryCrossEntropy }}, {{ error-function__YetiRankPairwise }}, {{ error-function__PairLogitPairwise }}: {{ fit__bootstrap-type__Bernoulli }} with the subsample parameter set to 0.5
- {{ error-function--MultiClass }} and {{ error-function--MultiClassOneVsAll }}: Bayesian
- Other modes:
  - GPU: {{ fit__bootstrap-type__Bayesian }}
  - CPU: {{ fit__bootstrap-type__MVS }} with the `subsample` parameter set to 0.8

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}


### --bagging-temperature

#### Description

Defines the settings of the Bayesian bootstrap. It is used by default in classification and regression modes.

Use the Bayesian bootstrap to assign random weights to objects.

The weights are sampled from exponential distribution if the value of this parameter is set to <q>1</q>. All weights are equal to 1 if the value of this parameter is set to <q>0</q>.

Possible values are in the range $[0; \inf)$. The higher the value the more aggressive the bagging is.

This parameter can be used if the selected bootstrap type is {{ fit__bootstrap-type__Bayesian }}.


**{{ cli__params-table__title__default }}**

{{ fit__bagging-temperature }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}


### --subsample

#### Description

Sample rate for bagging.

This parameter can be used if one of the following bootstrap types is selected:
- {{ fit__bootstrap-type__Poisson }}
- {{ fit__bootstrap-type__Bernoulli }}
- {{ fit__bootstrap-type__MVS }}


**{{ cli__params-table__title__default }}:**


{% include [reusage-default-values-subsample__default](../reusage-default-values/subsample__default.md) %}


**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --sampling-frequency

#### Description

Frequency to sample weights and objects when building trees.

Supported values:

- {{ fit__sampling-frequency__PerTree }} — Before constructing each new tree
- {{ fit__sampling-frequency__PerTreeLevel }} — Before choosing each new split of a tree

**{{ cli__params-table__title__default }}**

{{ fit__sampling-frequency }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}


### --sampling-unit

#### Description

The sampling scheme.

Possible values:
- {{ python__ESamplingUnit__type__Object }} — The weight $w_{i}$ of the i-th object $o_{i}$ is used for sampling the corresponding object.
- {{ python__ESamplingUnit__type__Group }} — The weight $w_{j}$ of the group $g_{j}$ is used for sampling each object $o_{i_{j}}$ from the group $g_{j}$.


**{{ cli__params-table__title__default }}**

{{ python__ESamplingUnit__type__default }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}


### --mvs-reg

#### Description

Affects the weight of the denominator and can be used for balancing between the importance and Bernoulli sampling (setting it to 0 implies importance sampling and to $\infty$ - {{ fit__bootstrap-type__Bernoulli }}).

{% note info %}

This parameter is supported only for the {{ fit__bootstrap-type__MVS }} sampling method (the `--bootstrap-type` must be set to {{ fit__bootstrap-type__MVS }}).

{% endnote %}


**{{ cli__params-table__title__default }}:**

The value is {{ fit__mvs_head_fraction }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}

### --random-strength

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


**{{ cli__params-table__title__default }}**

{{ fit--random-strength }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}

### --use-best-model

#### Description

If this parameter is set, the number of trees that are saved in the resulting model is defined as follows:
1. Build the number of trees defined by the training parameters.
1. Use the validation dataset to identify the iteration with the optimal value of the metric specified in  `--eval-metric` (`--eval-metric`).

No trees are saved after this iteration.

This option requires a validation dataset to be provided.

**{{ cli__params-table__title__default }}:**

True if a validation set is input (the `-t` or the `--test-set` parameter is defined) and at least one of the label values of objects in this set differs from the others. False otherwise.

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}


### --best-model-min-trees

#### Description

The minimal number of trees that the best model should have. If set, the output model contains at least the given number of trees even if the best model is located within these trees.

Should be used with the `--use-best-model` parameter.

**{{ cli__params-table__title__default }}**

{{ fit__best-model-min-trees }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}


### -n, --depth

Depth of the trees.

The range of supported values depends on the processing unit type and the type of the selected loss function:
- CPU — Any integer up to  {{ fit--maxtree }}.

- GPU — Any integer up to {{ fit__maxtree__pairwise }}pairwise modes ({{ error-function__YetiRank }}, {{ error-function__PairLogitPairwise }} and {{ error-function__QueryCrossEntropy }}) and up to   {{ fit--maxtree }} for all other loss functions.

**{{ cli__params-table__title__default }}**

{{ fit--depth }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --grow-policy

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


**{{ cli__params-table__title__default }}**

{{ growing_policy__default }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --min-data-in-leaf

#### Description

The minimum number of training samples in a leaf. {{ product }} does not search for new splits in leaves with samples count less than the specified value.
Can be used only with the {{ growing_policy__Lossguide }} and {{ growing_policy__Depthwise }} growing policies.


**{{ cli__params-table__title__default }}**

{{ min-samples-in-leaf__default }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --max-leaves

#### Description

The maximum number of leaves in the resulting tree. Can be used only with the {{ growing_policy__Lossguide }} growing policy.

{% note info %}

It is not recommended to use values greater than 64, since it can significantly slow down the training process.

{% endnote %}


**{{ cli__params-table__title__default }}**

{{ max-leaves-count__default }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### -I, --ignore-features

#### Description

Feature indices or names to exclude from the training. It is assumed that all passed values are feature names if at least one of the passed values can not be converted to a number or a range of numbers. Otherwise, it is assumed that all passed values are feature indices.

Specifics:

- Non-negative indices that do not match any features are successfully ignored. For example, if five features are defined for the objects in the dataset and this parameter is set to <q>42</q>, the corresponding non-existing feature is successfully ignored.

- The identifier corresponds to the feature's index. Feature indices used in train and feature importance are numbered from 0 to `featureCount – 1`. If a file is used as [input data](../../../concepts/input-data.md) then any non-feature column types are ignored when calculating these indices. For example, each row in the input file contains data in the following order: `cat feature<\t>label value<\t>num feature`. So for the row `rock<\t>0<\t>42`, the identifier for the <q>rock</q> feature is 0, and for the <q>42</q> feature it's 1.

- The addition of a non-existing feature name raises an error.

Supported operators:

- <q>:</q> — Value separator.
- <q>-</q> — Range of values (the left and right edges are included).

For example, if training should exclude features with the identifiers 1, 2, 7, 42, 43, 44, 45, use the following construction:
```
1:2:7:42-45
```

**{{ cli__params-table__title__default }}**

{{ fit--ignored_features }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --one-hot-max-size

#### Description

Use one-hot encoding for all categorical features with a number of different values less than or equal to the given parameter value. Ctrs are not calculated for such features.

See [details](../../../features/categorical-features.md).


**{{ cli__params-table__title__default }}:**


{% include [reusage-default-values-one-hot-max-size-default](../reusage-default-values/one-hot-max-size-default.md) %}


**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --has-time

#### Description

Use the order of objects in the input data (do not perform random permutations during the [Transforming categorical features to numerical features](../../../concepts/algorithm-main-stages_cat-to-numberic.md) and [Choosing the tree structure](../../../concepts/algorithm-main-stages_choose-tree-structure.md) stages).

The {{ cd-file__col-type__Timestamp }} column type is used to determine the order of objects if specified in the [input data](../../../concepts/input-data.md).


**{{ cli__params-table__title__default }}**

{{ fit--has_time }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --rsm

#### Description

Random subspace method. The percentage of features to use at each split selection, when features are selected over again at random.

The value must be in the range (0;1].


**{{ cli__params-table__title__default }}**

{{ fit--rsm }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }} and {{ calcer_type__gpu }} for pairwise ranking



### --fold-permutation-block

#### Description

Objects in the dataset are grouped in blocks before the random permutations. This parameter defines the size of the blocks. The smaller is the value, the slower is the training. Large values may result in quality degradation.


**{{ cli__params-table__title__default }}**

{{ fit--fold-permutation-block }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}


### --leaf-estimation-method

#### Description

The method used to calculate the values in leaves.

Possible values:
- {{ fit__leaf_estimation_method__Newton }}
- {{ fit__leaf_estimation_method__Gradient }}
- {{ fit__leaf_estimation_method__Exact }}


**{{ cli__params-table__title__default }}:**

Depends on the mode and the selected loss function:
- Regression with {{ error-function--Quantile }} or {{ error-function--MAE }} loss functions — One {{ fit__leaf_estimation_method__Exact }} iteration.
- Regression with any loss function but {{ error-function--Quantile }} or {{ error-function--MAE }} – One {{ fit__leaf_estimation_method__Gradient }} iteration.
- Classification mode – Ten {{ fit__leaf_estimation_method__Newton }} iterations.
- Multiclassification mode – One {{ fit__leaf_estimation_method__Newton }} iteration.


**{{ cli__params-table__title__processing-units-type }}**

- The {{ fit__leaf_estimation_method__Exact }} method is available only on {{ calcer_type__cpu }}
- All other methods are available on both {{ calcer_type__cpu }} and {{ calcer_type__gpu }}


### --leaf-estimation-iterations

#### Description

{{ product }} might calculate leaf values using several gradient or newton steps instead of a single one.

This parameter regulates how many steps are done in every tree when calculating leaf values.


**{{ cli__params-table__title__default }}**

{{ fit--gradient_iterations }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}


### --leaf-estimation-backtracking

#### Description

When the value of the `--leaf-estimation-iterations`for the Command-line version parameter is greater than 1, {{ product }} makes several gradient or newton steps when calculating the resulting leaf values of a tree.

The behaviour differs depending on the value of this parameter:

- No — Every next step is a regular gradient or newton step: the gradient step is calculated and added to the leaf.
- Any other value —Backtracking is used.
    In this case, before adding a step, a condition is checked. If the condition is not met, then the step size is reduced (divided by 2), otherwise the step is added to the leaf.

When `--leaf-estimation-iterations`for the Command-line version is set to `n`, the leaf estimation iterations are calculated as follows: each iteration is either an addition of the next step to the leaf value, or it's a scaling of the leaf value. Scaling counts as a separate iteration. Thus, it is possible that instead of having `n` gradient steps, the algorithm makes a single gradient step that is reduced `n` times, which means that it is divided by $2\cdot n$ times.

Possible values:
- {{ cli__leaf-estimation-backtracking__No }} — Do not use backtracking. Supported on {{ calcer_type__cpu }} and {{ calcer_type__gpu }}.
- {{ cli__leaf-estimation-backtracking__AnyImprovement }} — Reduce the descent step up to the point when the loss function value is smaller than it was on the previous step. The trial reduction factors are 2, 4, 8, and so on. Supported on {{ calcer_type__cpu }} and {{ calcer_type__gpu }}.
- {{ cli__leaf-estimation-backtracking__Armijo }} — Reduce the descent step until the Armijo condition is met. Supported only on {{ calcer_type__gpu }}.


**{{ cli__params-table__title__default }}**

{{ cli__leaf-estimation-backtracking__default }}

**{{ cli__params-table__title__processing-units-type }}**

 Depends on the selected value

### --name

#### Description

The experiment name to display in [visualization tools](../../../features/visualization.md).

**{{ cli__params-table__title__default }}**

{{ fit--name }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --prediction-type

#### Description

A comma-separated list of prediction types to output during training for the validation dataset. This information is output if a validation dataset is provided.

Supported prediction types:
- {{ prediction-type--Probability }}
- {{ prediction-type--Class }}
- {{ prediction-type--RawFormulaVal }}
- {{ prediction-type--Exponent }}
- {{ prediction-type--LogProbability }}


**{{ cli__params-table__title__default }}**

{{ prediction-type--RawFormulaVal }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}



### --fold-len-multiplier

#### Description

Coefficient for changing the length of folds.

The value must be greater than 1. The best validation result is achieved with minimum values.

With values close to 1 (for example, $1+\epsilon$), each iteration takes a quadratic amount of memory and time for the number of objects in the iteration. Thus, low values are possible only when there is a small number of objects.


**{{ cli__params-table__title__default }}**

{{ fit--fold-len-multiplier }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --approx-on-full-history

#### Description

The principles for calculating the approximated values.

Possible values:
- <q>False</q> — Use only а fraction of the fold for calculating the approximated values. The size of the fraction is calculated as follows: $\frac{1}{{X}}$, where `X` is the specified coefficient for changing the length of folds. This mode is faster and in rare cases slightly less accurate
- <q>True</q> — Use all the preceding rows in the fold for calculating the approximated values. This mode is slower and in rare cases slightly more accurate.


**{{ cli__params-table__title__default }}**

{{ fit--approx_on_full_history }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}



### --class-weights

#### Description

Class weights. The values are used as multipliers for the object weights. This parameter can be used for solving binary classification and multiclassification problems.

For imbalanced datasets with binary classification, the weight multiplier can be set to 1 for class 0 and to $\frac{sum\_negative}{sum\_positive}$ for class 1.

{% note info %}

- The quantity of class weights must match the quantity of class names specified in the `--class-names` parameter and the number of classes specified in the `--classes-count` parameter.

- For imbalanced datasets with binary classification the weight multiplier can be set to 1 for class 0 and to $\left(\frac{sum\_negative}{sum\_positive}\right)$ for class 1.

{% endnote %}

Format:
```
<value for class 1>,..,<values for class N>
```

For example:
```
0.85,1.2,1
```

{% note info %}

Do not use this parameter with `--auto-class-weights`.

{% endnote %}

**{{ cli__params-table__title__default }}**

{{ fit--class-weights }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --auto-class-weights

#### Description

Automatically calculate class weights based either on the total weight or the total number of objects in each class. The values are used as multipliers for the object weights.

Supported values:

- {{ autoclass__weights__default }}
- {{ autoclass__weights__balanced }}:

    $CW_k=\displaystyle\frac{max_{c=1}^K(\sum_{t_{i}=c}{w_i})}{\sum_{t_{i}=k}{w_{i}}}$

- {{ autoclass__weights__SqrtBalanced }}:

    $CW_k=\sqrt{\displaystyle\frac{max_{c=1}^K(\sum_{t_i=c}{w_i})}{\sum_{t_i=k}{w_i}}}$


{% note alert %}

Do not use this parameter with `--class-weights`.

{% endnote %}


**{{ cli__params-table__title__default }}**

{{ autoclass__weights__default }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}


### --boosting-type

#### Description

Boosting scheme.

Possible values:
- {{ fit__boosting-type__ordered }} — Usually provides better quality on small datasets, but it may be slower than the Plain scheme.
- {{ fit__boosting-type__plain }} — The classic gradient boosting scheme.


**{{ cli__params-table__title__default }}:**


{% cut "{{ calcer_type__cpu }}" %}

{{ fit__boosting-type__plain }}

{% endcut %}

{% cut "{{ calcer_type__gpu }}" %}

- Any number of objects, {{ error-function--MultiClass }} or {{ error-function--MultiClassOneVsAll }} mode: {{ fit__boosting-type__plain }}
- More than 50 thousand objects, any mode: {{ fit__boosting-type__plain }}
- Less than or equal to 50 thousand objects, any mode but {{ error-function--MultiClass }} or {{ error-function--MultiClassOneVsAll }}: {{ fit__boosting-type__ordered }}

{% endcut %}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}


Only the {{ fit__boosting-type__plain }} mode is supported for the {{ error-function--MultiClass }} loss on GPU


### --boost-from-average

#### Description

Initialize approximate values by best constant value for the specified loss function. Sets the value of bias to the initial best constant value.

Available for the following loss functions:
- {{ error-function--RMSE }}
- {{ error-function--Logit }}
- {{ error-function--CrossEntropy }}
- {{ error-function--Quantile }}
- {{ error-function--MAE }}
- {{ error-function--MAPE }}


**{{ cli__params-table__title__default }}:**


{% include [reusage-default-values-boost-from-average](../reusage-default-values/boost-from-average.md) %}



**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --langevin

#### Description

Enables the Stochastic Gradient Langevin Boosting mode.

Refer to the [SGLB: Stochastic Gradient Langevin Boosting]({{ stochastic-gradient-langevin-boosting }}) paper for details.


**{{ cli__params-table__title__default }}**

False

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}

### --diffusion-temperature

#### Description


The diffusion temperature of the Stochastic Gradient Langevin Boosting mode.

Only non-negative values are supported.


**{{ cli__params-table__title__default }}**

10000

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}

### --allow-const-label

#### Description


Use it to train models with datasets that have equal label values for all objects.


**{{ cli__params-table__title__default }}**

{{ fit__allow-const-label }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --score-function

#### Description


The [score type](../../../concepts/algorithm-score-functions.md) used to select the next split during the tree construction.

Possible values:

- {{ scorefunction__Correlation }} (do not use this score type with the {{ growing_policy__Lossguide }} tree growing policy)
- {{ scorefunction__L2 }}
- {{ scorefunction__NewtonCorrelation }} (do not use this score type with the {{ growing_policy__Lossguide }} tree growing policy)
- {{ scorefunction__NewtonL2 }}


**{{ cli__params-table__title__default }}**

{{ scorefunction__default }}

**{{ cli__params-table__title__processing-units-type }}**

The supported score functions vary depending on the processing unit type:

- {{ calcer_type__gpu }} — All score types

- {{ calcer_type__cpu }} — {{ scorefunction__Correlation }}, {{ scorefunction__L2 }}



### --monotone-constraints

#### Description


Impose monotonic constraints on numerical features.

Possible values:

- <q>1</q> — Increasing constraint on the feature. The algorithm forces the model to be a non-decreasing function of this features.

- <q>-1</q> — Decreasing constraint on the feature. The algorithm forces the model to be a non-increasing function of this features.

- <q>0</q> — constraints are disabled.


Supported formats for setting the value of this parameter (all feature indices are zero-based):

- Set constraints individually for each feature as a string (the number of features is n).

    {% cut "Format" %}

    ```
    "(<constraint_0>, <constraint_2>, .., <constraint_n-1>)"
    ```

    {% endcut %}

    The values should be passed as a parenthesized string of comma-separated values. Zero constraints for features at the end of the list may be dropped.

    In this example `--monotone-constraints "(1,0,-1)"` an increasing constraint is set on the first feature and a decreasing one on the third. Constraints are disabled for all other features.

- Set constraints individually for each explicitly specified feature as a string (the number of features is n).

    {% cut "Format" %}

    ```
    "<feature index or name>:<constraint>, .., <feature index or name>:<constraint>"
    ```

    {% endcut %}

    {% cut "These examples" %}

    ```
    --monotone-constraints "2:1,4:-1"
    ```

    ```
    --monotone-constraints "Feature2:1,Feature4:-1"
    ```

    {% endcut %}

    are identical, given that the name of the feature indexed 2 is <q>Feature2</q> and the name of the feature indexed 4 is <q>Feature4</q>.

- Set constraints individually for each required feature as an array or a dictionary (the number of features is n).

    {% cut "Format" %}

    ```
    [<constraint_0>, <constraint_2>, .., <constraint_n-1>]
    ```

    ```
    {"<feature index or name>":<constraint>, .., "<feature index or name>":<constraint>}
    ```

    {% endcut %}

    This format can be used if parameters are passed in a JSON file (see the `--params-file` parameter).

    {% cut "Examples" %}

    ```json
    {
    "monotone_constraints": {"Feature2":1,"Feature4":-1}
    }
    ```

    ```json
    {
    "monotone_constraints": {"2":1, "4":-1}
    }
    ```

    ```json
    {
    "monotone_constraints": [0,0,1,0,-1]
    }
    ```

    {% endcut %}



**{{ cli__params-table__title__default }}**

Ommited

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}

### --feature-weights

#### Description


Per-feature multiplication weights used when choosing the best split. The score of each candidate is multiplied by the weights of features from the current split.

Non-negative float values are supported for each weight.

Supported formats for setting the value of this parameter:

- Set the multiplication weight for each feature as a string (the number of features is n).

    {% cut "Format" %}

    ```
    "(<feature-weight_0>,<feature-weight_2>,..,<feature-weight_n-1>)"
    ```

    {% note info %}

    Spaces between values are not allowed.

    {% endnote %}

    {% endcut %}

    Values should be passed as a parenthesized string of comma-separated values. Multiplication weights equal to 1 at the end of the list may be dropped.

    In this
    {% cut "example" %}

    ```
    --feature-weights "(0.1,1,3)"
    ```

    {% endcut %}

    the multiplication weight is set to 0.1, 1 and 3 for the first, second and third features respectively. The multiplication weight for all other features is set to 1.

- Set the multiplication weight individually for each explicitly specified feature as a string (the number of features is n).

    {% cut "Format" %}

    ```
    "<feature index or name>:<weight>, .., <feature index or name>:<weight>"
    ```

    {% note info %}

    Spaces between values are not allowed.

    {% endnote %}

    {% endcut %}

    {% cut "These examples" %}

    ```
    --feature-weights "2:0.1,4:1.3"
    ```

    ```
    --feature-weights "Feature2:0.1,Feature4:1.3"
    ```

    {% endcut %}

    are identical, given that the name of the feature indexed 2 is <q>Feature2</q> and the name of the feature indexed 4 is <q>Feature4</q>.

- Set the multiplication weight individually for each required feature as an array or a dictionary (the number of features is n).

    {% cut "Format" %}

    ```
    [<feature-weight_0>, <feature-weight_2>, .., <feature-weight_n-1>]
    ```

    ```
    {"<feature index or name>":<weight>, .., "<feature index or name>":<weight>}
    ```

    {% endcut %}

    This format can be used if parameters are passed in a JSON file (see the `--params-file` parameter).

    {% cut "Examples" %}

    ```json
    {
    "feature_weights": {"Feature2":0.1, "Feature4":1.3}
    }
    ```

    ```json
    {
    "feature_weights": {"2":0.1, "4":1.3}
    }
    ```

    ```json
    {
    "feature_weights": [0.1,0.2,1.3,0.4,2.3]
    }
    ```

    {% endcut %}



**{{ cli__params-table__title__default }}**

1 for all features

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}

### --first-feature-use-penalties

#### Description


Per-feature penalties for the first occurrence of the feature in the model. The given value is subtracted from the score if the current candidate is the first one to include the feature in the model.

Refer to the [Per-object and per-feature penalties](../../../concepts/algorithm-score-functions.md) section for details on applying different score penalties.

Non-negative float values are supported for each penalty.

{% include [penalties-format-cli__penalties__format](reusage_cli.md#cli__penalties__format) %}

**{{ cli__params-table__title__default }}**

0 for all features

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}

### --penalties-coefficient


#### Description

A single-value common coefficient to multiply all penalties.

Non-negative values are supported.


**{{ cli__params-table__title__default }}**

1

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}

### --per-object-feature-penalties

#### Description


Per-object penalties for the first use of the feature for the object. The given value is multiplied by the number of objects that are divided by the current split and use the feature for the first time.

Refer to the [Per-object and per-feature penalties](../../../concepts/algorithm-score-functions.md) section for details on applying different score penalties.

Non-negative float values are supported for each penalty.

{% include [penalties-format-cli__penalties__format](reusage_cli.md#cli__penalties__format)) %}

**{{ cli__params-table__title__default }}**

0 for all objects

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}

### --model-shrink-rate

#### Description

The constant used to calculate the coefficient for multiplying the model on each iteration.
The actual model shrinkage coefficient calculated at each iteration depends on the value of the
`--model-shrink-mode`for the Command-line version parameter. The resulting value of the coefficient should be always in the range (0, 1].

**{{ cli__params-table__title__default }}:**


The default value depends on the values of the following parameters: `--model-shrink-mode`, `--monotone-constraints`for the Command-line version.


**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}

### --model-shrink-mode

#### Description

Determines how the actual model shrinkage coefficient is calculated at each iteration.

Possible values:
- {{ model_shrink_mode__Constant }}:

    $1 - model\_shrink\_rate \cdot learning\_rate {,}$

    - $model\_shrink\_rate$ is the value of the `--model-shrink-rate`for the Command-line version parameter.
    - $learning\_rate$ is the value of the `--learning-rate`for the Command-line version parameter.

- {{ model_shrink_mode__Decreasing }}:
    $1 - \frac{model\_shrink\_rate}{i} {,}$

    - $model\_shrink\_rate$ is the value of the `--model-shrink-rate`for the Command-line version parameter.
    - $i$ is the identifier of the iteration.

**{{ cli__params-table__title__default }}**

{{ model_shrink_mode__Constant }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}

## Text processing parameters

### --tokenizers

#### Description

Tokenizers used to preprocess {{ data-type__text }} type feature columns before creating the dictionary.

Format:

```
TokenizerId[:option_name=option_value]
```

- `TokenizerId` — The unique name of the tokenizer.
- `option_name` — One of the [supported tokenizer options](../../../references/tokenizer_options.md).

{% note info %}

This parameter works with `--dictionaries` and `--feature-calcers` parameters.

For example, if a single tokenizer, three dictionaries and two feature calcers are given, a total of 6 new groups of features are created for each original text feature ($1 \cdot 3 \cdot 2 = 6$).

{% cut "Usage example" %}

```
--tokenizers "Space:delimiter= :separator_type=ByDelimiter,Sense:separator_type=BySense"
```

{% endcut %}

{% endnote %}



**{{ cli__params-table__title__default }}**

–

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__gpu }}



### --dictionaries

#### Description

Dictionaries used to preprocess {{ data-type__text }} type feature columns.

Format:

```
DictionaryId[:option_name=option_value]
```

- `DictionaryId` — The unique name of dictionary.
- `option_name` — One of the [supported dictionary options](../../../references/dictionaries_options.md).

{% note info %}

This parameter works with `--tokenizers` and `--feature-calcers` parameters.

{% include [reusage-tokenizer-dictionaries-feature-calcers__note_div](tokenizer-dictionaries-feature-calcers__note_div.md) %}

{% cut "Usage example" %}

```
--dictionaries "Unigram:gram_count=1:max_dictionary_size=50000,Bigram:gram_count=2:max_dictionary_size=50000"
```

{% endcut %}

{% endnote %}



**{{ cli__params-table__title__default }}**

–

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__gpu }}



### --feature-calcers

#### Description

Feature calcers used to calculate new features based on preprocessed {{ data-type__text }} type feature columns.

Format:

```
FeatureCalcerName[:option_name=option_value]
```

- `FeatureCalcerName` — The required [feature calcer](../../../references/text-processing__feature_calcers.md).

- `option_name` — Additional options for feature calcers. Refer to the [list of supported calcers](../../../references/text-processing__feature_calcers.md) for details on options available for each of them.


{% note info %}

This parameter works with `--tokenizers` and `--dictionaries` parameters.

{% include [reusage-tokenizer-dictionaries-feature-calcers__note_div](tokenizer-dictionaries-feature-calcers__note_div.md) %}

{% cut "Usage example" %}

```
--feature-calcers BoW:top_tokens_count=1000,NaiveBayes
```

{% endcut %}

{% endnote %}


**{{ cli__params-table__title__default }}**

–

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__gpu }}



### --text-processing

#### Description

A JSON specification of tokenizers, dictionaries and feature calcers, which determine how text features are converted into a list of float features.

[Example](../../../references/text-processing__specification-example.md)

Refer to the description of the following parameters for details on supported values:

- `--tokenizers`
- `--dictionaries`
- `--feature-calcers`

{% note alert %}

Do not use this parameter with the following ones:
- `--tokenizers`
- `--dictionaries`
- `--feature-calcers`

{% endnote %}

**{{ cli__params-table__title__default }}**

[Default value](../../../references/text-processing__test-processing__default-value.md)

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__gpu }}



## Overfitting detection settings

### --od-type

#### Description

The type of the overfitting detector to use.

Possible values:
- {{ fit--od-type-inctodec }}
- {{ fit--od-type-iter }}


**{{ cli__params-table__title__default }}**

{{ fit--od-type-inctodec }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --od-pval

#### Description

The threshold for the {{ fit--od-type-inctodec }} [overfitting detector](../../../concepts/overfitting-detector.md) type. The training is stopped when the specified value is reached. Requires that a validation dataset was input.

For best results, it is recommended to set a value in the range $[10^{–10}; 10^{-2}]$.

The larger the value, the earlier overfitting is detected.

{% note alert %}

Do not use this parameter with the {{ fit--od-type-iter }} overfitting detector type.

{% endnote %}

**{{ cli__params-table__title__default }}**

{{ fit--auto_stop_pval }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --od-wait

#### Description

The number of iterations to continue the training after the iteration with the optimal metric value.
The purpose of this parameter differs depending on the selected overfitting detector type:
- {{ fit--od-type-inctodec }} — Ignore the overfitting detector when the threshold is reached and continue learning for the specified number of iterations after the iteration with the optimal metric value.
- {{ fit--od-type-iter }} — Consider the model overfitted and stop training after the specified number of iterations since the iteration with the optimal metric value.


**{{ cli__params-table__title__default }}**

{{ fit--od-wait }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



## Quantization settings

### --target-border

#### Description

If set, defines the border for converting target values to 0 and 1.

Depending on the specified value:

- $target\_value \le border\_value$ the target is converted to 0
- $target\_value > border\_value$ the target is converted to 1


**{{ cli__params-table__title__default }}:**

The value is not set

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### -x, --border-count

The number of splits for numerical features. Allowed values are integers from 1 to 65535 inclusively.


**{{ cli__params-table__title__default }}:**


{% include [reusage-default-values-border_count](../reusage-default-values/border_count.md) %}


**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --feature-border-type

The [quantization mode](../../../concepts/quantization.md) for numerical features.

Possible values:
- Median
- Uniform
- UniformAndQuantiles
- MaxLogSum
- MinEntropy
- GreedyLogSum


**{{ cli__params-table__title__default }}**

{{ fit--feature_border_type }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --output-borders-file

Save quantization borders for the current dataset to a file.

Refer to the [file format description](../../../concepts/output-data_custom-borders.md).


**{{ cli__params-table__title__default }}**

{{ cli__output-borders-file__default }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --input-borders-file


Load [Custom quantization borders and missing value modes](../../../concepts/input-data_custom-borders.md) from a file (do not generate them).

Borders are automatically generated before training if this parameter is not set.


**{{ cli__params-table__title__default }}**

{{ cli__input-borders-file__default }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --per-float-feature-quantization




A semicolon separated list of quantization descriptions.

Format:

```
FeatureId[:border_count=BorderCount][:nan_mode=BorderType][:border_type=border_selection_method]
```

Examples:

- ```
    --per-float-feature-quantization 0:border_count=1024
    ```

    In this example, the feature indexed 0 has 1024 borders.

- ```
    --per-float-feature-quantization 0:border_count=1024;1:border_count=1024
    ```

    In this example, features indexed 0 and 1 have 1024 borders.



**{{ cli__params-table__title__default }}**

Omitted

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



## Multiclassification settings

### --classes-count




The upper limit for the numeric class label. Defines the number of classes for multiclassification.

Only non-negative integers can be specified. The given integer should be greater than any of the label values.

If this parameter is specified and the `--class-names` is not the labels for all classes in the input dataset should be smaller than the given value.


**{{ cli__params-table__title__default }}:**


- `{{ fit--classes-count }}` if the `--class-names` parameter is not specified
- the quantity of classes names if the `--class-names` parameter is specified


**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --class-names

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


**{{ cli__params-table__title__default }}**

{{ fit--class-names }}


**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



## Performance settings

### -T


`--thread-count`



The number of threads to use during the training.

- **For CPU**

    Optimizes the speed of execution. This parameter doesn't affect results.

- **For GPU**
    The given value is used for reading the data from the hard drive and does not affect the training.

    During the training one main thread and one thread for each GPU are used.


**{{ cli__params-table__title__default }}**

{{ fit--thread_count }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --used-ram-limit

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


**{{ cli__params-table__title__default }}**

{{ fit__used-ram-limit }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}



### --gpu-ram-part


How much of the GPU RAM to use for training.


**{{ cli__params-table__title__default }}**

{{ fit__gpu__gpu-ram-part }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__gpu }}



### --pinned-memory-size

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


**{{ cli__params-table__title__default }}**

{{ fit__gpu__pinned-memory-size }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__gpu }}



### --gpu-cat-features-storage




The method for storing the categorical features' values.

Possible values:
- {{ fit__gpu__gpu_cat_features_storage__value__CpuPinnedMemory }}
- {{ fit__gpu__gpu_cat_features_storage__value__GpuRam }}

{% note info %}

Use the {{ fit__gpu__gpu_cat_features_storage__value__CpuPinnedMemory }} value if feature combinations are used and the available GPU RAM is not sufficient.

{% endnote %}

**{{ cli__params-table__title__default }}**

{{ fit__gpu__use-cpu-ram-for-catfeatures }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__gpu }}



### --data-partition


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


**{{ cli__params-table__title__default }}**

{{ fit__gpu__data-partition }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__gpu }}



## Processing unit settings

### --task-type




The processing unit type to use for training.

Possible values:
- CPU
- GPU


**{{ cli__params-table__title__default }}**

{{ fit__python-r__calcer_type }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --devices




IDs of the GPU devices to use for training (indices are zero-based).

Format

- `<unit ID>` for one device (for example, `3`)
- `<unit ID1>:<unit ID2>:..:<unit IDN>` for multiple devices (for example, `devices='0:1:3'`)
- `<unit ID1>-<unit IDN>` for a range of devices (for example, `devices='0-3'`)


**{{ cli__params-table__title__default }}**

-1 (use all devices)

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__gpu }}



## Output settings

### --logging-level




The logging level to output to stdout.

Possible values:
- Silent — Do not output any logging information to stdout.

- Verbose — Output the following data to stdout:

    - optimized metric
    - elapsed time of training
    - remaining time of training

- Info — Output additional information and the number of trees.

- Debug — Output debugging information.


**{{ cli__params-table__title__default }}**

{{ fit--verbose }}


**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --metric-period




The frequency of iterations to calculate the values of [objectives and metrics](../../../concepts/loss-functions.md). The value should be a positive integer.

The usage of this parameter speeds up the training.

{% note info %}

It is recommended to increase the value of this parameter to maintain training speed if a GPU processing unit type is used.

{% endnote %}



**{{ cli__params-table__title__default }}**

{{ fit__metric-period }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --verbose


The frequency of iterations to print the information to stdout. The value of this parameter should be divisible by the value of the frequency of iterations to calculate the values of [objectives and metrics](../../../concepts/loss-functions.md).

{% note alert %}

Do not use this parameter with the `--logging-level` parameter.

{% endnote %}



**{{ cli__params-table__title__default }}**

{{ train_verbose_fr-of-iterations-to-output__default }}


**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --train-dir

The directory for storing the files generated during training.


**{{ cli__params-table__title__default }}**

{{ cli__fit__train_dir }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --model-size-reg

The model size regularization coefficient. The larger the value, the smaller the model size. Refer to the [Model size regularization coefficient](../../../references/model-size-reg.md) section for details.

Possible values are in the range $[0; \inf)$.

This regularization is needed only for models with categorical features (other models are small). Models with categorical features might weight tens of gigabytes or more if categorical features have a lot of values. If the value of the regularizer differs from zero, then the usage of categorical features or feature combinations with a lot of values has a penalty, so less of them are used in the resulting model.

Note that the resulting quality of the model can be affected. Set the value to 0 to turn off the model size optimization option.


**{{ cli__params-table__title__default }}**

{{ fit__model_size_reg }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --snapshot-file




Settings for [recovering training after an interruption](../../../features/snapshots.md).

Depending on whether the specified file exists in the file system:
- Missing — Write information about training progress to the specified file.
- Exists — Load data from the specified file and continue training from where it left off.


**{{ cli__params-table__title__default }}**

{{ fit--snapshot-file }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --snapshot-interval




{% include [python-python__snapshot_interval__desc__div](python__snapshot_interval__desc__div.md) %}

**{{ cli__params-table__title__default }}**

{{ fit__snapshot-interval__default }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### -m


`--model-file`



The name of the resulting  files with the model description.

Used for solving other machine learning problems (for instance, applying a model) or defining the names of models in different output formats.

Corresponding file extensions are added to the given value if several output formats are defined in the `--model-format` parameter.


**{{ cli__params-table__title__default }}**

{{ fit--ModelFileName }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --model-format




A comma-separated list of output model formats.

Possible values:

- {{ fit__model-format_CatboostBinary }}.
- {{ fit__model-format_applecoreml }}(only datasets without categorical features are currently supported).
- {{ fit__model-format_json }} (multiclassification models are not currently supported). Refer to the [CatBoost JSON model tutorial]({{ catboost-repo__json-tutorial }}) for format details.
- {{ fit__model-format_python }} (multiclassification models are not currently supported).See the [Python](../../../concepts/python-reference_apply_catboost_model.md) section for details on applying the resulting model.
- {{ fit__model-format_cpp }} (multiclassification models are not currently supported). See the [C++](../../../concepts/c-plus-plus-api_applycatboostmodel.md) section for details on applying the resulting model.
- {{ fitpython__model-format_onnx }} — ONNX-ML format (only datasets without categorical features are currently supported). Refer to [https://onnx.ai/](https://onnx.ai/) for details. See the [ONNX](../../../concepts/apply-onnx-ml.md) section for details on applying the resulting model.
- {{ fitpython__model-format_pmml }} — [PMML version {{ pmml-supported-version }}]({{ pmml-v4point3 }}) format. Categorical features must be interpreted as one-hot encoded during the training if present in the training dataset. This can be accomplished by setting the `--one-hot-max-size`/`one_hot_max_size` parameter to a value that is greater than the maximum number of unique categorical feature values among all categorical features in the dataset. See the [PMML](../../../concepts/apply-pmml.md) section for details on applying the resulting model.



**{{ cli__params-table__title__default }}**

{{ fit__model-format }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --fstr-file




The name of the resulting file that contains [regular feature importance](../../../concepts/output-data_feature-analysis_feature-importance.md#per-feature-importance) data (see [Feature importance](../../../concepts/fstr.md)).


**{{ cli__params-table__title__default }}**

{{ fit--FstrFileName }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --fstr-type




The feature strength type of the [regular feature importance](../../../concepts/output-data_feature-analysis_feature-importance.md#per-feature-importance). The selected type is output if the `--fstr-file` option is specified.

{% include [reusage-formats-regular-feature-importance-type](../reusage-formats/regular-feature-importance-type.md) %}

**{{ cli__params-table__title__default }}**

{{ cli__fstr-type__default }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --fstr-internal-file

The name of the resulting file that contains [internal feature importance](../../../concepts/output-data_feature-analysis_feature-importance.md#internal-feature-importance) data (see [Feature importance](../../../concepts/fstr.md)).


**{{ cli__params-table__title__default }}**

{{ fit--IFstrFileName }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --eval-file

The name of the resulting file that contains the model values on the validation datasets.

The format of the [output file](../../../concepts/output-data_model-value-output.md) depends on the problem being solved and the number of input validation datasets.


**{{ cli__params-table__title__default }}:**

Save the file to the current directory. The name of the file differs depending on the machine learning problem being solved and the selected metric. The file extensions is `eval`.

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --output-columns




A comma-separated list of columns names to output when forming the [results of applying the model](../../../concepts/output-data_model-value-output.md) (including the ones obtained for the validation dataset when training).

Prediction and feature values can be output for each object of the input dataset. Additionally, some [column types](../../../concepts/input-data_column-descfile.md) can be output if specified in the input data.

{% cut "Supported prediction types" %}

- {{ prediction-type--Probability }}
- {{ prediction-type--Class }}
- {{ prediction-type--RawFormulaVal }}
- {{ prediction-type--Exponent }}
- {{ prediction-type--LogProbability }}

{% endcut %}

{% cut "Supported column types" %}

- `{{ cd-file__col-type__label }}`
- `{{ cd-file__col-type__Baseline }}`
- `{{ cd-file__col-type__Weight }}`
- `{{ cd-file__col-type__SampleId }}` (`{{ cd-file__col-type__DocId }}`)
- `{{ cd-file__col-type__GroupId }}` (`{{ cd-file__col-type__QueryId }}`)
- `{{ cd-file__col-type__QueryId }}`
- `{{ cd-file__col-type__SubgroupId }}`
- `{{ cd-file__col-type__Timestamp }}`
- `{{ cd-file__col-type__GroupWeight }}`

{% endcut %}

The output columns can be set in any order. Format:
```
<prediction type 1>,[<prediction type 2> .. <prediction type N>][columns to output],[#<feature index 1>[:<name to output (user-defined)>] .. #<feature index N>[:<column name to output>]]
```

#### Example

```
--output-columns Probability,#3,#4:Feature4,Label,SampleId
```

In this example, features with indices 3 and 4 are output. The header contains the index (<q>#3</q>) for the feature indexed 3 and the string <q>Feature4</q> for the feature indexed 4.

{% cut "A fragment of the output" %}

```
Probability	#3	Feature4	Label	SampleId
0.4984999565	1	50.7799987793	0	0
0.8543220144	1	48.6333312988	2	1
0.7358535042	1	52.5699996948	1	2
0.8788711681	1	48.1699981689	2	3
```

{% endcut %}

{% note info %}

At least one of the specified columns must contain prediction values. For example, the following value raises an error:
```
--output-columns SampleId
```

{% endnote %}


**{{ cli__params-table__title__default }}**

All columns that are supposed to be output according to the chosen parameters are output


**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --json-log




The name of the resulting file that contains [metric values and time information](../../../concepts/output-data_training-log.md).


**{{ cli__params-table__title__default }}**

{{ fit__json-log }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --detailed-profile



Generate a file that contains [profiler information](../../../concepts/output-data_profiler.md).

**{{ cli__params-table__title__default }}**

The file is not generated

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --profiler-log




The name of the resulting file that contains [profiler information](../../../concepts/output-data_profiler.md).


**{{ cli__params-table__title__default }}**

{{ fit__profiler-log }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --learn-err-log




The name of the resulting file that contains the [metric value](../../../concepts/output-data_loss-function.md) for the training dataset.


**{{ cli__params-table__title__default }}**

{{ fit--LearnErrorLog }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --test-err-log




The name of the resulting file that contains the [metric value](../../../concepts/output-data_loss-function.md) for the validation dataset.


**{{ cli__params-table__title__default }}**

{{ fit--TestErrorLog }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



## CTR settings

### --simple-ctr




Quantization settings for simple [categorical features](../../../concepts/algorithm-main-stages_cat-to-numberic.md). Use this parameter to specify the principles for defining the class of the object for regression tasks. By default, it is considered that an object belongs to the positive class if its' label value is greater than the median of all label values of the dataset.

{% include [ctr-params-ctr__desc__format](ctr__desc__format.md) %}


```
CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]
```

Components:
- `CtrType` — The method for transforming categorical features to numerical features.

    Supported methods for training on CPU:

    - Borders
    - Buckets
    - BinarizedTargetMeanValue
    - Counter

    Supported methods for training on GPU:

    - Borders
    - Buckets
    - FeatureFreq
    - FloatTargetMeanValue

- `{{ ctr-types__TargetBorderCount }}` — The number of borders for label value [quantization](../../../concepts/quantization.md). Only used for regression problems. Allowed values are integers from 1 to 255 inclusively. The default value is {{ fit--target_border_count }}.

    This option is available for training on CPU only.

- `TargetBorderType` — The [quantization](../../../concepts/quantization.md) type for the label value. Only used for regression problems.

    Possible values:

    - Median
    - Uniform
    - UniformAndQuantiles
    - MaxLogSum
    - MinEntropy
    - GreedyLogSum

    By default, {{ fit--target_border_type }}.

    This option is available for training on CPU only.

- `CtrBorderCount` — The number of splits for categorical features. Allowed values are integers from 1 to 255 inclusively.
- `CtrBorderType` — The quantization type for categorical features.

    Supported values for training on CPU:
    - Uniform

    Supported values for training on GPU:

    - Median
    - Uniform
    - UniformAndQuantiles
    - MaxLogSum
    - MinEntropy
    - GreedyLogSum

- `Prior` — Use the specified priors during training (several values can be specified).

    Possible formats:
    - One number — Adds the value to the numerator.
    - Two slash-delimited numbers (for GPU only) — Use this format to set a fraction. The number is added to the numerator and the second is added to the denominator.

{% cut "Examples" %}

- ```
    simple_ctr='Borders:{{ ctr-types__TargetBorderCount }}=2'
    ```

    Two new features with differing quantization settings are generated. The first one concludes that an object belongs to the positive class when the label value exceeds the first border. The second one concludes that an object belongs to the positive class when the label value exceeds the second border.

    For example, if the label takes three different values (0, 1, 2), the first border is 0.5 while the second one is 1.5.

- ```
    simple_ctr='Buckets:{{ ctr-types__TargetBorderCount }}=2'
    ```

    The number of features depends on the number of different labels. For example, three new features are generated if the label takes three different values (0, 1, 2). In this case, the first one concludes that an object belongs to the positive class when the value of the feature is equal to 0 or belongs to the bucket indexed 0. The second one concludes that an object belongs to the positive class when the value of the feature is equal to 1 or belongs to the bucket indexed 1, and so on.

{% endcut %}


**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --combinations-ctr




Quantization settings for combinations of [categorical features](../../../concepts/algorithm-main-stages_cat-to-numberic.md).

{% include [ctr-params-ctr__desc__format](ctr__desc__format.md) %}


```
CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]
```

Components:
- `CtrType` — The method for transforming categorical features to numerical features.

    Supported methods for training on CPU:

    - Borders
    - Buckets
    - BinarizedTargetMeanValue
    - Counter

    Supported methods for training on GPU:

    - Borders
    - Buckets
    - FeatureFreq
    - FloatTargetMeanValue

- `{{ ctr-types__TargetBorderCount }}` — The number of borders for label value [quantization](../../../concepts/quantization.md). Only used for regression problems. Allowed values are integers from 1 to 255 inclusively. The default value is {{ fit--target_border_count }}.

    This option is available for training on CPU only.

- `TargetBorderType` — The [quantization](../../../concepts/quantization.md) type for the label value. Only used for regression problems.

    Possible values:

    - Median
    - Uniform
    - UniformAndQuantiles
    - MaxLogSum
    - MinEntropy
    - GreedyLogSum

    By default, {{ fit--target_border_type }}.

    This option is available for training on CPU only.

- `CtrBorderCount` — The number of splits for categorical features. Allowed values are integers from 1 to 255 inclusively.
- {% include [ctr-params-ctr__desc__ctrbordertype_intro](ctr__desc__ctrbordertype_intro.md) %}

    {% include [ctr-params-ctr__desc__ctrbordertype__supported-cpu](ctr__desc__ctrbordertype__supported-cpu.md) %}

    Supported values for training on GPU:
    - Uniform
    - Median

- `Prior` — Use the specified priors during training (several values can be specified).

    Possible formats:
    - One number — Adds the value to the numerator.
    - Two slash-delimited numbers (for GPU only) — Use this format to set a fraction. The number is added to the numerator and the second is added to the denominator.


**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --per-feature-ctr




Per-feature quantization settings for [categorical features](../../../concepts/algorithm-main-stages_cat-to-numberic.md).

{% include [ctr-params-ctr__desc__format](ctr__desc__format.md) %}


```
FeatureId:CtrType:[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_1/denum_1]
```

Components:
- `FeatureId` — A zero-based feature identifier.


**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --ctr-target-border-count




The maximum number of borders to use in target quantization for categorical features that need it. Allowed values are integers from 1 to 255 inclusively.

The value of the `{{ ctr-types__TargetBorderCount }}` component overrides this parameter if it is specified for one of the following options:

- `--simple-ctr`
- `--combinations-ctr`
- `--per-feature-ctr`


**{{ cli__params-table__title__default }}**

{{ parameters__ctr-target-border-count__default }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --counter-calc-method




The method for calculating the Counter CTR type.

Possible values:
- {{ counter-calculation-method--static }} — Objects from the validation dataset are not considered at all
- {{ counter-calculation-method--full }} — All objects from both learn and validation datasets are considered


**{{ cli__params-table__title__default }}**

{{ fit--counter-calc-method }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --max-ctr-complexity




The maximum number of features that can be combined.

Each resulting combination consists of one or more categorical features and can optionally contain binary features in the following form: <q>numeric feature > value</q>.


**{{ cli__params-table__title__default }}:**


{% include [reusage-default-values-max_xtr_complexity](../reusage-default-values/max_xtr_complexity.md) %}



**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --ctr-leaf-count-limit




The maximum number of leaves with categorical features. If the quantity exceeds the specified value a part of leaves is discarded.

The leaves to be discarded are selected as follows:

1. The leaves are sorted by the frequency of the values.
1. The top `N` leaves are selected, where N is the value specified in the parameter.
1. All leaves starting from `N+1` are discarded.

This option reduces the resulting model size and the amount of memory required for training. Note that the resulting quality of the model can be affected.


**{{ cli__params-table__title__default }}**

{{ fit--ctr-leaf-count-limit }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}



### --store-all-simple-ctr




Ignore categorical features, which are not used in feature combinations, when choosing candidates for exclusion.

There is no point in using this parameter without the `--ctr-leaf-count-limit`for the Command-line version parameter.


**{{ cli__params-table__title__default }}**

{{ fit--store-all-simple-ctr }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ calcer_type__cpu }}



### --final-ctr-computation-mode




Final CTR computation mode.

Possible values:
- {{ cli__fit__final-ctr-computation-mode__possible-values__Default }} — Compute final CTRs for learn and validation datasets.
- {{ cli__fit__final-ctr-computation-mode__possible-values__Skip }} — Do not compute final CTRs for learn and validation datasets. In this case, the resulting model can not be applied. This mode decreases the size of the resulting model. It can be useful for research purposes when only the metric values have to be calculated.


**{{ cli__params-table__title__default }}**

{{ cli__fit__final-ctr-computation-mode__default }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ cpu-gpu }}



### --posterior-sampling




If this parameter is set several options are specified as follows and model parameters are checked to obtain uncertainty predictions with good theoretical properties.

Possible values:

- `Langevin`: true,
- `DiffusionTemperature`: objects in learn pool count,
- `ModelShrinkRate`: 1 / (2. * objects in learn pool count).

See the Uncertainty section [Uncertainty](../../../references/uncertainty.md)
