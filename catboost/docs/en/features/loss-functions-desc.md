# Implemented metrics

{{ product }} provides built-in metrics for various machine learning problems. These functions can be used for model optimization or reference purposes. See the [Objectives and metrics](../concepts/loss-functions.md) section for details on the calculation principles.

{% include [feature-importance-choose-the-required-implementation-for-more-details](../_includes/work_src/reusage-common-phrases/choose-the-required-implementation-for-more-details.md) %}

- [python](#python)
- [r-package](#r-package)
- [cli](#command-line-version)

## {{ python-package }}

{% include [feature-importance-classes-params-can-be-used-for-train](../_includes/work_src/reusage-common-phrases/classes-params-can-be-used-for-train.md) %}

### Parameters for trained model

Classes:
- [CatBoost](../concepts/python-reference_catboost.md)
- [CatBoostClassifier](../concepts/python-reference_catboostclassifier.md)
- [CatBoostRegressor](../concepts/python-reference_catboostregressor.md)

#### loss-function

{% include [reusage-loss-function-short-desc](../_includes/work_src/reusage/loss-function-short-desc.md) %}

{% include [reusage-loss-function-format](../_includes/work_src/reusage/loss-function-format.md) %}

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

A custom python object can also be set as the value of this parameter (see an [example](../concepts/python-usages-examples.md)).

{% include [reusage-loss-function--example](../_includes/work_src/reusage/loss-function--example.md) %}


#### custom_metric

{% include [reusage-custom-loss--basic](../_includes/work_src/reusage/custom-loss--basic.md) %}

{% include [reusage-loss-function-format](../_includes/work_src/reusage/loss-function-format.md) %}

[Supported metrics](../references/custom-metric__supported-metrics.md)

Examples:
- Calculate the value of {{ error-function--CrossEntropy }}

    ```
    {{ error-function--CrossEntropy }}
    ```

- Calculate the value of {{ error-function--Quantile }} with the coefficient $\alpha = 0.1$
    ```
    {{ error-function--Quantile }}:alpha=0.1
    ```


- Calculate the values of {{ error-function--Logit }} and {{ error-function--AUC }}
    ```python
    ['{{ error-function--Logit }}', '{{ error-function--AUC }}']
    ```

{% include [reusage-custom-loss--values-saved-to](../_includes/work_src/reusage/custom-loss--values-saved-to.md) %}


Use the [visualization tools](../features/visualization.md) to see a live chart with the dynamics of the specified metrics.

#### use-best-model

If this parameter is set, the number of trees that are saved in the resulting model is defined as follows:
1. Build the number of trees defined by the training parameters.
1. Use the validation dataset to identify the iteration with the optimal value of the metric specified in  `--eval-metric` (`--eval-metric`).

No trees are saved after this iteration.

This option requires a validation dataset to be provided.

#### eval-metric

{% include [reusage-eval-metric--basic](../_includes/work_src/reusage/eval-metric--basic.md) %}

{% include [reusage-eval-metric--format](../_includes/work_src/reusage/eval-metric--format.md) %}

[Supported metrics](../references/eval-metric__supported-metrics.md)

A user-defined function can also be set as the value (see an [example](../concepts/python-usages-examples.md)).

{% include [reusage-eval-metric--examples](../_includes/work_src/reusage/eval-metric--examples.md) %}

{% include [feature-importance-method-params-training-and-applying](../_includes/work_src/reusage-common-phrases/method-params-training-and-applying.md) %}

### Parameters for trained or applied model

The following parameters can be set for the corresponding methods and are used when the model is trained or applied.

Classes:
- [fit](../concepts/python-reference_catboost_fit.md) ([CatBoost](../concepts/python-reference_catboost.md))
- [fit](../concepts/python-reference_catboostclassifier_fit.md) ([CatBoostClassifier](../concepts/python-reference_catboostclassifier.md))
- [fit](../concepts/python-reference_catboostregressor_fit.md) ([CatBoostRegressor](../concepts/python-reference_catboostregressor.md))


#### use_best_model

If this parameter is set, the number of trees that are saved in the resulting model is defined as follows:
1. Build the number of trees defined by the training parameters.
1. Use the validation dataset to identify the iteration with the optimal value of the metric specified in  `--eval-metric` (`--eval-metric`).

No trees are saved after this iteration.

This option requires a validation dataset to be provided.

#### verbose

Output the measured evaluation metric to stderr.


#### plot

Plot the following information during training:
- the metric values;
- the custom loss values;
- the loss function change during feature selection;
- the time has passed since training started;
- the remaining time until the end of training.
This [option can be used](../features/visualization_jupyter-notebook.md) if training is performed in Jupyter notebook.


## {{ r-package }}

{% include [feature-importance-method-params-training-and-applying](../_includes/work_src/reusage-common-phrases/method-params-training-and-applying.md) %}

Method:  [catboost.train](../concepts/r-reference_catboost-train.md)

### loss_function

**{{ features__table__title__r__description }}**

{% include [reusage-loss-function-short-desc](../_includes/work_src/reusage/loss-function-short-desc.md) %}

{% include [reusage-loss-function-format](../_includes/work_src/reusage/loss-function-format.md) %}

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

{% include [reusage-loss-function--example](../_includes/work_src/reusage/loss-function--example.md) %}

### custom_loss

**{{ features__table__title__r__parameters }}**

{% include [reusage-custom-loss--basic](../_includes/work_src/reusage/custom-loss--basic.md) %}

Format:
```
<Metric>[:<parameter 1>=<value>;..;<parameter N>=<value>]
```

[Supported metrics](../references/custom-metric__supported-metrics.md)

Examples:
- Calculate the value of {{ error-function--CrossEntropy }}

    ```no-highlight
    c('CrossEntropy')
    ```

    Or simply:
    ```
    'CrossEntropy'
    ```

- Calculate the values of {{ error-function--Logit }} and {{ error-function--AUC }}

    ```
    c('{{ error-function--Logit }}', '{{ error-function--AUC }}')
    ```

- Calculate the value of {{ error-function--Quantile }} with the coefficient $\alpha = 0.1$
    ```no-highlight
    c('{{ error-function--Quantile }}alpha=0.1')
    ```

{% include [reusage-custom-loss--values-saved-to](../_includes/work_src/reusage/custom-loss--values-saved-to.md) %}

### use-best-model

If this parameter is set, the number of trees that are saved in the resulting model is defined as follows:
1. Build the number of trees defined by the training parameters.
1. Use the validation dataset to identify the iteration with the optimal value of the metric specified in  `--eval-metric` (`--eval-metric`).

No trees are saved after this iteration.

This option requires a validation dataset to be provided.

### eval-metric

**{{ features__table__title__r__parameters }}**

{% include [reusage-eval-metric--basic](../_includes/work_src/reusage/eval-metric--basic.md) %}

{% include [reusage-eval-metric--format](../_includes/work_src/reusage/eval-metric--format.md) %}

[Supported metrics](../references/eval-metric__supported-metrics.md)

```
Quantile:alpha=0.3
```

## {{ title__implementation__cli }}

{% include [feature-importance-command-keys-trained-or-applied](../_includes/work_src/reusage-common-phrases/command-keys-trained-or-applied.md) %}

Params for the [catboost fit](../references/training-parameters/index.md) command:

### --loss-function

The [metric](../concepts/loss-functions.md) to use in training. The specified value also determines the machine learning problem to solve. Some metrics support optional parameters (see the [Objectives and metrics](../concepts/loss-functions.md) section for details on each metric).

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

{% endcut %}

For example, use the following construction to calculate the value of {{ error-function--Quantile }} with the coefficient $\alpha = 0.1$:
```
{{ error-function--Quantile }}alpha=0.1
```

### --custom-metric

[Metric](../concepts/loss-functions.md) values to output during training. These functions are not optimized and are displayed for informational purposes only. Some metrics support optional parameters (see the [Objectives and metrics](../concepts/loss-functions.md) section for details on each metric).

Format:
```
<Metric 1>[:<parameter 1>=<value>;..;<parameter N>=<value>],<Metric 2>[:<parameter 1>=<value>;..;<parameter N>=<value>],..,<Metric N>[:<parameter 1>=<value>;..;<parameter N>=<value>]
```

[Supported metrics](../references/custom-metric__supported-metrics.md)

Examples:
- Calculate the value of {{ error-function--CrossEntropy }}

    ```no-highlight
    {{ error-function--CrossEntropy }}
    ```

- Calculate the value of {{ error-function--Quantile }} with the coefficient $\alpha = 0.1$
    ```
    {{ error-function--Quantile }}alpha=0.1
    ```

Values of all custom metrics for learn and validation datasets are saved to the [Metric](../concepts/output-data_loss-function.md) output files (`learn_error.tsv` and `test_error.tsv` respectively). The directory for these files is specified in the `--train-dir` (`train_dir`) parameter.

### --use-best-model

If this parameter is set, the number of trees that are saved in the resulting model is defined as follows:
1. Build the number of trees defined by the training parameters.
1. Use the validation dataset to identify the iteration with the optimal value of the metric specified in  `--eval-metric` (`--eval-metric`).

No trees are saved after this iteration.

This option requires a validation dataset to be provided.


### --eval-metric

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
