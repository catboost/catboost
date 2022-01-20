# Objectives and metrics

This section contains basic information regarding the supported metrics for various machine learning problems.
- [Regression](loss-functions-regression.md)
- [Multiregression](loss-functions-multiregression.md)
- [Classification](loss-functions-classification.md)
- [Multiclassification](loss-functions-multiclassification.md)
- [Multilabel classification](loss-functions-multilabel-classification.md)
- [Ranking](loss-functions-ranking.md)

Refer to the [Variables used in formulas](loss-functions-variables-used.md) section for the description of commonly used variables in the listed metrics.

Metrics can be calculated during the training or separately from the training for a specified model. The calculated values are written to files and can be plotted by [visualization tools](../features/visualization.md) (both during and after the training) for further analysis.

## User-defined parameters {#user-defined-parameters}

Some metrics provide user-defined parameters. These parameters must be set together with the metric name when it is being specified.

The parameters for each metric are set in the following format:
```
<Metric>[:<parameter 1>=<value>;..;<parameter N>=<value>]
```

The supported parameters vary from one metric to another and are listed alongside the corresponding descriptions.

#### {{ dl__usage-examples }}

```
{{ error-function--Quantile }}:alpha=0.1
```

#### List of most important parameters

The following table contains the description of parameters that are used in several metrics. The default values vary from one metric to another and are listed alongside the corresponding descriptions.

{% include [table-for-reusage-loss-functions__desc__for-reusage-without-percs](../_includes/work_src/reusage-loss-functions/loss-functions__desc__for-reusage-without-percs.md) %}

## Enable, disable and configure metrics calculation {#enable-disable-configure-metrics}

The calculation of metrics can be resource-intensive. It creates a bottleneck in some cases, for example, if many metrics are calculated during the training or the computation is performed on GPU.

The training can be sped up by disabling the calculation of some metrics for the training dataset. Use the `hints=skip_train~true` parameter to disable the calculation of the specified metrics.

{% note info %}

{% include [loss-functions-calculations-of-some-metrics-disabled-by-default](../_includes/work_src/reusage-common-phrases/calculations-of-some-metrics-disabled-by-default.md) %}


{% cut "Metrics that are not calculated by default for the train dataset" %}

- {{ error-function__PFound }}
- {{ error-function__YetiRank }}
- {{ error-function__ndcg }}
- {{ error-function__YetiRankPairwise }}
- {{ error-function--AUC }}
- {{ error-function--NormalizedGini }}
- {{ error-function__FilteredDCG }}
- {{ error-function__dcg }}

{% endcut %}


{% endnote %}


{% cut "{{ dl__usage-examples }}" %}

Enable the calculation of the {{ error-function--AUC }} metric:
```
AUC:hints=skip_train~false
```

Disable the calculation of the {{ error-function--Logit }} metric:
```
Logloss:hints=skip_train~true
```

{% endcut %}


Another way to speed up the training is to set up the frequency of iterations to calculate the values of metrics. Use one of the following parameters:

**Command-line version parameters:** `--metric-period`

**Python parameters:** `metric_period`

**R parameters:** `metric_period`

For example, use the following parameter in Python or R to calculate metrics once per 50 iterations:
```
metric_period=50
```

