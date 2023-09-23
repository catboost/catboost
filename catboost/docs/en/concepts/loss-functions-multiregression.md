# Multiregression: objectives and metrics

- [Objectives and metrics](#objectives-and-metrics)
- [{{ title__loss-functions__text__optimization }}](#used-for-optimization)

## Objectives and metrics

### {{ error-function__MultiRMSE }} {#MultiRMSE}

$\sqrt{\displaystyle\frac{\sum\limits_{i=1}^{N}\sum\limits_{d=1}^{dim}(a_{i,d} - t_{i, d})^{2} w_{i}}{\sum\limits_{i=1}^{N} w_{i}}}$

$dim$ is the identifier of the dimension of the label.

**{{ optimization }}**

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function__MultiRMSEWithMissingValues }} {#MultiRMSEWithMissingValues}

$\sqrt{\sum_{d=1}^{dim} \frac{\sum_{i=1}^N Num(a_{i,d}, t_{i,d}, w_i)}{\sum_{i=1}^N Den(t_{i,d}, w_i)}}$

$Num(a, t, w) = \begin{cases} w(a - t)^2, \space if \space t \neq NaN\\ 0 \end{cases}$

$Den(t, w) = \begin{cases} w, \space if \space t \neq NaN\\ 0 \end{cases}$

$dim$ is the identifier of the dimension of the label.

**{{ optimization }}**

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

## {{ title__loss-functions__text__optimization }}

| Name                                                                            | Optimization            | GPU Support             |
----------------------------------------------------------------------------------|-------------------------|-------------------------|
[{{ error-function__MultiRMSE }}](#MultiRMSE)                                     |     +                   |     +                   |
[{{ error-function__MultiRMSEWithMissingValues }}](#MultiRMSEWithMissingValues)   |     +                   |     -                   |
