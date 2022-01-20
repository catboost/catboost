# Multiregression: objectives and metrics

- [Objectives and metrics](#objectives-and-metrics)
- [{{ title__loss-functions__text__optimization }}](#used-for-optimization)

## Objectives and metrics

### {{ error-function__MultiRMSE }} {#MultiRMSE}

$\sqrt{\displaystyle\frac{\sum\limits_{i=1}^{N}\sum\limits_{d=1}^{dim}(a_{i,d} - t_{i, d})^{2} w_{i}}{\sum\limits_{i=1}^{N} w_{i}}}$
$dim$ is the identifier of the dimension of the label.

{{ title__loss-functions__text__user-defined-params }}:

{% include [use-weights__desc__without__full](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__full.md) %}


_Default:_  {{ loss-functions__use_weights__default }}


## {{ title__loss-functions__text__optimization }}


| Name                                           | Optimization            |
-------------------------------------------------|-------------------------|
[{{ error-function--MultiRMSE  }}](#MultiRMSE)   |     +                   |
