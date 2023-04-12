# {{ error-function__PFound }}

{% include [loss-functions-ranking-quality-assessment](../_includes/work_src/reusage-common-phrases/ranking-quality-assessment.md) %}


- [{{ title__loss-functions__calculation-principles }}](#calculation)
- [{{ title__loss-functions__text__user-defined-params }}](#user-defined-parameters)

## {{ title__loss-functions__calculation-principles }} {#calculation}

{% include [loss-functions-possible-lable-values](../_includes/work_src/reusage-common-phrases/possible-lable-values.md) %}


{% include [loss-functions-function-calculation](../_includes/work_src/reusage-common-phrases/function-calculation.md) %}


1. {% include [loss-functions-ascending_a_i](../_includes/work_src/reusage-common-phrases/ascending_a_i.md) %}

1. The {{ error-function__PFound }} metric is calculated for each group ($group \in groups$). To do this, the label values ($t_{i}$) of the objects from the sorted list are multiplied by their weight ($P(i, group, decay)$) and summed up as follows:

    $PFound(group, top, decay) = \sum_{i = 0}^{top} P(i, group, decay) * t_{g(i, group)}{, where}$

    - $P(0, group, decay) = 1$
    - $P(i, group, decay) = P(i – 1, group, decay) * (1 - t_{g(i – 1, group)}) * decay$
    - $decay$ is a constant, $decay \in [0, 1]$
    - $g(i, group)$ is the global index of the $i$-th best object in the group . The $i$-th object is considered better than the $j$-th object if the following inequality is true: $a_{i} > a_{j}$

1. The aggregated value of the metric for all groups is calculated as follows:

    $PFound(top, decay) = \displaystyle\frac{1}{\sum\limits_{group\in groups} w_{group}}\sum_{group\in groups} w_{group} PFound(group, top, decay)$



## {{ title__loss-functions__text__user-defined-params }} {#user-defined-parameters}

### decay

#### Description

The probability of search continuation after reaching the current object.

#### top

The number of top samples in a group that are used to calculate the ranking metric. Top samples are either the samples with the largest approx values or the ones with the lowest target values if approx values are the same.

_Default:_ –1 (all label values are used)
