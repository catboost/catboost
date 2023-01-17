# Classification: objectives and metrics

- [Objectives and metrics](#objectives-and-metrics)
- [{{ title__loss-functions__text__optimization }}](#usage-information)

## Objectives and metrics

### {{ error-function--Logit }} {#Logit}

$\displaystyle\frac{ - \sum\limits_{i=1}^N w_{i}\left(c_i \log(p_{i}) + (1-c_{i}) \log(1 - p_{i})\right)}{\sum\limits_{i = 1}^{N} w_{i}}$

**{{ optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function--CrossEntropy }} {#CrossEntropy}

$\displaystyle\frac{- \sum\limits_{i=1}^N w_{i} \left(t_{i} \log(p_{i}) + (1 - t_{i}) \log(1 - p_{i})\right)}{\sum\limits_{i = 1}^{N} w_{i}}$

**{{ optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function--Precision }} {#Precision}

$\frac{TP}{TP + FP}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function--Recall }} {#Recall}

$\frac{TP}{TP+FN}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function--F }} {#F}

$(1 + \beta^2) \cdot  \frac{Precision * Recall}{(\beta^2 \cdot Precision) + Recall}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [beta_for_F__desc](../_includes/work_src/reusage-loss-functions/beta_for_F__desc.md) %}

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function--F1 }} {#F1}

$2 \frac{Precision * Recall}{Precision + Recall}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function__BalancedAccuracy }} {#BalancedAccuracy}

$\frac{1}{2} \left(\frac{TP}{P} + \frac{TN}{N} \right)$
{{ title__loss-functions__text__user-defined-params }}:

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function__BalancedErrorRate }} {#BalancedErrorRate}

$\frac{1}{2} \left( \displaystyle\frac{FP}{TN + FP} + \displaystyle\frac{FN}{FN + TP} \right)$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function--MCC }} {#MCC}

$\displaystyle\frac{TP * TN - FP * FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function--Accuracy }} {#Accuracy}

$\frac{TP + TN}{\sum\limits_{i=1}^{N} w_{i}}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function__CtrFactor }} {#CtrFactor}

$\displaystyle\frac{\left(\sum\limits_{i = 1}^{N} w_{i} t_{i}/N\right)}{\left(\sum\limits_{i = 1}^{N} w_{i} p_{i} /N\right)}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function--AUC }} {#AUC}

The calculation of this metric is disabled by default for the training dataset to speed up the training. Use the `hints=skip_train~false` parameter to enable the calculation.

#### {{ loss-functions__params__auc__type__Classic }}

$\displaystyle\frac{\sum I(a_{i}, a_{j}) \cdot w_{i} \cdot w_{j}} {\sum w_{i} \cdot w_{j}}$
The sum is calculated on all pairs of objects $(i,j)$ such that:
- $t_{i} = 0$
- $t_{j} = 1$
- $I(x, y) = \begin{cases} 0 { , } & x < y \\ 0.5 { , } & x=y \\ 1 { , } & x>y \end{cases}$

Refer to the [Wikipedia article]({{ wikipedia_under-the-curve }}) for details.

If the target type is not binary, then every object with target value $t$ and weight $w$ is replaced with two objects for the metric calculation:

- $o_{1}$ with weight $t \cdot w$ and target value 1
- $o_{2}$ with weight $(1 – t) \cdot w$ and target value 0.

Target values must be in the range [0; 1].

#### {{ loss-functions__params__auc__type__Ranking }}

$\displaystyle\frac{\sum I(a_{i}, a_{j}) \cdot w_{i} \cdot w_{j}} {\sum w_{i} * w_{j}}$

The sum is calculated on all pairs of objects $(i,j)$ such that:
- $t_{i} < t_{j}$
- $I(x, y) = \begin{cases} 0 { , } & x < y \\ 0.5 { , } & x=y \\ 1 { , } & x>y \end{cases}$

**{{ title__loss-functions__text__user-defined-params }}**

{% cut "{{ loss-functions__params__type }}" %}

{% include [reusage-loss-functions-type_of_auc__p](../_includes/work_src/reusage-loss-functions/type_of_auc__p.md) %}

_Default_: `Ranking`.
_Possible values:_ `{{ loss-functions__params__auc__type__Classic }}`, `{{ loss-functions__params__auc__type__Ranking }}`.
_Examples:_ `AUC:type=Classic`, `AUC:type=Ranking`.

{% endcut %}

{% cut "{{loss-functions__params__use_weights}}" %}

{% include [use-weights__desc__without__note](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__note.md) %}

_Default_: `False`.
_Examples_: `QueryAUC:type=Ranking;use_weights=False`.

{% endcut %}

{% include [query-auc](../_includes/concepts/query-auc.md) %}

{% include [pr-auc](../_includes/concepts/pr-auc.md) %}

### {{ error-function--NormalizedGini }} {#NormalizedGini}

See {{ error-function--AUC }}.

$2 AUC - 1$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function__Brierscore }} {#Brierscore}

$\displaystyle\frac{\sum\limits_{i=1}^{N} w_{i}\left(p_{i} - t_{i} \right)^{2}}{\sum\limits_{i=1}^{N} w_{i}}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function__HingeLoss }} {#HingeLoss}

$\displaystyle\frac{\sum\limits_{i=1}^{N} w_{i} max\{1 - t_{i} p_{i}, 0\}}{\sum\limits_{i=1}^{N} w_{i}} , t_{i} = \pm 1$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function__HammingLoss }} {#HammingLoss}

$\displaystyle\frac{\sum\limits_{i = 1}^{N} w_{i} [[p_{i} > 0.5] == t_{i}]]}{\sum\limits_{i=1}^{N} w_{i}}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function__ZeroOneLoss }} {#ZeroOneLoss}

$1 - Accuracy$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function__Kappa }} {#Kappa}

$1 - \displaystyle\frac{1 - Accuracy}{1 - RAccuracy}$

$RAccuracy = \displaystyle\frac{(TN + FP) (TN + FN) + (FN + TP) (FP + TP)}{(\sum\limits_{i=1}^{N} w_{i})^{2}}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function__WKappa }} {#WKappa}

See the formula on page 3 of the [A note on the linearly weighted kappa coefficient for ordinal scales](https://orbi.uliege.be/bitstream/2268/2262/1/STATMED-174.pdf) paper.

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function__LLP }} {#LLP}

The calculation consists of the following steps:

1. Define the sum of weights ($W$) and the mean target ($\bar{t}$):

    $W = \sum\limits_{i} w_{i}$

    $\bar{t} = \frac{1}{W} \sum\limits_{i} t_{i} w_{i}$

1. Denote log-likelihood of a constant prediction:

    $ll_0 = \sum\limits_{i} w_{i} (\bar{t} \cdot log(\bar{t}) + (1 - \bar{t}) \cdot log(1 - \bar{t}))$

1. Calculate {{ error-function__LLP }} ($llp$), which reflects how the likelihood ($ll$) differs from the constant prediction:

    $llp = \displaystyle\frac{ll(t, w) - ll_0}{\sum\limits_{i} t_{i} w_{i}}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}



## {{ title__loss-functions__text__optimization }} {#usage-information}


| Name                                                        | Optimization            | GPU Support             |
--------------------------------------------------------------|-------------------------|-------------------------|
[{{ error-function--Logit }}](#Logit)                         |     +                   |     +                   |
[{{ error-function--CrossEntropy }}](#CrossEntropy)           |     +                   |     +                   |
[{{ error-function--Precision }}](#Precision)                 |     -                   |     +                   |
[{{ error-function--Recall }}](#Recall)                       |     -                   |     +                   |
[{{ error-function--F }}](#F)                                 |     -                   |     -                   |
[{{ error-function--F1 }}](#F1)                               |     -                   |     +                   |
[{{ error-function__BalancedAccuracy }}](#BalancedAccuracy)   |     -                   |     -                   |
[{{ error-function__BalancedErrorRate }}](#BalancedErrorRate) |     -                   |     -                   |
[{{ error-function--MCC }}](#MCC)                             |     -                   |     +                   |
[{{ error-function--Accuracy }}](#Accuracy)                   |     -                   |     +                   |
[{{ error-function__CtrFactor }}](#CtrFactor)                 |     -                   |     -                   |
[{{ error-function--AUC }}](#AUC)                             |     -                   |     -                   |
[{{ error-function--QueryAUC }}](#QueryAUC)                   |     -                   |     -                   |
[{{ error-function--NormalizedGini }}](#ormalizedGini)        |     -                   |     -                   |
[{{ error-function__Brierscore }}](#Brierscore)               |     -                   |     -                   |
[{{ error-function__HingeLoss }}](#HingeLoss)                 |     -                   |     -                   |
[{{ error-function__HammingLoss }}](#HammingLoss)             |     -                   |     -                   |
[{{ error-function__ZeroOneLoss }}](#ZeroOneLoss)             |     -                   |     +                   |
[{{ error-function__Kappa }}](#Kappa)                         |     -                   |     -                   |
[{{ error-function__WKappa }}](#WKappa)                       |     -                   |     -                   |
[{{ error-function__LLP }}](#LLP)                             |     -                   |     -                   |


