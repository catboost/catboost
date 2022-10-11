# Multiclassification: objectives and metrics

- [Objectives and metrics](#objectives-and-metrics)
- [{{ title__loss-functions__text__optimization }}](#usage-information)

## Objectives and metrics

### {{ error-function--MultiClass }} {#MultiClass}

$\displaystyle\frac{\sum\limits_{i=1}^{N}w_{i}\log\left(\displaystyle\frac{e^{a_{it_{i}}}}{ \sum\limits_{j=0}^{M - 1}e^{a_{ij}}} \right)}{\sum\limits_{i=1}^{N}w_{i}} { ,}$

$t \in \{0, ..., M - 1\}$

**{{ optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function--MultiClassOneVsAll }} {#MultiClassOneVsAll}

$\displaystyle\frac{\frac{1}{M}\sum\limits_{i = 1}^N w_i \sum\limits_{j = 0}^{M - 1} [j = t_i] \log(p_{ij}) + [j \neq t_i] \log(1 - p_{ij})}{\sum\limits_{i = 1}^N w_i} { ,}$

$t \in \{0, ..., M - 1\}$

**{{ optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function--Precision }} {#Precision}

{% include [reusage-loss-function__for-multiclass](../_includes/work_src/reusage/loss-function__for-multiclass.md) %}

$\frac{TP}{TP + FP}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function--Recall }} {#Recall}

{% include [reusage-loss-function__for-multiclass](../_includes/work_src/reusage/loss-function__for-multiclass.md) %}

$\frac{TP}{TP+FN}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function--F }} {#F}

{% include [reusage-loss-function__for-multiclass](../_includes/work_src/reusage/loss-function__for-multiclass.md) %}

$(1 + \beta^2) \cdot  \frac{Precision * Recall}{(\beta^2 \cdot Precision) + Recall}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [beta_for_F__desc](../_includes/work_src/reusage-loss-functions/beta_for_F__desc.md) %}

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function--F1 }} {#F1}

{% include [reusage-loss-function__for-multiclass](../_includes/work_src/reusage/loss-function__for-multiclass.md) %}

$2 \frac{Precision * Recall}{Precision + Recall}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function--totalF1 }} {#totalF1}

The formula depends on the value of the {{ loss-functions__params__average__name }} parameter:

#### {{ loss-functions__params__average__Weighted }}

$\frac{\sum\limits_{i=1}^{M} w_{i} F1_{i}}{\sum\limits_{i=1}^{M}w_{i}} {, where}$

$w_{i}$ is the sum of the weights of the documents which correspond to the i-th class. If document weights are not specified $w_{i}$ stands for the number of times the i-th class is found among the label values.

#### {{ loss-functions__params__average__Macro }}

$\displaystyle\frac{\sum\limits_{i=1}^{M}F1_{i}}{M}$

#### {{ loss-functions__params__average__Micro }}

$TotalF1 = \displaystyle\frac{2 \cdot TP}{2 \cdot TP + FP + FN} {, where}$
- $TP = \sum\limits_{i=1}^{M} TP_{i}$
- $FP = \sum\limits_{i=1}^{M} FP_{i}$
- $FN = \sum\limits_{i=1}^{M} FN_{i}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

{% cut "{{ loss-functions__params__average__name }}" %}

The method for averaging the value of the metric that is initially individually calculated for each class.

_Default:_  `{{ loss-functions__params__average__Weighted }}`.
_Possible values:_ `{{ loss-functions__params__average__Weighted }}`, `{{ loss-functions__params__average__Macro }}`, `{{ loss-functions__params__average__Micro }}`.

{% endcut %}

### {{ error-function--MCC }} {#MCC}

This functions is defined in terms of a $k \times k$ confusion matrix $C$ (where k is the number of classes):

$\displaystyle\frac{\sum\limits_{k}\sum\limits_{l}\sum\limits_{m} C_{kk} C_{lm} - C_{kl}C_{mk}}{\sqrt{\sum\limits_{k} \left(\sum\limits_{l} C_{kl}\right) \left(\sum\limits_{k' | k' \neq k} \sum\limits_{l'} C_{k'l'}\right)}\sqrt{\sum\limits_{k} \left(\sum\limits_{l} C_{lk}\right) \left(\sum\limits_{k' | k' \neq k} \sum\limits_{l'} C_{l' k'}\right)}}$

See the [Wikipedia article](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient) for more details.

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function--Accuracy }} {#Accuracy}

$\displaystyle\frac{\sum\limits_{i=1}^{N}w_{i}[argmax_{j=0,...,M - 1}(a_{ij})==t_{i}]}{\sum\limits_{i=1}^{N}w_{i}} { , }$

$t \in \{0, ..., M - 1\}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function__HingeLoss }} {#HingeLoss}

See the [Wikipedia article](https://en.wikipedia.org/wiki/Hinge_loss).

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function__HammingLoss }} {#HammingLoss}

$\displaystyle\frac{\sum\limits_{i = 1}^{N} w_{i} [argmax_{j=0,...,M - 1}(a_{ij})\neq t_{i}]}{\sum\limits_{i = 1}^{N} w_{i}}$

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

$RAccuracy = \displaystyle\frac{\sum\limits_{k=0}^{M - 1} n_{k_{a}}n_{k_{t}}}{(\sum\limits_{i=1}^{N}w_{i})^{2}}$

$k_{a}$ is the weighted number of times class k is predicted by the model

$k_{t}$ is the weighted number of times class k is set as the label for input objects

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function__WKappa }} {#WKappa}

See the formula on page 3 of the [A note on the linearly weighted kappa coefficient for ordinal scales](https://orbi.uliege.be/bitstream/2268/2262/1/STATMED-174.pdf) paper.

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function--AUC }} {#AUC}

The calculation of this metric is disabled by default for the training dataset to speed up the training. Use the `hints=skip_train~false` parameter to enable the calculation.

- {{ loss-functions__params__auc__type__mu }}

    Refer to the [A Performance Metric for Multi-Class Machine Learning Models](http://proceedings.mlr.press/v97/kleiman19a/kleiman19a.pdf) paper for calculation principles

- {{ loss-functions__params__auc__type__onevsall }}

    The value is calculated separately for each class k numbered from 0 to M–1 according to the [binary classification calculation principles](../concepts/loss-functions-classification.md#auc__full-desc). The objects of class k are considered positive, while all others are considered negative.

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% cut "{{loss-functions__params__use_weights}}" %}

{% include [use-weights__desc__without__note](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__note.md) %}

_Default_: `False`.
_Examples_: `AUC:type=Ranking;use_weights=False`.

{% endcut %}

{% cut "{{ loss-functions__params__auc__type }}" %}

{% include [reusage-loss-functions-type_of_auc__p](../_includes/work_src/reusage-loss-functions/type_of_auc__p.md) %}

_Default:_ {{ loss-functions__params__auc__type__mu }}
_Possible values:_ `{{ loss-functions__params__auc__type__mu }}`, `{{ loss-functions__params__auc__type__onevsall }}`.
_Examples_: `AUC:type=Mu`, `AUC:type=OneVsAll`.

{% endcut %}

{% cut "{{ loss-functions__params__auc__misclass_cost_matrix }}" %}

The matrix _M_ with misclassification cost values. $M[i,j]$ in this matrix is the cost of classifying an object as a member of the class _i_ when its' actual class is _j_. Applicable only if the used type of AUC is {{ loss-functions__params__auc__type__mu }}.

Format for a matrix of size C:
```
<Value for M[0,0]>, <Value for M[0,1]>, ..., <Value for M[0,C-1]>, <Value for M[1,0]>, ..., <Value for M[C-1,0]>, ..., <Value for M[C-1,C-1]>
```

All diagonal elements $M[i, j]$ (such that _i=j_) must be equal to 0.

{% note info %}

The `{{ loss-functions__params__auc__type }}` parameter is optional and is assumed to be set to `{{ loss-functions__params__auc__type__mu }}` if the parameter is explicitly specified.

{% endnote %}

_Default:_  All non-diagonal matrix elements are set to 1. All diagonal elements $М[i, j]$ (such that _i = j_) are set to 0.
_Examples:_ Three classes — `AUC:misclass_cost_matrix=0/0.5/2/1/0/1/0/0.5/0`, Two classes — `AUC:type=Mu;misclass_cost_matrix=0/0.5/1/0`.

{% endcut %}

{% include [pr-auc](../_includes/concepts/pr-auc.md) %}


## {{ title__loss-functions__text__optimization }} {#usage-information}

| Name                                                          | Optimization            | GPU Support             |
----------------------------------------------------------------|-------------------------|-------------------------|
[{{ error-function--MultiClass }}](#MultiClass)                 |     +                   |     +                   |
[{{ error-function--MultiClassOneVsAll }}](#MultiClassOneVsAll) |     +                   |     +                   |
[{{ error-function--Precision }}](#Precision)                   |     -                   |     +                   |
[{{ error-function--Recall }}](#Recall)                         |     -                   |     +                   |
[{{ error-function--F }}](#F)                                   |     -                   |     -                   |
[{{ error-function--F1 }}](#F1)                                 |     -                   |     +                   |
[{{ error-function--totalF1 }}](#totalF1)                       |     -                   |     +                   |
[{{ error-function--MCC }}](#MCC)                               |     -                   |     +                   |
[{{ error-function--Accuracy }}](#Accuracy)                     |     -                   |     +                   |
[{{ error-function__HingeLoss }}](#HingeLoss)                   |     -                   |     -                   |
[{{ error-function__HammingLoss }}](#HammingLoss)               |     -                   |     -                   |
[{{ error-function__ZeroOneLoss }}](#ZeroOneLoss)               |     -                   |     +                   |
[{{ error-function__Kappa }}](#Kappa)                           |     -                   |     -                   |
[{{ error-function__WKappa }}](#WKappa)                         |     -                   |     -                   |
[{{ error-function--AUC }}](#AUC)                               |     -                   |     -                   |
