# MultiLabel Classification: objectives and metrics

- [Objectives and metrics](#objectives-and-metrics)
- [{{ title__loss-functions__text__optimization }}](#usage-information)

## Objectives and metrics

### {{ error-function__MultiLogloss }} {#MultiLogloss}

  $\displaystyle\frac{-\sum\limits_{j=0}^{M-1} \sum\limits_{i=1}^{N} w_{i} (c_{ij} \log p_{ij} + (1-c_{ij}) \log (1 - p_{ij}) )}{M\sum\limits_{i=1}^{N}w_{i}} { ,}$

  where $p_{ij} = \sigma(a_{ij}) = \frac{e^{a_{ij}}}{1 + e^{a_{ij}}}$ and $c_{ij} \in {0, 1}$

**{{ optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function__MultiCrossEntropy }} {#MultiCrossEntropy}

$\displaystyle\frac{-\sum\limits_{j=0}^{M-1} \sum\limits_{i=1}^{N} w_{i} (t_{ij} \log p_{ij} + (1-t_{ij}) \log (1 - p_{ij}) )}{M\sum\limits_{i=1}^{N}w_{i}} { ,}$

  where $p_{ij} = \sigma(a_{ij}) = \frac{e^{a_{ij}}}{1 + e^{a_{ij}}}$ and $t_{ij} \in [0, 1]$

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

### {{ error-function--Accuracy }} {#Accuracy}

The formula depends on the value of the ${{ loss-functions__params__type }}$ parameter:

#### {{ loss-functions__params__accuracy__type__Classic }}

$\displaystyle\frac{\sum\limits_{i=1}^{N}w_{i} \prod\limits_{j=0}^{M-1} [[p_{ij} > 0.5]==t_{ij}]}{\sum\limits_{i=1}^{N}w_{i}} { , }$

where $p_{ij} = \sigma(a_{ij}) = \frac{e^{a_{ij}}}{1 + e^{a_{ij}}}$

#### {{ loss-functions__params__accuracy__type__PerClass }}

{% include [reusage-loss-function__for-multiclass](../_includes/work_src/reusage/loss-function__for-multiclass.md) %}

$\frac{TP + TN}{\sum\limits_{i=1}^{N} w_{i}}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% cut "{{ loss-functions__params__type }}" %}

The type of calculated accuracy.

_Default:_ `Classic`.
_Possible values_: `{{ loss-functions__params__accuracy__type__Classic }}`, `{{ loss-functions__params__accuracy__type__PerClass }}`.

{% endcut %}


### {{ error-function__HammingLoss }} {#HammingLoss}

$\displaystyle\frac{\sum\limits_{j=0}^{M-1} \sum\limits_{i = 1}^{N} w_{i} [[p_{ij} > 0.5] == t_{ij}]]}{M \sum\limits_{i=1}^{N} w_{i}}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}



## {{ title__loss-functions__text__optimization }} {#usage-information}

| Name                                                          | Optimization            | GPU Support             |
----------------------------------------------------------------|-------------------------|-------------------------|
[{{ error-function__MultiLogloss }}](#MultiLogloss)             |     +                   |     +                   |
[{{ error-function__MultiCrossEntropy }}](#MultiCrossEntropy)   |     +                   |     +                   |
[{{ error-function--Precision }}](#Precision)                   |     -                   |     -                   |
[{{ error-function--Recall }}](#Recall)                         |     -                   |     -                   |
[{{ error-function--F }}](#F)                                   |     -                   |     -                   |
[{{ error-function--F1 }}](#F1)                                 |     -                   |     -                   |
[{{ error-function--Accuracy }}](#Accuracy)                     |     -                   |     -                   |
[{{ error-function__HammingLoss }}](#HammingLoss)               |     -                   |     -                   |
