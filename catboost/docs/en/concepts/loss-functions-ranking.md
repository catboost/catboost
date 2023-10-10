# Ranking: objectives and metrics

## Pairwise metrics {#pairwise-metrics}

Pairwise metrics use special labeled information — pairs of dataset objects where one object is considered the <q>winner</q> and the other is considered the <q>loser</q>. This information might be not exhaustive (not all possible pairs of objects are labeled in such a way). It is also possible to specify the weight for each pair.

If GroupId is specified, then all pairs must have both members from the same group if this dataset is used in pairwise modes.

{% cut "Read more about {{ cd-file__col-type__GroupId }}" %}

The identifier of the object's group. An arbitrary string, possibly representing an integer.

{% endcut %}

If the labeled pairs data is not specified for the dataset, then pairs are generated automatically in each group using per-object label values (labels must be specified and must be numerical). The object with a greater label value in the pair is considered the <q>winner</q>.

The following variables are used in formulas of the described pairwise metrics:
- $p$ is the positive object in the pair.
- $n$ is the negative object in the pair.

See all common variables in [Variables used in formulas](loss-functions-variables-used).

### {{ error-function__PairLogit }} {#PairLogit}

$\displaystyle\frac{-\sum\limits_{p, n \in Pairs} w_{pn} \left(log(\displaystyle\frac{1}{1 + e^{- (a_{p} - a_{n})}})\right)}{\sum\limits_{p, n \in Pairs} w_{pn}}$

{% include [objectives-and-metrics-the-weights-of-objects-not-used-to-calculate-metrics](../_includes/work_src/reusage-loss-functions/the-weights-of-objects-not-used-to-calculate-metrics.md) %}

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

{% cut "{{ loss-functions__params__max_pairs }}" %}

The maximum number of generated pairs in each group. Takes effect if no pairs are given and therefore are generated without repetition.

_Default:_ {{ loss-functions__params__max_pairs_default }}

{% endcut %}

### {{ error-function__PairLogitPairwise }} {#PairLogitPairwise}

$\displaystyle\frac{-\sum\limits_{p, n \in Pairs} w_{pn} \left(log(\displaystyle\frac{1}{1 + e^{- (a_{p} - a_{n})}})\right)}{\sum\limits_{p, n \in Pairs} w_{pn}}$

This metric may give more accurate results on large datasets compared to {{ error-function__PairLogit }} but it is calculated significantly slower.

This technique is described in the [Winning The Transfer Learning Track of Yahoo!’s Learning To Rank Challenge with YetiRank](http://proceedings.mlr.press/v14/gulin11a.html) paper.

**{{ optimization }}** See [more](#usage-information).

{% include [objectives-and-metrics-the-weights-of-objects-not-used-to-calculate-metrics](../_includes/work_src/reusage-loss-functions/the-weights-of-objects-not-used-to-calculate-metrics.md) %}

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

{% cut "{{ loss-functions__params__max_pairs }}" %}

The maximum number of generated pairs in each group. Takes effect if no pairs are given and therefore are generated without repetition.

_Default:_  {{ loss-functions__params__max_pairs_default }}

{% endcut %}

### {{ error-function__PairAccuracy }} {#PairAccuracy}

$\displaystyle\frac{\sum\limits_{p, n \in Pairs} w_{pn} [a_{p} > a_{n}] }{\sum\limits_{p, n \in Pairs} w_{pn} }$

{% include [objectives-and-metrics-the-weights-of-objects-not-used-to-calculate-metrics-not-optimize](../_includes/work_src/reusage-loss-functions/the-weights-of-objects-not-used-to-calculate-metrics-not-optimize.md) %}

**{{ no-optimization }}**  See [more](#usage-information).

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

## Groupwise metrics {#groupwise-metrics}

### {{ error-function__YetiRank }} {#YetiRank}

The calculation of this metric is disabled by default for the training dataset to speed up the training. Use the `hints=skip_train~false` parameter to enable the calculation.

{% include [objectives-and-metrics-yetirank__desc__first](../_includes/work_src/reusage-loss-functions/yetirank__desc__first.md) %}

{% include [objectives-and-metrics-yetirank__desc__last](../_includes/work_src/reusage-loss-functions/yetirank__desc__last.md) %}

This metric gives less accurate results on big datasets compared to {{ error-function__YetiRankPairwise }} but it is significantly faster.

{% include [objectives-and-metrics-the-weights-of-groups](../_includes/work_src/reusage-loss-functions/the-weights-of-groups.md) %}

**{{ optimization }}** See [more](#usage-information).

Since CatBoost 1.2.1 YetiRank meaning has been expanded to allow for optimizing specific ranking loss functions by specifying `mode` loss function parameter. Default YetiRank can now also be referred as `mode=Classic`.

**{{ title__loss-functions__text__user-defined-params }}**

{% cut "{{ loss-functions__params__mode }}" %}

The mode of operation. Either `Classic` - the traditional YetiRank as described in [Winning The Transfer Learning Track of Yahoo!’s Learning To Rank Challenge with YetiRank](http://proceedings.mlr.press/v14/gulin11a.html) or a specific ranking loss function to optimize as described in [Which Tricks are Important for Learning to Rank?](https://arxiv.org/abs/2204.01500) paper. Possible loss function values are `DCG`, `NDCG`, `MRR`, `ERR`, `MAP`. Non-Classic modes are supported only on CPU.

_Default:_  `Classic`

{% endcut %}

{% cut "{{ loss-functions__params__permutations }}" %}

The number of permutations.

_Default:_ {{ fit__permutations }}

{% endcut %}

{% cut "{{ loss-functions__params__decay }}" %}

Used only in `Classic` mode.
The probability of search continuation after reaching the current object.

_Default:_  {{ fit__decay__yetirank }}

{% endcut %}

{% cut "{{ loss-functions__params__top }}" %}

Used in all modes except `Classic`.

The number of top samples in a group that are used to calculate the ranking metric. Top samples are either the samples with the largest approx values or the ones with the lowest target values if approx values are the same.

Unlimited by default.

{% endcut %}

{% cut "dcg_type" %}

Used in modes `DCG` and `NDCG`.

Principle of calculation of \*DCG metrics.

_Default_: {{ error-function__filtereddcg__type__default }}.
_Possible values_: `{{ error-function__ndcg__type__Base }}`, `{{ error-function__ndcg__type__Exp }}`.

{% endcut %}

{% cut "dcg_denominator" %}

Used in modes `DCG` and `NDCG`.

Principle of calculation of the denominator in \*DCG metrics.

_Default_: {{ error-function__filtereddcg__denominator__default }}.
_Possible values_: `{{ error-function__ndcg__denominator__LogPosition }}`, `{{ error-function__ndcg__denominator__Position }}`.

{% endcut %}

{% cut "noise" %}

Type of noise to add to approxes.

_Default_: `Gumbel`.
_Possible values_: `Gumbel`, `Gauss`, `No`.

{% endcut %}

{% cut "noise_power" %}

Power of noise to add (multiplier). Used only for `Gauss` noise for now.

_Default_: 1.

{% endcut %}

{% cut "num_neighbors" %}

Used in all modes except `Classic`.

Number of neighbors used in the metric calculation.

_Default_: 1.

{% endcut %}

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function__YetiRankPairwise }} {#YetiRankPairwise}

The calculation of this metric is disabled by default for the training dataset to speed up the training. Use the `hints=skip_train~false` parameter to enable the calculation.

{% include [objectives-and-metrics-yetirank__desc__first](../_includes/work_src/reusage-loss-functions/yetirank__desc__first.md) %}

{% include [objectives-and-metrics-yetirank__desc__last](../_includes/work_src/reusage-loss-functions/yetirank__desc__last.md) %}

This metric gives more accurate results on big datasets compared to {{ error-function__YetiRank }} but it is significantly slower.

This technique is described in the [Winning The Transfer Learning Track of Yahoo!’s Learning To Rank Challenge with YetiRank](http://proceedings.mlr.press/v14/gulin11a.html) paper.

{% include [objectives-and-metrics-the-weights-of-groups](../_includes/work_src/reusage-loss-functions/the-weights-of-groups.md) %}

**{{ optimization }}** See [more](#usage-information).

Since CatBoost 1.2.1 YetiRankPairwise meaning has been expanded to allow for optimizing specific ranking loss functions by specifying `mode` loss function parameter. Default YetiRankPairwise can now also be referred as `mode=Classic`.

**{{ title__loss-functions__text__user-defined-params }}**

{% cut "{{ loss-functions__params__mode }}" %}

The mode of operation. Either `Classic` - the traditional YetiRankPairwise as described in [Winning The Transfer Learning Track of Yahoo!’s Learning To Rank Challenge with YetiRank](http://proceedings.mlr.press/v14/gulin11a.html) or a specific ranking loss function to optimize as described in [Which Tricks are Important for Learning to Rank?](https://arxiv.org/abs/2204.01500) paper. Possible loss function values are `DCG`, `NDCG`, `MRR`, `ERR`, `MAP`. Non-Classic modes are supported only on CPU.

_Default:_  `Classic`

{% endcut %}

{% cut "{{ loss-functions__params__permutations }}" %}

The number of permutations.

_Default:_ {{ fit__permutations }}

{% endcut %}

{% cut "{{ loss-functions__params__decay }}" %}

Used only in `Classic` mode.
The probability of search continuation after reaching the current object.

_Default:_  {{ fit__decay__yetirank }}

{% endcut %}

{% cut "{{ loss-functions__params__top }}" %}

Used in all modes except `Classic`.

The number of top samples in a group that are used to calculate the ranking metric. Top samples are either the samples with the largest approx values or the ones with the lowest target values if approx values are the same.

Unlimited by default.

{% endcut %}

{% cut "dcg_type" %}

Used in modes `DCG` and `NDCG`.

Principle of calculation of \*DCG metrics.

_Default_: {{ error-function__filtereddcg__type__default }}.
_Possible values_: `{{ error-function__ndcg__type__Base }}`, `{{ error-function__ndcg__type__Exp }}`.

{% endcut %}

{% cut "dcg_denominator" %}

Used in modes `DCG` and `NDCG`.

Principle of calculation of the denominator in \*DCG metrics.

_Default_: {{ error-function__filtereddcg__denominator__default }}.
_Possible values_: `{{ error-function__ndcg__denominator__LogPosition }}`, `{{ error-function__ndcg__denominator__Position }}`.

{% endcut %}

{% cut "noise" %}

Type of noise to add to approxes.

_Default_: `Gumbel`.
_Possible values_: `Gumbel`, `Gauss`, `No`.

{% endcut %}

{% cut "noise_power" %}

Power of noise to add (multiplier). Used only for `Gauss` noise for now.

_Default_: 1.

{% endcut %}

{% cut "num_neighbors" %}

Used in all modes except `Classic`.

Number of neighbors used in the metric calculation.

_Default_: 1.

{% endcut %}

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function__LambdaMart }} {#LambdaMart}

Directly optimize the selected metric. The value of the selected metric is written to [output data](../concepts/output-data.md)

Refer to the [From RankNet to LambdaRank to LambdaMART](https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf) paper for details.

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% cut "{{ loss-functions__params__metric }}" %}

The metric that should be optimized.

_Default_: `{{ error-function__ndcg }}`
_Supported values_: `{{ error-function__dcg }}`, `{{ error-function__ndcg }}`, `{{ error-function__mrr }}`, `{{ error-function__err }}`, `{{ error-function__mapk }}`.

{% endcut %}

{% cut "{{ loss-functions__params__sigma }}" %}

General sigmoid parameter. See [From RankNet to LambdaRank to LambdaMART](https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf) paper for details.

_Default_: 1.0
_Supported values_: Real positive values.

{% endcut %}

{% cut "norm" %}

Derivatives should be normalized.

_Default_: True
_Supported values_: False, True.

{% endcut %}


### {{ error-function__StochasticFilter }} {#StochasticFilter}

Directly optimize the {{ error-function__FilteredDCG }} metric calculated for a pre-defined order of objects for filtration of objects under a fixed ranking. As a result, the {{ error-function__FilteredDCG }} metric can be used for optimization.

$FilteredDCG = \sum\limits_{i=1}^{n}\displaystyle\frac{t_{i}}{i} { , where}$

$t_{i}$ is the relevance of an object in the group and the sum is computed over the documents with $a > 0$.

The filtration is defined via the raw formula value:

![](../images/formula__stohastic.png)

Zeros correspond to filtered instances and ones correspond to the remaining ones.

The ranking is defined by the order of objects in the dataset.

{% note warning %}

Sort objects by the column you are interested in before training with this loss function and use the `--has-time`for the Command-line version option to avoid further objects reordering.

{% endnote %}

For optimization, a distribution of filtrations is defined:

$\mathbb{P}(\text{filter}|x) = \sigma(a) { , where}$
- $\sigma(z) = \displaystyle\frac{1}{1 + \text{e}^{-z}}$
- The gradient is estimated via REINFORCE.

Refer to the [Learning to Select for a Predefined Ranking]({{ learning-to-select-for-predefined-ranking }}) paper for calculation details.

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% cut "{{ loss-functions__params__sigma }}" %}

The scale for multiplying predictions.

_Default:_ 1

{% endcut %}

{% cut "{{ loss-functions__params__num_estimations }}" %}

The number of gradient samples.

_Default:_ 1

{% endcut %}


### {{ error-function__StochasticRank }} {#StochasticRank}

Directly optimize the selected metric. The value of the selected metric is written to [output data](../concepts/output-data.md)

Refer to the [StochasticRank: Global Optimization of Scale-Free Discrete Functions](https://arxiv.org/abs/2003.02122v1) paper for details.

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

Common parameters:

{% cut "{{ loss-functions__params__metric }}" %}

The metric that should be optimized.

_Default_: {{ loss-functions__params__q__default }}
_Supported values_: `{{ error-function__dcg }}`, `{{ error-function__ndcg }}`, `{{ error-function__PFound }}`.

{% endcut %}

{% cut "{{ loss-functions__params__num_estimations }}" %}

The number of gradient estimation iterations.

_Default_: 1

{% endcut %}

{% cut "{{ loss-functions__params__mu }}" %}

Controls the penalty for coinciding predictions (aka _ties_).

_Default_: 0

{% endcut %}

Metric-specific parameters:

Available if the corresponding metric is set in the {{ loss-functions__params__metric }} parameter.

**{{ error-function__dcg }}**

{% include [top__desc__full](../_includes/work_src/reusage-loss-functions/top__desc__full.md) %}

{% include [ndcg__type__desc__full](../_includes/work_src/reusage-loss-functions/ndcg__type__desc__full.md) %}

{% cut "{{ loss-functions__params__denominator }}" %}

Metric denominator type.

_Default_: _Default_: {{ error-function__dcg__denominator__default }}.
_Possible values_: `{{ error-function__ndcg__denominator__LogPosition }}`, `{{ error-function__ndcg__denominator__Position }}`.

{% endcut %}

**{{ error-function__ndcg }}**

{% include [top__desc__full](../_includes/work_src/reusage-loss-functions/top__desc__full.md) %}

{% include [ndcg__type__desc__full](../_includes/work_src/reusage-loss-functions/ndcg__type__desc__full.md) %}

{% cut "{{ loss-functions__params__denominator }}" %}

Metric denominator type.

_Default_: {{ error-function__dcg__denominator__default }}.
_Possible values_: `{{ error-function__ndcg__denominator__LogPosition }}`, `{{ error-function__ndcg__denominator__Position }}`.

{% endcut %}

**{{ error-function__PFound }}**

{% include [decay__desc__full](../_includes/work_src/reusage-loss-functions/decay__desc__full.md) %}

{% include [top__desc__full](../_includes/work_src/reusage-loss-functions/top__desc__full.md) %}

### {{ error-function__QueryCrossEntropy }} {#QueryCrossEntropy}

$QueryCrossEntropy(\alpha) = (1 - \alpha) \cdot LogLoss + \alpha \cdot LogLoss_{group}$

See the [{{ error-function__QueryCrossEntropy }}](../references/querycrossentropy.md) section for more details.

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

{% cut "{{ loss-functions__params__alpha }}" %}

The coefficient used in quantile-based losses.

_Default:_ 0.95

{% endcut %}

### {{ error-function__QueryRMSE }} {#QueryRMSE}

$\displaystyle\sqrt{\displaystyle\frac{\sum\limits_{Group \in Groups} \sum\limits_{i \in Group} w_{i} \left( t_{i} - a_{i} - \displaystyle\frac{\sum\limits_{j \in Group} w_{j} (t_{j} - a_{j})}{\sum\limits_{j \in Group} w_{j}} \right)^{2}} {\sum\limits_{Group \in Groups} \sum\limits_{i \in Group} w_{i}}}$

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function__QuerySoftMax }} {#QuerySoftMax}

$- \displaystyle\frac{\sum\limits_{Group \in Groups} \sum\limits_{i \in Group}w_{i} t_{i} \log \left(\displaystyle\frac{w_{i} e^{\beta a_{i}}}{\sum\limits_{j\in Group} w_{j} e^{\beta a_{j}}}\right)} {\sum\limits_{Group \in Groups} \sum_{i\in Group} w_{i} t_{i}}$

**{{ optimization }}** See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

{% cut "{{ loss-functions__params__beta }}" %}

The input scale coefficient.

_Default:_ 1

{% endcut %}

### {{ error-function__PFound }} {#PFound}

The calculation of this metric is disabled by default for the training dataset to speed up the training. Use the `hints=skip_train~false` parameter to enable the calculation.

$PFound(top, decay) =$

$= \sum_{group \in groups} PFound(group, top, decay)$

See the [{{ error-function__PFound }}](../references/pfound.md) section for more details

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [decay__desc__full](../_includes/work_src/reusage-loss-functions/decay__desc__full.md) %}

{% include [top__desc__full](../_includes/work_src/reusage-loss-functions/top__desc__full.md) %}

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function__ndcg }} {#ndcg}

The calculation of this metric is disabled by default for the training dataset to speed up the training. Use the `hints=skip_train~false` parameter to enable the calculation.

$nDCG(top) = \frac{DCG(top)}{IDCG(top)}$

See the [{{ error-function__ndcg }}](../references/ndcg.md) section for more details.

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [top__desc__full](../_includes/work_src/reusage-loss-functions/top__desc__full.md) %}

{% include [ndcg__type__desc__full](../_includes/work_src/reusage-loss-functions/ndcg__type__desc__full.md) %}

{% include [ndcg__denominator__desc__full](../_includes/work_src/reusage-loss-functions/ndcg__denominator__desc__full.md) %}

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function__dcg }} {#dcg}

The calculation of this metric is disabled by default for the training dataset to speed up the training. Use the `hints=skip_train~false` parameter to enable the calculation.

$DCG(top)$

See the [{{ error-function__ndcg }}](../references/ndcg.md) section for more details.

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [top__desc__full](../_includes/work_src/reusage-loss-functions/top__desc__full.md) %}

{% include [ndcg__type__desc__full](../_includes/work_src/reusage-loss-functions/ndcg__type__desc__full.md) %}

{% include [ndcg__denominator__desc__full](../_includes/work_src/reusage-loss-functions/ndcg__denominator__desc__full.md) %}

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}

### {{ error-function__FilteredDCG }} {#FilteredDCG}

The calculation of this metric is disabled by default for the training dataset to speed up the training. Use the `hints=skip_train~false` parameter to enable the calculation.

See the [{{ error-function__FilteredDCG }}](../references/filtereddcg.md) section for more details.

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [ndcg__type__desc__full](../_includes/work_src/reusage-loss-functions/ndcg__type__desc__full.md) %}

{% include [ndcg__denominator__desc__full](../_includes/work_src/reusage-loss-functions/ndcg__denominator__desc__full.md) %}

### {{ error-function__QueryAverage }} {#QueryAverage}

Represents the average value of the label values for objects with the defined top $M$ label values.

See the [{{ error-function__QueryAverage }}](../references/queryaverage.md) section for more details.

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% cut "{{ loss-functions__params__top }}" %}

The number of top samples in a group that are used to calculate the ranking metric. Top samples are either the samples with the largest approx values or the ones with the lowest target values if approx values are the same.

_Default_: {{ loss-functions__obligatory-text }}.

{% endcut %}

{% include [use-weights__desc__with_default_value](../_includes/work_src/reusage-loss-functions/use-weights__desc__with__default__value.md) %}


### {{ error-function__PrecisionAtK }} {#PrecisionAtK}

{% include [loss-functions-function-calculation](../_includes/work_src/reusage-common-phrases/function-calculation.md) %}

1. The objects are sorted in descending order of predicted relevancies ($a_{i}$)

1. The metric is calculated as follows:

    $PrecisionAt(top, border) = \frac{\sum\limits_{i=1}^{top} Relevant_{i}}{top} { , where}$
    - $Relevant_{i} = \begin{cases} 1 { , } & t_{i} > {border} \\ 0 { , } & {in other cases} \end{cases}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [top__desc__full](../_includes/work_src/reusage-loss-functions/top__desc__full.md) %}

{% include [ranking__border__desc__full](../_includes/work_src/reusage-loss-functions/ranking__border__desc__full.md) %}


### {{ error-function__RecallAtK }} {#RecallAtK}

{% include [loss-functions-function-calculation](../_includes/work_src/reusage-common-phrases/function-calculation.md) %}

1. The objects are sorted in descending order of predicted relevancies ($a_{i}$)

1. The metric is calculated as follows:
    $RecalAt(top, border) = \frac{\sum\limits_{i=1}^{top} Relevant_{i}}{\sum\limits_{i=1}^{N} Relevant_{i}}$
    - $Relevant_{i} = \begin{cases} 1 { , } & t_{i} > {border} \\ 0 { , } & {in other cases} \end{cases}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [top__desc__full](../_includes/work_src/reusage-loss-functions/top__desc__full.md) %}

{% include [ranking__border__desc__full](../_includes/work_src/reusage-loss-functions/ranking__border__desc__full.md) %}


### {{ error-function__mapk }} {#mapk}

1. The objectsare sorted in descending order of predicted relevancies ($a_{i}$)

1. The metric is calculated as follows:
    $MAP(top, border) = \frac{1}{N_{groups}} \sum\limits_{j = 1}^{N_{groups}} AveragePrecisionAt_{j}(top, border) { , where}$
    - $N_{groups}$ is the number of groups
    - $AveragePrecisionAt(top, border) = \frac{\sum\limits_{i=1}^{top} Relevant_{i} * PrecisionAt_{i}}{\sum\limits_{i=1}^{top} Relevant_{i} }$

    The value is calculated individually for each _j_-th group.

    - $Relevant_{i} = \begin{cases} 1 { , } & t_{i} > {border} \\ 0 { , } & {in other cases} \end{cases}$
    - $PrecisionAt_{i} = \frac{\sum\limits_{j=1}^{i} Relevant_{j}}{i}$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [top__desc__full](../_includes/work_src/reusage-loss-functions/top__desc__full.md) %}

{% include [ranking__border__desc__full](../_includes/work_src/reusage-loss-functions/ranking__border__desc__full.md) %}


### {{ error-function__err }} {#err}

$ERR = \frac{1}{|Q|} \sum_{q=1}^{|Q|} ERR_q$

$ERR_q = \sum_{i=1}^{top} \frac{1}{i} t_{q,i} \prod_{j=1}^{i-1} (1 - t_{q,j})$

Targets should be from the range [0, 1].

$t_{q,i} \in [0, 1]$

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [top__desc__full](../_includes/work_src/reusage-loss-functions/top__desc__full.md) %}


### {{ error-function__mrr }} {#mrr}

$MRR = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{rank_q}$, where $rank_q$ refers to the rank position of the first relevant document for the _q_-th query.

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% include [top__desc__full](../_includes/work_src/reusage-loss-functions/top__desc__full.md) %}

{% include [ranking__border__desc__full](../_includes/work_src/reusage-loss-functions/ranking__border__desc__full.md) %}


### {{ error-function--AUC }} {#AUC}

The calculation of this metric is disabled by default for the training dataset to speed up the training. Use the `hints=skip_train~false` parameter to enable the calculation.

{% include [reusage-loss-functions-type_of_auc__p](../_includes/work_src/reusage-loss-functions/type_of_auc__p.md) %}

{% include [loss-functions-loss-functions-ranking__auc](../_includes/work_src/reusage-popups/loss-functions-ranking__auc.md) %}

**{{ no-optimization }}**  See [more](#usage-information).

**{{ title__loss-functions__text__user-defined-params }}**

{% cut "{{ loss-functions__params__type }}" %}

The type of AUC. Defines the metrics calculation principles.

_Default_: `{{ loss-functions__params__auc__type__Classic }}`.
_Possible values_: `{{ loss-functions__params__auc__type__Classic }}`, `{{ loss-functions__params__auc__type__Ranking }}`.
_Examples_: `AUC:type=Classic`, `AUC:type=Ranking`.

{% endcut %}

{% cut "{{loss-functions__params__use_weights}}" %}

{% include [use-weights__desc__without__note](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__note.md) %}

_Default_: `False` for Classic type, `True` for Ranking type.
_Examples_: `AUC:type=Ranking;use_weights=False`.

{% endcut %}

{% include [query-auc](../_includes/concepts/query-auc.md) %}

## {{ title__loss-functions__text__optimization }} {#usage-information}


| Name                                                        | Optimization            | GPU Support             |
--------------------------------------------------------------|-------------------------|-------------------------|
[{{ error-function__PairLogit }}](#PairLogit)                 |     +                   |     +                   |
[{{ error-function__PairLogitPairwise }}](#PairLogitPairwise) |     +                   |     +                   |
[{{ error-function__PairAccuracy }}](#PairAccuracy)           |     -                   |     -                   |
[{{ error-function__YetiRank }}](#YetiRank)                   |     +                   |     + (but only Classic mode) |
[{{ error-function__YetiRankPairwise }}](#YetiRankPairwise)   |     +                   |     + (but only Classic mode) |
[{{ error-function__LambdaMart }}](#LambdaMart)               |     +                   |     -                   |
[{{ error-function__StochasticFilter }}](#StochasticFilter)   |     +                   |     -                   |
[{{ error-function__StochasticRank }}](#StochasticRank)       |     +                   |     -                   |
[{{ error-function__QueryCrossEntropy }}](#QueryCrossEntropy) |     +                   |     +                   |
[{{ error-function__QueryRMSE }}](#QueryRMSE)                 |     +                   |     +                   |
[{{ error-function__QuerySoftMax }}](#QuerySoftMax)           |     +                   |     +                   |
[{{ error-function__PFound }}](#PFound)                       |     -                   |     -                   |
[{{ error-function__ndcg }}](#ndcg)                           |     -                   |     -                   |
[{{ error-function__dcg }}](#dcg)                             |     -                   |     -                   |
[{{ error-function__FilteredDCG }}](#PFilteredDCG)            |     -                   |     -                   |
[{{ error-function__QueryAverage }}](#QueryAverage)           |     -                   |     -                   |
[{{ error-function__PrecisionAtK }}](#PrecisionAtK)           |     -                   |     -                   |
[{{ error-function__RecallAtK }}](#RecallAtK)                 |     -                   |     -                   |
[{{ error-function__mapk }}](#mapk)                           |     -                   |     -                   |
[{{ error-function__err }}](#err)                             |     -                   |     -                   |
[{{ error-function__mrr }}](#mrr)                             |     -                   |     -                   |
[{{ error-function--AUC }}](#AUC)                             |     -                   |     -                   |
[{{ error-function--QueryAUC }}](#QueryAUC)                   |     -                   |     -                   |
