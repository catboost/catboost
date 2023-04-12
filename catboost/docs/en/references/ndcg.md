# {{ error-function__ndcg }}

{% include [loss-functions-ranking-quality-assessment](../_includes/work_src/reusage-common-phrases/ranking-quality-assessment.md) %}

- [{{ title__loss-functions__calculation-principles }}](#calculation)
- [{{ title__loss-functions__text__user-defined-params }}](#user-defined-parameters)

## {{ title__loss-functions__calculation-principles }} {#calculation}

{% include [loss-functions-function-calculation](../_includes/work_src/reusage-common-phrases/function-calculation.md) %}


1. {% include [loss-functions-ascending_a_i](../_includes/work_src/reusage-common-phrases/ascending_a_i.md) %}

1. The {{ error-function__dcg }} metric is calculated for each group ($group \in groups$) with sorted objects (see step [1](#ndcg__calc-principles__sort-predicted-relevancies)).

    The calculation principle depends on the specified value of the `{{ loss-functions__params__type }}` and `{{ loss-functions__params__denominator }}` parameters:

   | type/denominator|{{ error-function__ndcg__denominator__LogPosition }}| {{ error-function__ndcg__denominator__Position }}|
   |-----------------|-----------------------------------------------------|-------------------------------------------------|
   | **Base** | $DCG(group,top) = \sum\limits_{i=1}^{top}\displaystyle\frac{t_{g(i,group)}}{log_{2}(i+1)}$| $DCG(group,top) = \sum\limits_{i=1}^{top}\displaystyle\frac{t_{g(i,group)}}{i}$|
   | **Exp**  | $DCG(group,top) = \sum\limits_{i=1}^{top}\displaystyle\frac{2^{t_{g(i,group)}} - 1}{log_{2}(i+1)}$| $DCG(group,top) = \sum\limits_{i=1}^{top}\displaystyle\frac{2^{t_{g(i,group)}} - 1}{i}$|

    $t_{g(i,group)}$ is the label value for the i-th object in the group.

1. The objects in each group are sorted in descending order of target relevancies ($t_{i}$).

1. The {{ error-function__idcg }} metric is calculated for each group ($group \in groups$) with sorted objects (see step [3](#ndcg__calc-principles__sort-target-relevancies)).

    The calculation principle depends on the specified value of the `{{ loss-functions__params__type }}` and `{{ loss-functions__params__denominator }}` parameters:

    | type/denominator|{{ error-function__ndcg__denominator__LogPosition }}| {{ error-function__ndcg__denominator__Position }}|
    |-----------------|-----------------------------------------------------|-------------------------------------------------|
    | **Base** | $IDCG(group,top) = \sum\limits_{i=1}^{top}\displaystyle\frac{t_{g(i,group)}}{log_{2}(i+1)}$| $IDCG(group,top) = \sum\limits_{i=1}^{top}\displaystyle\frac{t_{g(i,group)}}{i}$|
    | **Exp**  | $IDCG(group,top) = \sum\limits_{i=1}^{top}\displaystyle\frac{2^{t_{g(i,group)}} - 1}{log_{2}(i+1)}$ | $IDCG(group,top) = \sum\limits_{i=1}^{top}\displaystyle\frac{2^{t_{g(i,group)}} - 1}{i}$|

1. The {{ error-function__ndcg }} metric is calculated for each group:
    $nDCG(group,top) = \displaystyle\frac{DCG}{iDCG}$
1. The aggregated value of the metric for all groups is calculated as follows:
    $nDCG(top) = \frac{\sum\limits_{group \in groups}  nDCG(group, top) * w_{group}}{\sum\limits_{group \in groups}  w_{group}}$


## {{ title__loss-functions__text__user-defined-params }} {#user-defined-parameters}

### top

#### Description

{% include [reusage-loss-functions-top__desc](../_includes/work_src/reusage-loss-functions/top__desc.md) %}

_Default_: {{ loss-functions__params__top__default }}


###  use_weights

#### Description

{% include [reusage-loss-functions-use-weights__desc__without__note](../_includes/work_src/reusage-loss-functions/use-weights__desc__without__note.md) %}

_Default_: {{ loss-functions__use_weights__default }}

### type

#### Description

Metric calculation principles.

Possible values:
- {{ error-function__ndcg__type__Base }}
- {{ error-function__ndcg__type__Exp }}

_Default_: {{ error-function__ndcg__type__default }}


### denominator

#### Description

Metric denominator type.

Possible values:
- {{ error-function__ndcg__denominator__LogPosition }}
- {{ error-function__ndcg__denominator__Position }}

_Default_: {{ error-function__ndcg__denominator__default }}
