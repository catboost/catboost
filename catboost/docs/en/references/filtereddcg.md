# {{ error-function__FilteredDCG }}

{% include [loss-functions-ranking-quality-assessment](../_includes/work_src/reusage-common-phrases/ranking-quality-assessment.md) %}

- [{{ title__loss-functions__calculation-principles }}](#calculation)
- [{{ title__loss-functions__text__user-defined-params }}](#user-defined-parameters)

## {{ title__loss-functions__calculation-principles }} {#calculation}

{% include [loss-functions-function-calculation](../_includes/work_src/reusage-common-phrases/function-calculation.md) %}


1. Filter out all objects with negative predicted relevancies ($a_i$).

1. The {{ error-function__FilteredDCG }} metric is calculated for each group ($group \in groups$) with filtered objects.

   The calculation principle depends on the specified value of the `{{ loss-functions__params__type }}` and `{{ loss-functions__params__denominator }}` parameters:

   | type/denominator|{{ error-function__ndcg__denominator__LogPosition }}| {{ error-function__ndcg__denominator__Position }}|
   |-----------------|-----------------------------------------------------|-------------------------------------------------|
   | **Base** | $FilteredDCG(group) = \sum\limits_{i}\displaystyle\frac{t_{g(i,group)}}{log_{2}(i+1)}$| $FilteredDCG(group) = \sum\limits_{i}\displaystyle\frac{t_{g(i,group)}}{i}$|
   | **Exp**  | $FilteredDCG(group) = \sum\limits_{i}\displaystyle\frac{2^{t_{g(i,group)}} - 1}{log_{2}(i+1)}$| $FilteredDCG(group) = \sum\limits_{i}\displaystyle\frac{2^{t_{g(i,group)}} - 1}{i}$|

   $t_{g(i, group)}$ is the label value for the i-th object in the group after filtering objects with negative predicted relevancies.

1. The aggregated value of the metric for all groups is calculated as follows:
    $FilteredDCG = \frac{\sum\limits_{group \in groups}  FilteredDCG(group)}{|groups|}$


## {{ title__loss-functions__text__user-defined-params }} {#user-defined-parameters}

### type

#### Description

Metric calculation principles.

Possible values:
- {{ error-function__ndcg__type__Base }}
- {{ error-function__ndcg__type__Exp }}

_Default_: {{ error-function__filtereddcg__type__default }}


### denominator

#### Description

Metric denominator type.

Possible values:
- {{ error-function__ndcg__denominator__LogPosition }}
- {{ error-function__ndcg__denominator__Position }}

_Default_: {{ error-function__filtereddcg__denominator__default }}

