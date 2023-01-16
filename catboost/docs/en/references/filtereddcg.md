# {{ error-function__FilteredDCG }}

{% include [loss-functions-ranking-quality-assessment](../_includes/work_src/reusage-common-phrases/ranking-quality-assessment.md) %}

- [{{ title__loss-functions__calculation-principles }}](#calculation)
- [{{ title__loss-functions__text__user-defined-params }}](#user-defined-parameters)

## {{ title__loss-functions__calculation-principles }} {#calculation}

The possible label values ($t_{i}$) are limited to the following range: $[0; +\infty)$.

The calculation principle depends on the specified value of the `` and `` parameters:
{{ error-function__filtereddcg__denominator__LogPosition }} {{ error-function__filtereddcg__denominator__Position }}

**{{ loss-functions__params__denominator }}:** $FilteredDCG = \sum\limits_{approx_{i} \geq 0}\displaystyle\frac{t_{g(i)}}{log_{2}(i+1)}$

**undefined:** $FilteredDCG = \sum\limits_{approx_{i} \geq 0}\displaystyle\frac{t_{g(i)}}{i}$

**{{ loss-functions__params__denominator }}:** $FilteredDCG = \sum\limits_{approx_{i} \geq 0}\displaystyle\frac{2^{t_{g(i)}} – 1}{log_{2}(i+1)}$

**undefined:** $FilteredDCG = \sum\limits_{approx_{i} \geq 0}\displaystyle\frac{2^{t_{g(i)}} – 1}{i}$

$t_{g(i)}$ is the label value for the i-th object.


## {{ title__loss-functions__text__user-defined-params }} {#user-defined-parameters}
**Parameter:** ``

#### Description


Metric calculation principles.

Possible values:
- {{ error-function__ndcg__type__Base }}
- {{ error-function__ndcg__type__Exp }}


**Parameter:** ``

#### Description


Metric denominator type.

Possible values:
- {{ error-function__ndcg__denominator__LogPosition }}
- {{ error-function__ndcg__denominator__Position }}


