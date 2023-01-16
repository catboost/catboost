# Using the overfitting detector

{% include [overfitting-detector-od__purpose](../_includes/work_src/reusage-common-phrases/od__purpose.md) %}

{% include [feature-importance-choose-the-required-implementation-for-more-details](../_includes/work_src/reusage-common-phrases/choose-the-required-implementation-for-more-details.md) %}

{% if audience == "internal" %}

## {{ common-text__title__nirvana_cubes__dl }}

**{{ common-text__title__nirvana_cubes__dl }}**

[CatBoost: Train](../yandex_specific/nirvana-operations/catboost__nirvana__train-catboost.md)

**Enabling guide**

Set the required overfitting detector type in the `od_type` option.

{% endif %}

## {{ python-package }}

The following parameters can be set for the corresponding methods and are used when the model is trained.

**{{ features__table__title__python__method }}**

- [fit](../concepts/python-reference_catboost_fit.md) ([CatBoost](../concepts/python-reference_catboost.md))
- [fit](../concepts/python-reference_catboostclassifier_fit.md) ([CatBoostClassifier](../concepts/python-reference_catboostclassifier.md))
- [fit](../concepts/python-reference_catboostregressor_fit.md) ([CatBoostRegressor](../concepts/python-reference_catboostregressor.md))


**{{ features__table__title__python__parameters }}**

`od_type`

**{{ features__table__title__python__description }}**

The type of the overfitting detector to use.

Possible values:
- {{ fit--od-type-inctodec }}
- {{ fit--od-type-iter }}


**{{ features__table__title__python__method }}**

 `od_pval`

**{{ features__table__title__python__parameters }}**

{% include [reusage-od-pval__short-desc](../_includes/work_src/reusage/od-pval__short-desc.md) %}


{% include [overfitting-detector-od-pval__greater-than-zero](../_includes/work_src/reusage-common-phrases/od-pval__greater-than-zero.md) %}

**{{ features__table__title__python__method }}**

 `od_wait`

**{{ features__table__title__python__parameters }}** The number of iterations to continue the training after the iteration with the optimal metric value.
The purpose of this parameter differs depending on the selected overfitting detector type:
- {{ fit--od-type-inctodec }} — Ignore the overfitting detector when the threshold is reached and continue learning for the specified number of iterations after the iteration with the optimal metric value.
- {{ fit--od-type-iter }} — Consider the model overfitted and stop training after the specified number of iterations since the iteration with the optimal metric value.

## {{ r-package }}

The following parameters can be set for the corresponding methods and are used when the model is trained.

For the [catboost.train](../concepts/r-reference_catboost-train.md) method:

### od_type

The type of the overfitting detector to use.

Possible values:
- {{ fit--od-type-inctodec }}
- {{ fit--od-type-iter }}

### od_pval

{% include [reusage-od-pval__short-desc](../_includes/work_src/reusage/od-pval__short-desc.md) %}


{% include [overfitting-detector-od-pval__greater-than-zero](../_includes/work_src/reusage-common-phrases/od-pval__greater-than-zero.md) %}


### od_wait

 The number of iterations to continue the training after the iteration with the optimal metric value.
The purpose of this parameter differs depending on the selected overfitting detector type:
- {{ fit--od-type-inctodec }} — Ignore the overfitting detector when the threshold is reached and continue learning for the specified number of iterations after the iteration with the optimal metric value.
- {{ fit--od-type-iter }} — Consider the model overfitted and stop training after the specified number of iterations since the iteration with the optimal metric value.


## {{ title__implementation__cli }}

The following command keys can be specified for the corresponding commands and are used when the model is trained.**

For the [catboost fit](../references/training-parameters/index.md) command:

### --od-type

The type of the overfitting detector to use.

Possible values:
- {{ fit--od-type-inctodec }}
- {{ fit--od-type-iter }}

### --od-pval

{% include [reusage-od-pval__short-desc](../_includes/work_src/reusage/od-pval__short-desc.md) %}


{% include [overfitting-detector-od-pval__greater-than-zero](../_includes/work_src/reusage-common-phrases/od-pval__greater-than-zero.md) %}


### --od-wait

 The number of iterations to continue the training after the iteration with the optimal metric value.
The purpose of this parameter differs depending on the selected overfitting detector type:
- {{ fit--od-type-inctodec }} — Ignore the overfitting detector when the threshold is reached and continue learning for the specified number of iterations after the iteration with the optimal metric value.
- {{ fit--od-type-iter }} — Consider the model overfitted and stop training after the specified number of iterations since the iteration with the optimal metric value.

