# Training

{{ product }} provides a variety of modes for training a model.

{% include [feature-importance-choose-the-required-implementation-for-more-details](../_includes/work_src/reusage-common-phrases/choose-the-required-implementation-for-more-details.md) %}


{% if audience == "internal" %}

## {{ common-text__title__nirvana_cubes }}

- [CatBoost: Train](../yandex_specific/nirvana-operations/catboost__nirvana__train-catboost.md)
- [Train {{ product }} with {{ product__matrixnet }} interface](../yandex_specific/nirvana-operations/catboost__nirvana__train-catboost-with-matrixnet-interface.md)
- [Train Formula on FML](../yandex_specific/nirvana-operations/catboost__nirvana__train-formula-on-fml.md)

{% endif %}

## {{ python-package }}

### Classes

#### [CatBoost](../concepts/python-reference_catboost.md)

**{{ features__table__title__python__class-applicability }}**


{% include [catboost-purpose](../_includes/work_src/reusage-python/purpose.md) %}

**{{ features__table__title__python__method }}**

 [fit](../concepts/python-reference_catboost_fit.md)

#### [CatBoostClassifier](../concepts/python-reference_catboostclassifier.md)

**{{ features__table__title__python__class-applicability }}**


{% include [catboost-classifier-purpose](../_includes/work_src/reusage-python/purpose.md) %}


**{{ features__table__title__python__method }}**

 [fit](../concepts/python-reference_catboostclassifier_fit.md)

#### [CatBoostRegressor](../concepts/python-reference_catboostregressor.md)

**{{ features__table__title__python__class-applicability }}**


{% include [catboost-regressor-purpose](../_includes/work_src/reusage-python/purpose.md) %}

**{{ features__table__title__python__method }}**

 [fit](../concepts/python-reference_catboostregressor_fit.md)

## {{ r-package }}

**{{ features__table__title__r__method }}**

[catboost.train](../concepts/r-reference_catboost-train.md)

**{{ features__table__title__r__purpose }}**

{% include [reusage-r-train__purpose](../_includes/work_src/reusage-r/train__purpose.md) %}

## {{ title__implementation__cli }}

**{{ features__table__title__cli__command }}**

[catboost fit](../references/training-parameters/index.md)

**{{ features__table__title__cli__purpose }}**

Train the model.


