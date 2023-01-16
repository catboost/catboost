# Staged prediction

{{ product }} allows to apply a trained model and calculate the results for each i-th tree of the model taking into consideration only the trees in the range `[0; i)`.

{% include [feature-importance-choose-the-required-implementation-for-more-details](../_includes/work_src/reusage-common-phrases/choose-the-required-implementation-for-more-details.md) %}


## {{ python-package }}

### Classes

#### [CatBoost](../concepts/python-reference_catboost.md)

**{{ features__table__title__python__class-applicability }}**

{% include [catboost-purpose](../_includes/work_src/reusage-python/purpose.md) %}

**{{ features__table__title__python__method }}**

 [staged_predict](../concepts/python-reference_catboost_staged_predict.md)

**{{ features__table__title__python__description }}**

{% include [sections-with-methods-desc-staged_predict--purpose](../_includes/work_src/reusage/staged_predict--purpose.md) %}

#### [CatBoostRegressor](../concepts/python-reference_catboostregressor.md)

**{{ features__table__title__python__class-applicability }}**

Training and applying models for the regression problems. When using the applying methods only the predicted class is returned. Provides compatibility with the scikit-learn tools.

**{{ features__table__title__python__method }}**

 [staged_predict](../concepts/python-reference_catboostregressor_staged_predict.md)

**{{ features__table__title__python__description }}**

{% include [sections-with-methods-desc-staged_predict--purpose](../_includes/work_src/reusage/staged_predict--purpose.md) %}


#### [CatBoostClassifier](../concepts/python-reference_catboostclassifier.md)

**{{ features__table__title__python__class-applicability }}**

Training and applying models for the classification problems. Provides compatibility with the scikit-learn tools.

**{{ features__table__title__python__method }}s**

 [staged_predict](../concepts/python-reference_catboostclassifier_staged_predict.md)

**{{ features__table__title__python__description }}**

{% include [sections-with-methods-desc-staged_predict--purpose](../_includes/work_src/reusage/staged_predict--purpose.md) %}

[staged_predict_proba](../concepts/python-reference_catboostclassifier_staged_predict_proba.md)

**{{ features__table__title__python__class-applicability }}**

The same as staged_predict with the difference that the results are probabilities that the object belongs to the positive class.

## {{ r-package }}

For the [catboost.staged_predict](../concepts/r-reference_catboost-staged_predict.md) method:

**{{ features__table__title__r__purpose }}**

{% include [reusage-r-staged_predict__purpose](../_includes/work_src/reusage-r/staged_predict__purpose.md) %}

## {{ title__implementation__cli }}

For the [catboost calc](../concepts/cli-reference_calc-model.md) command:

**{{ features__table__title__cli__purpose }}**

{% include [reusage-cli-purpose__apply-the-model](../_includes/work_src/reusage-cli/purpose__apply-the-model.md) %}

**{{ features__table__title__cli__keys }}**

`--eval-period`

**{{ features__table__title__cli__key-description }}**

To reduce the number of trees to use when the model is applied or the metrics are calculated, setthe step of the trees to use to `eval-period`.

{% include [eval-start-end-cli__eval__period__desc](../_includes/work_src/reusage-common-phrases/cli__eval__period__desc.md) %}

