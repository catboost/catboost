### cat_features

#### Description

A one-dimensional array of categorical columns indices (specified as integers) or names (specified as strings).

This array can contain both indices and names for different elements.

If any features in the `cat_features` parameter are specified as names instead of indices, feature names must be provided for the training dataset. Therefore, the type of the `X` parameter in the future calls of the `fit` function must be either [catboost.Pool](../../../concepts/python-reference_pool.md) with defined feature names data or pandas.DataFrame with defined column names.

{% note info %}

- If this parameter is not None and the training dataset passed as the value of the X parameter to the fit function of this class has the [catboost.Pool](../../../concepts/python-reference_pool.md) type, CatBoost checks the equivalence of the categorical features indices specification in this object and the one in the [catboost.Pool](../../../concepts/python-reference_pool.md) object.

- If this parameter is not None, passing objects of the [catboost.FeaturesData](../../../concepts/python-features-data__desc.md) type as the X parameter to the fit function of this class is prohibited.

{% endnote %}

{% include [default-value-none](../../concepts/default-value-none.md) %}
