The model prediction results will be correct only if the `data` parameter with feature values contains all the features used in the model. Typically, the order of these features must match the order of the corresponding columns that is provided during the training. But if feature names are provided both during the training and when applying the model, they can be matched by names instead of columns order. Feature names can be specified if the `data` parameter has one of the following types:
- [FeaturesData](../../../concepts/python-features-data__desc.md)
- [{{ python-type--pool }}](../../../concepts/python-reference_pool.md)
- [{{ python-type--pandasDataFrame }}](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html) (in this case, feature names are taken from column names)
