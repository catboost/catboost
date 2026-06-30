# {{ python-type__EFStrType }}

Enumeration with the list of supported [feature importance](fstr.md).

The following values are supported:
- FeatureImportance: Equal to [{{ title__regular-feature-importance-PredictionValuesChange }}](../concepts/fstr.md#regular-feature-importance) for non-ranking metrics and [{{ title__regular-feature-importance-LossFunctionChange }}](../concepts/fstr.md#regular-feature-importances__lossfunctionchange) for ranking metrics (the value is determined automatically).

- [{{ title__ShapValues }}](../concepts/shap-values.md): A vector $v$ with contributions of each feature to the prediction for every input object and the expected value of the model prediction for the object (average prediction given no knowledge about the object).
- [{{ title__Interaction }}](../concepts/feature-interaction.md#feature-interaction-strength): The value of the feature interaction strength for each pair of features.

- {{ title__predictiondiff }}: A vector with contributions of each feature to the {{ prediction-type--RawFormulaVal }} difference for each pair of objects.

