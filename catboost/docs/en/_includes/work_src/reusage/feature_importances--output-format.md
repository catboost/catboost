
Depends on the selected feature strength calculation method:
- {{ title__regular-feature-importance-PredictionValuesChange }}, {{ title__regular-feature-importance-LossFunctionChange }} or {{ cli__fstr-type__default }} with the `prettified` parameter set to <q>False</q>: a list of length `[n_features]` with float feature importances values for each feature
- {{ title__regular-feature-importance-PredictionValuesChange }} or {{ title__regular-feature-importance-LossFunctionChange }} with the `prettified` parameter set to <q>True</q>: a list of length `[n_features]` with `(feature_id (string), feature_importance (float))` pairs, sorted by feature importance values in descending order
- {{ title__ShapValues }}: np.array of shape `(n_objects, n_features + 1)` with float {{ title__ShapValues }} for each `(object, feature)`
- {{ title__Interaction }}: list of length [ n_features] of three element lists of `(first_feature_index, second_feature_index, interaction_score (float))`
