# CatBoost

```python
class CatBoost(params=None)
```

## {{ dl--purpose }} {#purpose}

{% include [catboost-purpose](../_includes/work_src/reusage-python/purpose.md) %}


## {{ dl--parameters }} {#parameters}

### params

#### Description

The list of [parameters](../references/training-parameters/index.md) to start training with.

If omitted, default values are used.

{% include [precedence-python--precedence](../_includes/work_src/reusage/python--precedence.md) %}

**Possible types:** `{{ python-type--dict }}`

**Default value**

 `None`


## {{ dl--attributes }} {#attributes}

### [tree_count_](python-reference_catboost_attributes.md#tree_count)

{% include [tree_count_-tree_count__desc](../_includes/work_src/reusage-attributes/tree_count__desc.md) %}

###  [feature_importances_](python-reference_catboost_attributes.md#feature_importances)

{% include [feature_importances_-feature_importances__desc](../_includes/work_src/reusage-attributes/feature_importances__desc.md) %}

### [random_seed_](python-reference_catboost_attributes.md#random_seed)

{% include [random_seed_-random_seed__desc](../_includes/work_src/reusage-attributes/random_seed__desc.md) %}

### [learning_rate_](python-reference_catboost_attributes.md#learning_rate)

{% include [learning_rate_-learning_rate__desc](../_includes/work_src/reusage-attributes/learning_rate__desc.md) %}

### [feature_names_](python-reference_catboost_attributes.md#feature_names)

{% include [feature_names_-feature_names__desc](../_includes/work_src/reusage-attributes/feature_names__desc.md) %}

### [evals_result_](python-reference_catboost_attributes.md#evals_result)

{% include [sections-with-methods-desc-python__get-evals-result__desc](../_includes/work_src/reusage/python__get-evals-result__desc.md) %}

### [best_score_](python-reference_catboost_attributes.md#best_score#best_score)

{% include [sections-with-methods-desc-python__method__get_best_score__desc](../_includes/work_src/reusage/python__method__get_best_score__desc.md) %}

### [best_iteration_](python-reference_catboost_attributes.md#best_iteration)

{% include [sections-with-methods-desc-python__method__get_best_iteration__desc](../_includes/work_src/reusage/python__method__get_best_iteration__desc.md) %}

### [classes_](python-reference_catboost_attributes.md#classes)

{% include [classes-attributes__classes__desc-div](../_includes/work_src/reusage-attributes/attributes__classes__desc-div.md) %}

## {{ dl--methods }} {#methods}

### [fit](python-reference_catboost_fit.md)



{% include [sections-with-methods-desc-fit--purpose-desc](../_includes/work_src/reusage/fit--purpose-desc.md) %}



### [predict](python-reference_catboost_predict.md)



{% include [sections-with-methods-desc-predict--purpose](../_includes/work_src/reusage/predict--purpose.md) %}



### [calc_feature_statistics](python-reference_catboost_calc_feature_statistics.md)



{% include [get_feature_statistics-get_feature_statistics__desc__div](../_includes/work_src/reusage-python/get_feature_statistics__desc__div.md) %}



### [calc_leaf_indexes](python-reference_catboost_calc_leaf_indexes.md)



{% include [python-reference_catboost_calc_leaf_indexes-calc_leaf_indexes__desc](../_includes/concepts/python-reference_catboost_calc_leaf_indexes/calc_leaf_indexes__desc.md) %}



### [compare](python-reference_catboost_modelcompare.md)



{% include [model-compare-compare__purpose](../_includes/work_src/reusage-python/compare__purpose.md) %}



### [copy](python-reference_catboost_copy.md)



{% include [sections-with-methods-desc-copy--purpose](../_includes/work_src/reusage/copy--purpose.md) %}



### [eval_metrics](python-reference_catboost_eval-metrics.md)



{% include [sections-with-methods-desc-python__eval-metrics__purpose](../_includes/work_src/reusage/python__eval-metrics__purpose.md) %}



### [get_all_params](python-reference_catboost_get_all_params.md)



{% include [get_all_params-python__get_all_params__desc__p](../_includes/work_src/reusage-python/python__get_all_params__desc__p.md) %}



### [get_best_iteration](python-reference_catboost_get_best_iteration.md)



{% include [sections-with-methods-desc-python__method__get_best_iteration__desc](../_includes/work_src/reusage/python__method__get_best_iteration__desc.md) %}



### [get_best_score](python-reference_catboost_get_best_score.md)



{% include [sections-with-methods-desc-python__method__get_best_score__desc](../_includes/work_src/reusage/python__method__get_best_score__desc.md) %}



### [get_borders](python-reference_catboost_get_borders.md)



{% include [get_borders-get_borders__desc__div](../_includes/work_src/reusage-python/get_borders__desc__div.md) %}



### [get_evals_result](python-reference_catboost_get_evals_result.md)



{% include [sections-with-methods-desc-python__get-evals-result__desc](../_includes/work_src/reusage/python__get-evals-result__desc.md) %}



### [{{ function-name__get-feature-importance }}](python-reference_catboost_get_feature_importance.md)



{% include [sections-with-methods-desc-feature_importances--purpose](../_includes/work_src/reusage/feature_importances--purpose.md) %}



### [get_metadata](python-reference_catboost_metadata.md)

 Return a proxy object with metadata from the model's internal key-value string storage.

### [get_object_importance](python-reference_catboost_get_object_importance.md)



{% include [sections-with-methods-desc-python__get_object_importance__desc](../_includes/work_src/reusage/python__get_object_importance__desc.md) %}



### [get_param](python-reference_catboost_get_param.md)



{% include [sections-with-methods-desc-get_param--purpose](../_includes/work_src/reusage/get_param--purpose.md) %}



### [get_params](python-reference_catboost_get_params.md)



{% include [sections-with-methods-desc-get_params--purpose](../_includes/work_src/reusage/get_params--purpose.md) %}



### [get_scale_and_bias](python-reference_catboost_get_scale_and_bias.md)



{% include [get_scale_and_bias-get_scale_and_bias__desc](../_includes/work_src/reusage-python/get_scale_and_bias__desc.md) %}



### [get_test_eval](python-reference_catboost_get_test_eval.md)



{% include [sections-with-methods-desc-get_test_val--desc](../_includes/work_src/reusage/get_test_val--desc.md) %}



### [grid_search](python-reference_catboost_grid_search.md)



{% include [grid_search-python__grid_search--desc](../_includes/work_src/reusage-python/python__grid_search--desc.md) %}



### [load_model](python-reference_catboost_load_model.md)



{% include [sections-with-methods-desc-load_model--purpose](../_includes/work_src/reusage/load_model--purpose.md) %}



### [plot_predictions](python-reference_catboost_plot_predictions.md)



{% include [plot_predictions-plot_predictions__desc__short](../_includes/work_src/reusage-python/plot_predictions__desc__short.md) %}



### [plot_tree](python-reference_catboost_plot_tree.md)



{% include [plot_tree-python__plot-tree_desc__div](../_includes/work_src/reusage-python/python__plot-tree_desc__div.md) %}



### [randomized_search](python-reference_catboost_randomized_search.md)



{% include [randomized_search-python__randomized_search--desc](../_includes/work_src/reusage-python/python__randomized_search--desc.md) %}



### [save_borders](python-reference_catboost_save_borders.md)



{% include [save_borders-save_model__div_desc](../_includes/work_src/reusage-python/save_model__div_desc.md) %}



### [select_features](python-reference_catboost_select_features.md)



Select the best features from the dataset using the [Recursive Feature Elimination](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) algorithm.


### [set_feature_names](python-reference_catboost_set_feature_names.md)



{% include [non_pool__set_feature_names-non_pool__set_feature_names__div](../_includes/work_src/reusage-python/non_pool__set_feature_names__div.md) %}



### [set_params](python-reference_catboost_set_params.md)



{% include [sections-with-methods-desc-set_params--purpose](../_includes/work_src/reusage/set_params--purpose.md) %}



### [set_scale_and_bias](python-reference_catboost_set_scale_and_bias.md)



{% include [set_scale_and_bias-set_scale_and_bias__desc](../_includes/work_src/reusage-python/set_scale_and_bias__desc.md) %}



### [shrink](python-reference_catboost_shrink.md)



{% include [sections-with-methods-desc-shrink__purpose](../_includes/work_src/reusage/shrink__purpose.md) %}



### [staged_predict](python-reference_catboost_staged_predict.md)



{% include [sections-with-methods-desc-staged_predict--purpose](../_includes/work_src/reusage/staged_predict--purpose.md) %}
