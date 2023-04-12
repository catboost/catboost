# CatBoostRegressor

```python
class CatBoostRegressor(iterations=None,
                        learning_rate=None,
                        depth=None,
                        l2_leaf_reg=None,
                        model_size_reg=None,
                        rsm=None,
                        loss_function='RMSE',
                        border_count=None,
                        feature_border_type=None,
                        per_float_feature_quantization=None,
                        input_borders=None,
                        output_borders=None,
                        fold_permutation_block=None,
                        od_pval=None,
                        od_wait=None,
                        od_type=None,
                        nan_mode=None,
                        counter_calc_method=None,
                        leaf_estimation_iterations=None,
                        leaf_estimation_method=None,
                        thread_count=None,
                        random_seed=None,
                        use_best_model=None,
                        best_model_min_trees=None,
                        verbose=None,
                        silent=None,
                        logging_level=None,
                        metric_period=None,
                        ctr_leaf_count_limit=None,
                        store_all_simple_ctr=None,
                        max_ctr_complexity=None,
                        has_time=None,
                        allow_const_label=None,
                        one_hot_max_size=None,
                        random_strength=None,
                        name=None,
                        ignored_features=None,
                        train_dir=None,
                        custom_metric=None,
                        eval_metric=None,
                        bagging_temperature=None,
                        save_snapshot=None,
                        snapshot_file=None,
                        snapshot_interval=None,
                        fold_len_multiplier=None,
                        used_ram_limit=None,
                        gpu_ram_part=None,
                        pinned_memory_size=None,
                        allow_writing_files=None,
                        final_ctr_computation_mode=None,
                        approx_on_full_history=None,
                        boosting_type=None,
                        simple_ctr=None,
                        combinations_ctr=None,
                        per_feature_ctr=None,
                        ctr_target_border_count=None,
                        task_type=None,
                        device_config=None,
                        devices=None,
                        bootstrap_type=None,
                        subsample=None,
                        sampling_unit=None,
                        dev_score_calc_obj_block_size=None,
                        max_depth=None,
                        n_estimators=None,
                        num_boost_round=None,
                        num_trees=None,
                        colsample_bylevel=None,
                        random_state=None,
                        reg_lambda=None,
                        objective=None,
                        eta=None,
                        max_bin=None,
                        gpu_cat_features_storage=None,
                        data_partition=None,
                        metadata=None,
                        early_stopping_rounds=None,
                        cat_features=None,
                        grow_policy=None,
                        min_data_in_leaf=None,
                        min_child_samples=None,
                        max_leaves=None,
                        num_leaves=None,
                        score_function=None,
                        leaf_estimation_backtracking=None,
                        ctr_history_unit=None,
                        monotone_constraints=None,
                        feature_weights=None,
                        penalties_coefficient=None,
                        first_feature_use_penalties=None,
                        model_shrink_rate=None,
                        model_shrink_mode=None,
                        langevin=None,
                        diffusion_temperature=None,
                        posterior_sampling=None,
                        boost_from_average=None,
                        fixed_binary_splits=None)
```

## {{ dl--purpose }} {#purpose}

{% include [catboost-regressor-purpose](../_includes/work_src/reusage-python/purpose.md) %}


## {{ dl--parameters }} {#parameters}

{% include [python__parameters__metadata](../_includes/work_src/reusage-python/python__parameters_metadata.md) %}

{% include [python__parameters__cat_features](../_includes/work_src/reusage-python/python__parameters_cat_features.md) %}


{% include [sections-with-methods-desc-see-training-params](../_includes/work_src/reusage/see-training-params.md) %}


{% include [precedence-python--regressor--precedence-note](../_includes/work_src/reusage/python--regressor--precedence-note.md) %}

## {{ dl--attributes }} {#attributes}

### [tree_count_](python-reference_catboostregressor_attributes.md)


{% include [tree_count_-tree_count__desc](../_includes/work_src/reusage-attributes/tree_count__desc.md) %}


### [feature_importances_](python-reference_catboostregressor_attributes.md)



{% include [feature_importances_-feature_importances__desc](../_includes/work_src/reusage-attributes/feature_importances__desc.md) %}


### [random_seed_](python-reference_catboostregressor_attributes.md)


{% include [random_seed_-random_seed__desc](../_includes/work_src/reusage-attributes/random_seed__desc.md) %}


### [learning_rate_](python-reference_catboostregressor_attributes.md)


{% include [learning_rate_-learning_rate__desc](../_includes/work_src/reusage-attributes/learning_rate__desc.md) %}


### [feature_names_](python-reference_catboostregressor_attributes.md)



{% include [feature_names_-feature_names__desc](../_includes/work_src/reusage-attributes/feature_names__desc.md) %}

### [evals_result_](python-reference_catboostregressor_attributes.md)


{% include [sections-with-methods-desc-python__get-evals-result__desc](../_includes/work_src/reusage/python__get-evals-result__desc.md) %}

### [best_score_](python-reference_catboostregressor_attributes.md)


{% include [sections-with-methods-desc-python__method__get_best_score__desc](../_includes/work_src/reusage/python__method__get_best_score__desc.md) %}

### [best_iteration_](python-reference_catboostregressor_attributes.md)



{% include [sections-with-methods-desc-python__method__get_best_iteration__desc](../_includes/work_src/reusage/python__method__get_best_iteration__desc.md) %}


## {{ dl--methods }} {#methods}

### [fit](python-reference_catboostregressor_fit.md)



{% include [sections-with-methods-desc-fit--purpose-desc](../_includes/work_src/reusage/fit--purpose-desc.md) %}

### [predict](python-reference_catboostregressor_predict.md)


{% include [sections-with-methods-desc-predict--purpose](../_includes/work_src/reusage/predict--purpose.md) %}

### [calc_leaf_indexes](python-reference_catboost_calc_leaf_indexes.md)



{% include [python-reference_catboost_calc_leaf_indexes-calc_leaf_indexes__desc](../_includes/concepts/python-reference_catboost_calc_leaf_indexes/calc_leaf_indexes__desc.md) %}

### [calc_feature_statistics](python-reference_catboostregressor_calc_feature_statistics.md)



{% include [get_feature_statistics-get_feature_statistics__desc__div](../_includes/work_src/reusage-python/get_feature_statistics__desc__div.md) %}

### [copy](python-reference_catboostregressor_copy.md)



{% include [sections-with-methods-desc-copy--purpose](../_includes/work_src/reusage/copy--purpose.md) %}

### [compare](python-reference_catboostregressor_modelcompare.md)



{% include [model-compare-compare__purpose](../_includes/work_src/reusage-python/compare__purpose.md) %}

### [eval_metrics](python-reference_catboostregressor_eval-metrics.md)



{% include [sections-with-methods-desc-python__eval-metrics__purpose](../_includes/work_src/reusage/python__eval-metrics__purpose.md) %}

### [get_all_params](python-reference_catboostregressor_get_all_params.md)



{% include [get_all_params-python__get_all_params__desc__p](../_includes/work_src/reusage-python/python__get_all_params__desc__p.md) %}

### [get_best_iteration](python-reference_catboostregressor_get_best_iteration.md)



{% include [sections-with-methods-desc-python__method__get_best_iteration__desc](../_includes/work_src/reusage/python__method__get_best_iteration__desc.md) %}


### [get_best_score](python-reference_catboostregressor_get_best_score.md)



{% include [sections-with-methods-desc-python__method__get_best_score__desc](../_includes/work_src/reusage/python__method__get_best_score__desc.md) %}

### [get_borders](python-reference_catboostregressor_get_borders.md)



{% include [get_borders-get_borders__desc__div](../_includes/work_src/reusage-python/get_borders__desc__div.md) %}


### [get_evals_result](python-reference_catboostregressor_get_evals_result.md)



{% include [sections-with-methods-desc-python__get-evals-result__desc](../_includes/work_src/reusage/python__get-evals-result__desc.md) %}

### [{{ function-name__get-feature-importance }}](python-reference_catboostregressor_get_feature_importance.md)



{% include [sections-with-methods-desc-feature_importances--purpose](../_includes/work_src/reusage/feature_importances--purpose.md) %}


### [get_metadata](python-reference_catboostregressor_metadata.md)

Return a proxy object with metadata from the model's internal key-value string storage.

### [get_object_importance](python-reference_catboostregressor_get_object_importance.md)


{% include [sections-with-methods-desc-python__get_object_importance__desc](../_includes/work_src/reusage/python__get_object_importance__desc.md) %}


### [get_param](python-reference_catboostregressor_get_param.md)



{% include [sections-with-methods-desc-get_param--purpose](../_includes/work_src/reusage/get_param--purpose.md) %}



### [get_params](python-reference_catboostregressor_get_params.md)



{% include [sections-with-methods-desc-get_params--purpose](../_includes/work_src/reusage/get_params--purpose.md) %}


### [get_scale_and_bias](python-reference_catboostregressor_get_scale_and_bias.md)


{% include [get_scale_and_bias-get_scale_and_bias__desc](../_includes/work_src/reusage-python/get_scale_and_bias__desc.md) %}



### [get_test_eval](python-reference_catboostregressor_get_test_eval.md)



{% include [sections-with-methods-desc-get_test_val--desc](../_includes/work_src/reusage/get_test_val--desc.md) %}



### [grid_search](python-reference_catboost_grid_search.md)



{% include [grid_search-python__grid_search--desc](../_includes/work_src/reusage-python/python__grid_search--desc.md) %}


### [is_fitted](python-reference_catboostregressor_is_fitted.md)


{% include [sections-with-methods-desc-is_fitted--purpose](../_includes/work_src/reusage/is_fitted--purpose.md) %}


### [load_model](python-reference_catboostregressor_load_model.md)



{% include [sections-with-methods-desc-load_model--purpose](../_includes/work_src/reusage/load_model--purpose.md) %}



### [plot_predictions](python-reference_catboostregressor_plot_predictions.md)


{% include [plot_predictions-plot_predictions__desc__short](../_includes/work_src/reusage-python/plot_predictions__desc__short.md) %}


### [plot_tree](python-reference_catboostregressor_plot_tree.md)


{% include [plot_tree-python__plot-tree_desc__div](../_includes/work_src/reusage-python/python__plot-tree_desc__div.md) %}


### [randomized_search](python-reference_catboost_randomized_search.md)



{% include [randomized_search-python__randomized_search--desc](../_includes/work_src/reusage-python/python__randomized_search--desc.md) %}


### [save_borders](python-reference_catboostregressor_save_borders.md)



{% include [save_borders-save_model__div_desc](../_includes/work_src/reusage-python/save_model__div_desc.md) %}


### [save_model](python-reference_catboostregressor_save_model.md)



{% include [sections-with-methods-desc-save_model--purpose](../_includes/work_src/reusage/save_model--purpose.md) %}


### [score](python-reference_catboostregressor_score.md)


{% include [catboost-regressor-score-scor__purpose](../_includes/work_src/reusage-python/scor__purpose.md) %}

### [select_features](python-reference_catboostregressor_select_features.md)


Select the best features from the dataset using the [Recursive Feature Elimination](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) algorithm.


### [set_feature_names](python-reference_catboostregressor_set_feature_names.md)


{% include [non_pool__set_feature_names-non_pool__set_feature_names__div](../_includes/work_src/reusage-python/non_pool__set_feature_names__div.md) %}


### [set_params](python-reference_catboostregressor_set_params.md)



{% include [sections-with-methods-desc-set_params--purpose](../_includes/work_src/reusage/set_params--purpose.md) %}


### [set_scale_and_bias](python-reference_catboostregressor_set_scale_and_bias.md)



{% include [set_scale_and_bias-set_scale_and_bias__desc](../_includes/work_src/reusage-python/set_scale_and_bias__desc.md) %}


### [shrink](python-reference_catboostregressor_shrink.md)



{% include [sections-with-methods-desc-shrink__purpose](../_includes/work_src/reusage/shrink__purpose.md) %}


### [staged_predict](python-reference_catboostregressor_staged_predict.md)



{% include [sections-with-methods-desc-staged_predict--purpose](../_includes/work_src/reusage/staged_predict--purpose.md) %}
