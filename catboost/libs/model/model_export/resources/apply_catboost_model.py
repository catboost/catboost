def hash_uint64(string):
    return cat_features_hashes.get(str(string), 0x7fFFffFF)


### Applicator for the CatBoost model
def apply_catboost_model_multi(float_features, cat_features=[], ntree_start=0, ntree_end=catboost_model.tree_count):
    """
    Applies the model built by CatBoost.

    Parameters
    ----------

    float_features : list of float features

    cat_features : list of categorical features
        You need to pass float and categorical features separately in the same order they appeared in train dataset.
        For example if you had features f1,f2,f3,f4, where f2 and f4 were considered categorical, you need to pass here float_features=f1,f3, cat_features=f2,f4


    Returns
    -------
    predictions : list of formula values for the model and the features

    """
    if ntree_end == 0:
        ntree_end = catboost_model.tree_count
    else:
        ntree_end = min(ntree_end, catboost_model.tree_count)

    model = catboost_model

    assert len(float_features) >= model.float_feature_count
    assert len(cat_features) >= model.cat_feature_count

    # Binarise features
    binary_features = [0] * model.binary_feature_count
    binary_feature_index = 0

    for i in range(len(model.float_feature_borders)):
        if len(model.float_feature_borders[i]) > 0:
            for border in model.float_feature_borders[i]:
                binary_features[binary_feature_index] += 1 if (float_features[model.float_features_index[i]] > border) else 0
            binary_feature_index += 1
    transposed_hash = [0] * model.cat_feature_count
    for i in range(model.cat_feature_count):
        transposed_hash[i] = hash_uint64(cat_features[i])

    if len(model.one_hot_cat_feature_index) > 0:
        cat_feature_packed_indexes = {}
        for i in range(model.cat_feature_count):
            cat_feature_packed_indexes[model.cat_features_index[i]] = i
        for i in range(len(model.one_hot_cat_feature_index)):
            cat_idx = cat_feature_packed_indexes[model.one_hot_cat_feature_index[i]]
            hash = transposed_hash[cat_idx]
            if len(model.one_hot_hash_values[i]) > 0:
                for border_idx in range(len(model.one_hot_hash_values[i])):
                    binary_features[binary_feature_index] |= (1 if hash == model.one_hot_hash_values[i][border_idx] else 0) * (border_idx + 1)
                binary_feature_index += 1

    if hasattr(model, 'model_ctrs') and model.model_ctrs.used_model_ctrs_count > 0:
        ctrs = [0.] * model.model_ctrs.used_model_ctrs_count;
        calc_ctrs(model.model_ctrs, binary_features, transposed_hash, ctrs)
        for i in range(len(model.ctr_feature_borders)):
            for border in model.ctr_feature_borders[i]:
                binary_features[binary_feature_index] += 1 if ctrs[i] > border else 0
            binary_feature_index += 1

    # Extract and sum values from trees
    results = [0.0] * model.dimension
    tree_splits_index = 0
    current_tree_leaf_values_index = 0
    for tree_id in range(ntree_start, ntree_end):
        current_tree_depth = model.tree_depth[tree_id]
        index = 0
        for depth in range(current_tree_depth):
            border_val = model.tree_split_border[tree_splits_index + depth]
            feature_index = model.tree_split_feature_index[tree_splits_index + depth]
            xor_mask = model.tree_split_xor_mask[tree_splits_index + depth]
            index |= ((binary_features[feature_index] ^ xor_mask) >= border_val) << depth
        results = [result + delta for result, delta in zip(results, model.leaf_values[current_tree_leaf_values_index + index])]
        tree_splits_index += current_tree_depth
        current_tree_leaf_values_index += (1 << current_tree_depth)
    return [model.scale * res + bias for res, bias in zip(results, model.biases)]


def apply_catboost_model(float_features, cat_features=[], ntree_start=0, ntree_end=catboost_model.tree_count):
    """
    Applies the model built by CatBoost.

    Parameters
    ----------

    float_features : list of float features

    cat_features : list of categorical features
        You need to pass float and categorical features separately in the same order they appeared in train dataset.
        For example if you had features f1,f2,f3,f4, where f2 and f4 were considered categorical, you need to pass here float_features=f1,f3, cat_features=f2,f4


    Returns
    -------
    predictions : single (first) formula value for the model and the features

    """
    return apply_catboost_model_multi(float_features, cat_features, ntree_start, ntree_end)[0]
