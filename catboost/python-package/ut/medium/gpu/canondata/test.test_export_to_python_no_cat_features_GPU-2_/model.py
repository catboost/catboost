###  Model data
class catboost_model(object):
    float_features_index = [
        0, 2, 20, 23, 36, 37, 39, 46, 48,
    ]
    float_feature_count = 50
    cat_feature_count = 0
    binary_feature_count = 9
    tree_count = 2
    float_feature_borders = [
        [0.0012634799],
        [0.44707751],
        [0.49607849],
        [1.5],
        [0.5],
        [0.00025269552],
        [0.0017220699],
        [0.67529798],
        [0.2598795, 0.66261351],
    ]
    tree_depth = [6, 5]
    tree_split_border = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    tree_split_feature_index = [8, 0, 1, 3, 5, 8, 3, 6, 2, 4, 7]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = []
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.0002299999759998173, 0, 0.001010674517601728, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001014084555208683, 0, 0, 0, 0.001166666625067592, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001468627364374697, 0.001049999962560833, 0.001727674854919314, 0.002749624894931912, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002085107145830989, 0.00265559321269393, 0, 0, 0.002889422932639718, 0.004042857326567173, 0, 0, 0, 0.001799999969080091, 0, 0, 0, 0.001049999962560833,
        0.0008321835193783045, 0, 0.001828600885346532, 0, 0.0005662433104589581, 0, 0.001081391936168075, 0, 0.001461817068047822, 0, 0.003061093855649233, 0.001769142807461321, 0.0009195390157401562, 0, 0.00115906388964504, 0.001042125048115849, 0, 0, 0.002436290495097637, 0, -1.295756101171719e-05, 0, 0.001525443862192333, 0, -1.295756101171719e-05, 0, 0.001712905708700418, 0, 0, 0, 0.002524094888940454, 0
    ]
cat_features_hashes = {
}

def hash_uint64(string):
    return cat_features_hashes.get(str(string), 0x7fFFffFF)


### Applicator for the CatBoost model

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
    prediction : formula value for the model and the features

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
    result = 0.
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
        result += model.leaf_values[current_tree_leaf_values_index + index]
        tree_splits_index += current_tree_depth
        current_tree_leaf_values_index += (1 << current_tree_depth)
    return result



