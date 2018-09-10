###  Model data
class catboost_model(object):
    float_features_index = [
        2, 15, 16, 23, 24, 31, 36, 38, 48,
    ]
    float_feature_count = 49
    cat_feature_count = 0
    binary_feature_count = 9
    tree_count = 2
    float_feature_borders = [
        [0.8374055],
        [0.60392153],
        [0.387779],
        [0.58333349, 1.5],
        [0.93881702],
        [0.061012201],
        [0.5],
        [0.97901797],
        [0.27336848, 0.66261351]
    ]
    tree_depth = [6, 5]
    tree_split_border = [2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
    tree_split_feature_index = [8, 4, 0, 3, 1, 8, 3, 5, 7, 6, 2]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = []
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.0005814876058138907, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001133998390287161, 0, 0.0009975000284612179, 0, 0, 0, 0.0008555554668419063, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001372431288473308, 0.002215142827481031, 0.001875349087640643, 0.002333333250135183, 0, 0, 0.001049999962560833, 0.006780804600566626, 0, 0, 0, 0, 0, 0, 0, 0, 0.002136547351256013, 0.002390978159382939, 0.002491590334102511, 0.003389361780136824, 0, 0, 0.002099999925121665, 0.003947368357330561, 0, 0, 0, 0.002099999925121665, 0, 0, 0, 0,
        0.001135568716563284, 0.001330881263129413, 0, 0, 0.0002743172226473689, 0, 0, 0, 0.001436591031961143, 0.001021189265884459, 0.003254958894103765, 0.003336918773129582, 0.000900790560990572, 0, 0.001032500062137842, 0, 0.002713314490392804, -2.869173840736039e-05, 0, 0, 0, 0, 0, 0, 0.00300825503654778, 0, 0.002350773196667433, 0.000385663821361959, 0, 0, 0, 0
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



