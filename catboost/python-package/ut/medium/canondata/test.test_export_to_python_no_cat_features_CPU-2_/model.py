###  Model data
class catboost_model(object):
    float_features_index = [
        0, 2, 3, 14, 15, 31, 32, 37, 46, 47, 48,
    ]
    float_feature_count = 50
    cat_feature_count = 0
    binary_feature_count = 11
    tree_count = 2
    float_feature_borders = [
        [0.323544502],
        [0.0031594648, 0.521369994],
        [0.5],
        [0.550980508],
        [0.979140043],
        [0.0950040966],
        [0.5],
        [0.239182502],
        [0.287964523],
        [0.238694996],
        [0.447151482],
    ]
    tree_depth = [6, 6]
    tree_split_border = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    tree_split_feature_index = [1, 1, 5, 4, 7, 10, 6, 2, 9, 3, 8, 0]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = []
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.001259011842534326, 0, 0.001474009916287763, 0.00064166661547497, 0, 0, 0.002863636311820962, 0.0009333333164453506, 0.0002386363593184135, 0, 0.002116666806116696, 0, 0, 0, 0, 0.001679999969601631, 0.0005249999905005097, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002088370528948082, 0, 0.001851857674803305, 0.001553571389056742, 0, 0, 0.002822325645650539, 0.004138979825677001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001799999967430319, 0, 0, 0, 0, 0, 0, 0, 0,
        0.0003675624400133544, 0, 0.001059092784733128, 0, 0, 0, 0.002066946665087761, 0.001192459419076518, 0.001103175683534166, 0, 0.0004773354283567961, 0, 0, 0, 0.001827927523090045, 0, 0, 0, 0.001934663401212227, 0, 0, 0, 0.002079562911984692, 0.002016393416732761, 0, 0, 0.00153485204725672, 0, 0, 0, 0.001034478467845754, 0, 0.0006763936415426848, 0, 0.0005583088061050648, 0, 0, 0, 0.00178538384126724, 0.003265464248219695, 0.0004448876734695067, 0, 0.0005178430285843198, 0, 0, 0, 0.00121787682371597, 0.003195419440159142, 0, 0, 0.002190701743713123, 0, 0, 0, 0.00230819225408413, 0.003807514519097499, 0, 0, 0.00167323412960814, 0, 0, 0, 0.001315321241806467, 0.003950911460742678
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



