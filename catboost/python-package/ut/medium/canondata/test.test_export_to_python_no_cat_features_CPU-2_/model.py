###  Model data
class catboost_model(object):
    float_features_index = [
        3, 11, 13, 16, 19, 21, 32, 48,
    ]
    float_feature_count = 49
    cat_feature_count = 0
    binary_feature_count = 8
    tree_count = 2
    float_feature_borders = [
        [0.5],
        [0.5],
        [0.5],
        [0.051106799],
        [0.5],
        [0.5],
        [0.5],
        [0.2598795]
    ]
    tree_depth = [6, 4]
    tree_split_border = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    tree_split_feature_index = [6, 4, 3, 5, 2, 7, 6, 0, 1, 4]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = []
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.0008211111193150276, 0, 0.0002249999959287899, 0, 0.001252702682193469, 0, 0.0003937499928753823, 0, 0.000989974279526389, 0.001679999969601631, 0.000647499988283962, 0, 0.0007328258582250274, 0, 0.001145454524728385, 0, 0.0008999999837151595, 0, 0, 0, 0, 0, 0, 0, 0.001374999961815775, 0, 0, 0, 0.001399999974668026, 0, 0.0005249999905005097, 0, 0.001039998001456305, 0, 0.002099999962002039, 0.001259999977201223, 0.001890090150245136, 0.002099999962002039, 0.002390476385184691, 0.00347205903071033, 0.001214583574654528, 0.0004199999924004077, 0.001259999977201223, 0.003790092530154747, 0.001781553489653492, 0.002363889468771702, 0.002231249959627166, 0.003923684268934943, 0, 0, 0, 0, 0.002604545391452584, 0, 0, 0, 0, 0, 0, 0, 0.00275191166696851, 0.002099999962002039, 0, 0,
        0.0007538577474999657, 0, 0.001673253488408078, 0.002738652127683553, 0.0002184838828884162, 0, 0.0008825505085869145, 0, 0.0003120346099853925, 0, 0.00158349975932882, 0.003739948724389159, 0, 0, 0, 0
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



