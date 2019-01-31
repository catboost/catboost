###  Model data
class catboost_model(object):
    float_features_index = [
        1, 2, 3, 13, 18, 31, 32, 34, 39, 48, 49,
    ]
    float_feature_count = 50
    cat_feature_count = 0
    binary_feature_count = 11
    tree_count = 2
    float_feature_borders = [
        [0.00132575491, 0.130032003],
        [0.521369994],
        [0.5],
        [0.5],
        [0.5],
        [0.134183004],
        [0.5],
        [0.5],
        [2.37056011e-05],
        [0.250777483],
        [0.5],
    ]
    tree_depth = [6, 6]
    tree_split_border = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1]
    tree_split_feature_index = [1, 5, 0, 8, 9, 7, 6, 4, 2, 10, 0, 3]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = []
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.0005996913577188494, 0, 0, 0, 0, 0, 0, 0, 0.000221052627579162, 0, 0, 0, 0.0009768016478353648, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001960402433831547, 0, 0, 0, 0.0004199999924004077, 0, 0, 0, 0, 0, 0, 0, 0.0006999999873340129, 0, 0, 0, 0.001368055558297782, 0.0001866665999591365, 0, 0.001679999969601631, 0.001604166730772702, 0.001049999981001019, 0.001499999972858599, 0.003954763324079735, 0, 0, 0, 0, 0, 0, 0.001049999981001019, 0, 0.001779299344200712, 0.001465909057127481, 0.002944927643757796, 0.003975728267858036,
        0.0008915419558863947, 0, 0.0003512460733605982, 0, 0.0004428744834216426, 0.0005497278883047848, 0.0007437548665033633, 0, 0, 0, 0, 0, 0.001210233305897622, 0.003847465523299554, 0.001161668204743658, 0.002286348013900796, 0, 0, 0, 0, 0.001845609629668785, 0.00280269449486017, 0.001692741537757669, 0, 0, 0, 0, 0, 0.002007853056220058, 0.003279802623184852, 0.002235919726910874, 0.003849602074058009, -1.028042304539491e-05, 0, -4.497685082360273e-06, 0, 0.001735347943945, 0, -1.657894669786881e-06, 0, 0, 0, 0, 0, 0.002337718806557816, 0.001027913024166518, 0.0008234633792182209, 0, 0, 0, 0, 0, 0, 0, 0.0008223767607047237, 0, 0, 0, 0, 0, 0.002439206489894105, 0.001638476035060068, 0.003277561698191198, 0
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



