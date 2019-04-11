###  Model data
class catboost_model(object):
    float_features_index = [
        0, 2, 3, 14, 15, 20, 32, 37, 46, 47, 48,
    ]
    float_feature_count = 50
    cat_feature_count = 0
    binary_feature_count = 11
    tree_count = 2
    float_feature_borders = [
        [0.322089016],
        [0.0031594648, 0.521369994],
        [0.5],
        [0.550980508],
        [0.996324539],
        [0.656862974],
        [0.5],
        [0.0069912402],
        [0.285950482],
        [0.236585006],
        [0.445143998],
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
        0.001294099685395057, 0, 0.001663731629059374, 0.0005249999905005097, 0.0008576818214875049, 0, 0.0005905404957665791, 0.001049999981001019, 0, 0, 0, 0, 0, 0, 0, 0, 0.0005414062402036506, 0, 0.002567500461246807, 0.0007333332748285372, 0.0005249999905005097, 0, 0, 0.001679999969601631, 0, 0, 0, 0, 0, 0, 0, 0, 0.002037932443256619, 0, 0.002110218347738532, 0.003083823474551387, 0.002161956464572121, 0, 0.001152380927474726, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0005249999905005097, 0, 0.003047058768395115, 0.004090148077996925, 0, 0, 0.0005249999905005097, 0.002062499998603015, 0, 0, 0, 0, 0, 0, 0, 0,
        0.0003231377723979838, 0, 0.00105791920933408, 0, 0, 0, 0.002063173418219615, 0.001217081715965126, 0.001102714260714291, 0, 0.0004444306049201454, 0, 0, 0, 0.001594016627768246, 0, 0, 0, 0.001935656303693709, 0, 0, 0, 0.002064051449280681, 0.002037610333493785, 0, 0, 0.001519155501243254, 0, 0, 0, 0.001067039113331727, 0, 0.0008372005929878915, 0, 0.0005312171758752161, 0, 0, 0, 0.001773409163652216, 0.003262653168409861, 0.0004447483693683492, 0, 0.0005997946154227071, 0, 0, 0, 0.001232067526134975, 0.003190115195858002, 0, 0, 0.002193397001291078, 0, 0, 0, 0.002352321022124672, 0.003810887531325806, 0, 0, 0.001677124391007667, 0, 0, 0, 0.001325081612224604, 0.003956065626275658
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



