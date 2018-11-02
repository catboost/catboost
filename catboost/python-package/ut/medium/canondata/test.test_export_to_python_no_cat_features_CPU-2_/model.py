###  Model data
class catboost_model(object):
    float_features_index = [
        1, 3, 4, 9, 11, 15, 19, 23, 32, 39, 47, 48,
    ]
    float_feature_count = 50
    cat_feature_count = 0
    binary_feature_count = 12
    tree_count = 2
    float_feature_borders = [
        [0.18692601],
        [0.5],
        [0.5],
        [0.5],
        [0.5],
        [0.95771205],
        [0.5],
        [0.097222149],
        [0.5],
        [0.00032979899],
        [0.75300753],
        [0.26409501],
    ]
    tree_depth = [6, 6]
    tree_split_border = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    tree_split_feature_index = [8, 0, 5, 9, 11, 6, 10, 2, 7, 4, 3, 1]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = []
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.0006955065930426542, 0, 0, 0, 0.0001810344794829344, 0, 0, 0, 0.001098490266995831, 0, 0.001589791657724417, 0, 0.001828124975902028, 0.001679999969601631, 0, 0, 0.001329545421779833, 0, 0, 0, 0.0005833335090428549, 0, 0, 0, 0.001529906971799882, 0.00269666736505924, 0.002342030875375432, 0.001938461503386497, 0.001049999981001019, 0.001049999981001019, 0.0008765624896273946, 0, 0.0002999999945717198, 0, 0, 0, 0, 0, 0, 0, 0.0003187499942324523, 0, 0.001949999964716179, 0, 0.0005249999905005097, 0, 0, 0, 0.001469999973401427, 0.002677381649373881, 0, 0, 0, 0, 0, 0, 0.001873529377864564, 0.003867032899167183, 0.003096296610434842, 0.003725676117001744, 0, 0.001679999969601631, 0, 0,
        0.0006982795337251036, 0, 0, 0, 0.0004140269530082293, 0, 0, 0, 0.000228045619107548, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0009179783236801756, 0.004755924048153645, 0.001566909289921064, 0.002035708918338523, 0.0007676849182978572, 0.002487475692542658, 0.0005208372840066653, 0.0024006384178372, 0.0005134157523437949, 0, 0.0009316294913245664, 0.0005135256884684811, 0, 0, 0, 0, 0, 0, 0.00191797439944338, 0.003727659281110782, 0, 0, 0.002112067698191821, 0.003495110258087365, 0, 0, 0.000791394002761522, 0.001031641075398032, 0, 0, 0.0006124347462520387, 0
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



