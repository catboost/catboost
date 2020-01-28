###  Model data
class catboost_model(object):
    float_features_index = [
        2, 15, 16, 20, 23, 36, 37, 39, 47, 48,
    ]
    float_feature_count = 50
    cat_feature_count = 0
    binary_feature_count = 10
    tree_count = 2
    float_feature_borders = [
        [0.937671542],
        [0.215384498],
        [0.630002975, 0.795647502],
        [0.441176474, 0.492156982],
        [0.0416666493],
        [0.5],
        [0.026869949],
        [0.00181464502],
        [0.753007531],
        [0.211401001],
    ]
    tree_depth = [6, 6]
    tree_split_border = [1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1]
    tree_split_feature_index = [8, 7, 3, 4, 2, 9, 1, 0, 6, 3, 5, 2]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = []
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.04591508209705353, 0.06050201132893562, 0.05813673883676529, 0.06050201132893562, 0.03692908212542534, 0.07043926417827606, 0.04850820451974869, 0.06050201132893562, 0.05122855305671692, 0.06050201132893562, 0.1041892617940903, 0.06050201132893562, 0.06240160763263702, 0.06050201132893562, 0.05293925851583481, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.05293925851583481, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.05210814252495766, 0.1044023558497429, 0.06566349416971207, 0.09627319872379303, 0.05096060782670975, 0.05293925851583481, 0.04979848116636276, 0.08113814145326614, 0.04691294580698013, 0.06698042154312134, 0.07311651110649109, 0.09571218490600586, 0.04537650942802429, 0.07043926417827606, 0.06404908001422882, 0.08321572095155716, 0.06050201132893562, 0.06050201132893562, 0.1438247114419937, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.07107286155223846, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.05073493719100952, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.05966817587614059, 0.06050201132893562,
        -0.01313269883394241, -0.005075628403574228, 0, 0, 0, 0, 0, 0, -0.006224810145795345, -0.008623102679848671, 0, 0, 0, 0, 0, 0, -0.01049762405455112, 0.005245530512183905, 0.06421585381031036, 0.07963623106479645, 0, 0.01120027247816324, 0, 0.007361581083387136, 0.02234085090458393, -0.003822155995294452, 0, 0.008695092052221298, 0, 0.01350340619683266, 0.05900333076715469, 0.06428597867488861, 0, 0.03240378201007843, 0, 0, 0, 0, 0, 0, 0, 0.00427056523039937, 0, 0, 0, 0, 0, 0, 0, 0.02046489156782627, 0, 0, 0, 0, 0, 0, 0, 0.009019204415380955, 0, 0, 0, 0, 0, 0
    ]
    scale = 1
    bias = 0
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
    return model.scale * result + model.bias



