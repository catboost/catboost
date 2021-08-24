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
        -0.0145869292318821, 0, -0.002365272026509047, 0, -0.02357292920351028, 0.00993724912405014, -0.01199380867183208, 0, -0.009273458272218704, 0, 0.04368724673986435, 0, 0.00189959816634655, 0, -0.007562751416116953, 0, 0, 0, -0.007562751416116953, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.008393868803977966, 0.04390034452080727, 0.005161481443792582, 0.03577119112014771, -0.009541401639580727, -0.007562751416116953, -0.01070352923125029, 0.02063612826168537, -0.01358906365931034, 0.006478413008153439, 0.01261449605226517, 0.03521017357707024, -0.01512550283223391, 0.00993724912405014, 0.003547068685293198, 0.02271371148526669, 0, 0, 0.08332269638776779, 0, 0, 0, 0.01057085301727057, 0, 0, 0, -0.009767072275280952, 0, 0, 0, -0.0008338363841176033, 0,
        -0.01313269883394241, -0.005075628403574228, 0, 0, 0, 0, 0, 0, -0.006224810145795345, -0.008623102679848671, 0, 0, 0, 0, 0, 0, -0.01049762405455112, 0.005245530512183905, 0.06421585381031036, 0.07963623106479645, 0, 0.01120027247816324, 0, 0.007361581083387136, 0.02234085090458393, -0.003822155995294452, 0, 0.008695092052221298, 0, 0.01350340619683266, 0.05900333076715469, 0.06428597867488861, 0, 0.03240378201007843, 0, 0, 0, 0, 0, 0, 0, 0.00427056523039937, 0, 0, 0, 0, 0, 0, 0, 0.02046489156782627, 0, 0, 0, 0, 0, 0, 0, 0.009019204415380955, 0, 0, 0, 0, 0, 0
    ]
    scale = 1
    bias = 0.06050201133
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



