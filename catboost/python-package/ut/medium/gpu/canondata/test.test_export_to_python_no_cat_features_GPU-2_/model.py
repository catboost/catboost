###  Model data
class catboost_model(object):
    float_features_index = [
        6, 15, 16, 20, 23, 31, 36, 38, 39, 48,
    ]
    float_feature_count = 50
    cat_feature_count = 0
    binary_feature_count = 10
    tree_count = 2
    float_feature_borders = [
        [0.5],
        [0.215384498],
        [0.387778997, 0.795647502],
        [0.405882478],
        [0.0416666493, 0.416666508],
        [0.0950040966],
        [0.5],
        [0.682412982],
        [0.00160859502],
        [0.662613511],
    ]
    tree_depth = [6, 6]
    tree_split_border = [1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1]
    tree_split_feature_index = [9, 8, 3, 4, 2, 0, 5, 4, 1, 7, 6, 2]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = []
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        -0.0159872081130743, 0, 0.001827788073569536, -0.00241227587684989, -0.01710828207433224, 0, -0.003780113765969872, 0.02732429653406143, -0.01001864671707153, 0.00189959816634655, -0.0001757035061018541, -0.005100402049720287, -0.01767570339143276, 0, 0.005213711876422167, 0.04889960214495659, 0, 0, 0.09147840738296509, 0.02057085372507572, 0, 0, 0.02487449534237385, 0.0115411626175046, 0, 0, 0, -0.007562751416116953, 0, 0, 0, 0, 0.01425643637776375, -0.007286288775503635, 0.01101872883737087, 0.02800958603620529, -0.008713617920875549, 0.1065425053238869, -0.01192301977425814, 0.02110324613749981, -0.007286288775503635, 0.002374497475102544, 0.01222138572484255, 0.02749908715486526, 0.00189959816634655, 0.00993724912405014, 0.03425620496273041, 0.02792106010019779, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.004646088927984238, 0, 0, 0, -0.0008338363841176033,
        -0.01520246081054211, 0, 0, 0, -0.006262023467570543, 0, -0.003924420569092035, 0, 0, 0, 0, 0, -0.01667694933712482, 0, -0.003839465323835611, 0, -0.01459948904812336, 0.05867299810051918, 0, -0.01128849759697914, -0.0004827358352486044, 0.01916735246777534, 0.0005391894374042749, 0.01980392262339592, 0, 0.06027792394161224, 0, 0, -0.01042931713163853, 0.02095017768442631, 0.001649297773838043, 0.01597516052424908, 0, 0, 0, 0, 0.01961536332964897, 0, -0.01132655702531338, 0, 0, 0, 0, 0, -0.02513201162219048, 0, -0.01091300323605537, 0, 0, 0, 0, 0, 0.02840910479426384, 0.002173124812543392, 0, -0.00408032163977623, 0, 0, 0, 0, -0.0216811764985323, -0.01041869167238474, 0, 0.003041477873921394
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



