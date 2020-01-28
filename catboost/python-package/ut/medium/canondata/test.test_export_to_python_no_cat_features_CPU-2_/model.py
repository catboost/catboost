###  Model data
class catboost_model(object):
    float_features_index = [
        0, 1, 2, 15, 16, 31, 33, 35, 37, 39, 46,
    ]
    float_feature_count = 50
    cat_feature_count = 0
    binary_feature_count = 11
    tree_count = 2
    float_feature_borders = [
        [0.834502459],
        [0.156479999],
        [0.937812984],
        [0.748417497],
        [0.594812512],
        [0.0867389515],
        [0.185759991],
        [0.423586994],
        [0.00513814017],
        [0.00177894998, 0.005463365],
        [0.464960515],
    ]
    tree_depth = [6, 6]
    tree_split_border = [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    tree_split_feature_index = [5, 1, 9, 3, 8, 7, 2, 9, 10, 0, 4, 6]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = []
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        -0.01499078282534252, 0.00332429613918066, -0.005713707430565611, -0.01512550283223391, -0.008831512074296673, 0.002103412213424841, 0.0001156789504668929, 0.03964748047292233, -0.01583432078382558, 0.008207830600440502, 0.0298728235630375, 0, 0.02366911508142948, 0.02338480926118791, 0.02104378228143948, 0.01871371109570776, -0.007375754183158278, 0.002013050951063633, -0.007562751416116953, 0.0023744972422719, 0.008899597823619843, 0.0366896146110126, 0.03626341403772434, 0.02045945468403044, -0.01853587350773591, 0.01367971766740084, 0.0140411639586091, 0.009937248658388853, 0.01987449731677771, 0.02625836556156476, 0.01609312160871923, 0.03387760596039394, -0.01026646057143807, -0.001406878465786576, -0.01346335476264358, 0, -0.006807229557150119, -0.01228628892983709, 0.007414435004730793, 0.01585341275980075, -0.004237469251549572, 0.007713711155312402, 0.0422179326415062, 0, 0.006258371010146759, 0.03761044061846203, 0.003625589655712247, 0.008207830600440502, -0.01015687850303948, 0.07274493551813066, 0, 0, 0, 0.02855278165744884, 0, 0.05904116512586673, -0.01539658755064011, -0.007562751416116953, 0, 0, 0, 0.03179919570684433, 0.00118724862113595, -0.005100402235984802,
        -0.009776548494163753, 0, -0.0004823473147140704, 0, 0.0006686191379420487, 0, 0.00197923692820046, 0.001307706499383562, -0.007401229853826732, 0, 0.02096871966459878, 0.005962349195033311, 0, 0, 0.007140203175090606, 0, 0.008357482100824973, 0, 0.0744871213682976, 0, 0, 0, 0.01239996451039868, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004691376493660588, 0.0003303209319710732, 0.009402653884111427, 0.1217359174825972, -0.003059979277895764, 0, 0.004428125522045438, 0.009124076661343378, 0.05959413343225606, 0.05539775782381184, -0.00358151692808384, 0.005702547913339611, 0, 0, 0.01204003252658165, 0.01557480075163767, 0, 0, 0, 0, 0, 0, -0.009735384877496088, 0, 0, 0, 0, 0, 0, 0, 0, 0
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



