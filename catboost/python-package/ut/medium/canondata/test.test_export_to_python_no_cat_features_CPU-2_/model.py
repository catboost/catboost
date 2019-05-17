###  Model data
class catboost_model(object):
    float_features_index = [
        0, 1, 7, 15, 20, 26, 28, 37, 38, 46, 47,
    ]
    float_feature_count = 50
    cat_feature_count = 0
    binary_feature_count = 11
    tree_count = 2
    float_feature_borders = [
        [0.168183506],
        [0.136649996],
        [0.5],
        [0.645990014],
        [0.272549003],
        [0.5],
        [0.5],
        [0.00311657996],
        [0.392024517],
        [0.163893014],
        [0.387494028, 0.767975509],
    ]
    tree_depth = [6, 6]
    tree_split_border = [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]
    tree_split_feature_index = [7, 5, 0, 4, 9, 3, 10, 6, 1, 10, 2, 8]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = []
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0, 0, 0.001049999981001019, 0, 0.0006999999873340129, 0.0004199999924004077, 0, 0.0003499999936670065, 0.0006999999873340129, 0, 0, 0, 0.0008999999837151595, 0.0008399999848008155, 0, 0.005499108720590722, 0, 0, 0.001310204828003209, 0, 0, 0, 0.002099799992893635, 0.003281666943095616, 0, 0, 0.001081739094670376, 0, 0, 0, 0.001261285375551461, 0.003276219466932844, 0.0001235294095295317, 0, 0.001143749976810068, 0, 0, 0, 0.002145652130229966, 0.001699999969239746, 0.0005409089896827957, 0, 0.0004052632035001316, 0, 0.001076041724695822, 0, 0.001088611397153381, 0.001412068939966888, 0, 0, 0.001978133278083802, 0.001049999981001019, 0, 0, 0.002257627220401316, 0.002847457878670443, 0, 0, 0.002380742281139534, 0, 0, 0, 0.00181075114546265, 0.003175870739941051,
        0.0008166451595296908, 0, 0.001435124236388357, 0, 0.002230458559199401, 0, 0.002666874626702434, 0, 0.001534999525103938, 0.001002557986130936, 0.001371602072510968, 0.001025428335548242, 0.002692151343928724, 0.002324272621936243, 0.001188379847259753, 0.002102531171547649, 0.0002273645927514973, 0, 0.002259198170953273, 0, 0.0005932082092034878, 0, 0.001936697595469175, 0, 0.002274218690520189, 0.00329724810179383, 0.003553552019915716, -2.457164545277724e-05, 0.001361408072189387, 0.002058595183732444, 0.002040357509679229, 0.006411161288685438, 0.001238019183794491, 0, 0.0007759661277461006, 0, 0.002378015135461136, 0, 0.0005649959300306291, 0, 0.001080913918043391, 0.005514189264833609, 0.001790046575156435, 0.001103960055365507, 0.002146005492857177, 0.002352553092417847, 0.001165392567571486, 0, 0.0003389142165572722, 0, 0.000740077287450055, 0, 0.0006711111753359577, 0, 0.001173741518772528, 0, 0.0008677387848554152, 0.003458858412712405, 0.001394273280892068, 0.002416533521423629, 0.003145683167659726, 0.002795911932059462, 0.001047576512888525, 0.003163031305315038
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



