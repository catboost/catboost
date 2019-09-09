###  Model data
class catboost_model(object):
    float_features_index = [
        15, 16, 20, 23, 36, 38, 39, 47, 48,
    ]
    float_feature_count = 50
    cat_feature_count = 0
    binary_feature_count = 9
    tree_count = 2
    float_feature_borders = [
        [0.215384498],
        [0.630002975, 0.795647502],
        [0.492156982],
        [0.0416666493],
        [0.5],
        [0.02983425],
        [0.00181464502, 0.0072560953],
        [0.753007531],
        [0.211401001, 0.640262485],
    ]
    tree_depth = [6, 6]
    tree_split_border = [1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2]
    tree_split_feature_index = [7, 6, 2, 3, 1, 8, 8, 0, 1, 5, 4, 6]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = []
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.05962679535150528, 0.06050201132893562, 0.06036009639501572, 0.06050201132893562, 0.05908763408660889, 0.06109824776649475, 0.05978238210082054, 0.06050201132893562, 0.05994560569524765, 0.06050201132893562, 0.06312324851751328, 0.06050201132893562, 0.06061598658561707, 0.06050201132893562, 0.06004824489355087, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06004824489355087, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.05999837815761566, 0.0631360337138176, 0.06081169843673706, 0.06264828145503998, 0.05992952734231949, 0.06004824489355087, 0.05985980108380318, 0.06174017861485481, 0.05968666821718216, 0.06089071556925774, 0.06125888228416443, 0.06261461973190308, 0.05959448218345642, 0.06109824776649475, 0.06071483716368675, 0.06186483427882195, 0.06050201132893562, 0.06050201132893562, 0.065501369535923, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06113626062870026, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.05991598591208458, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06045198068022728, 0.06050201132893562,
        -0.0004489484999794513, 0, -0.0008267186349257827, -0.0004494714376050979, 0, 0, 0.004414282273501158, -0.0004493698943406343, -0.000456087727798149, 0, -0.000785626529250294, 0.0002510512131266296, 0, 0, 0, 0, -0.0009024837054312229, -0.0004476500034797937, -0.0003274795599281788, 0.0009805334266275167, 0, 0, 0.003764267079532146, -0.0004489484999794513, 0.000147815648233518, 0.003849697066470981, -0.0003933792468160391, 0.001169789349660277, 0, 0, 0, 0, -0.000456087727798149, 0, 0.0001724530884530395, 0.001614656648598611, 0, 0, -0.0001406052906531841, 0.0006504327175207436, 0, 0, 0.000112106507003773, 0.0002320445637451485, 0, 0, 0.002150259679183364, 0.0009266755660064518, 0, 0, 0.0002637545403558761, 0.004137720447033644, 0, 0, 0, 7.605149585288018e-05, -0.000456087727798149, 0.004558220505714417, 0.0007700700662098825, 0.001277962350286543, 0, 0, 0, -0.0003066200588364154
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



