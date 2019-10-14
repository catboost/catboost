###  Model data
class catboost_model(object):
    float_features_index = [
        1, 5, 15, 16, 20, 26, 31, 46, 47,
    ]
    float_feature_count = 50
    cat_feature_count = 0
    binary_feature_count = 9
    tree_count = 2
    float_feature_borders = [
        [0.129020989],
        [0.5],
        [0.645990014, 0.731531501],
        [0.343203485],
        [0.272549003, 0.552941501],
        [0.5],
        [0.0950040966],
        [0.163893014],
        [0.387494028, 0.767975509],
    ]
    tree_depth = [6, 6]
    tree_split_border = [1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1]
    tree_split_feature_index = [6, 5, 3, 4, 7, 2, 8, 0, 4, 8, 2, 1]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = []
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.05946996308189739, 0.06050201132893562, 0.0597759872092163, 0.06036483399876603, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06036571608229641, 0.06050201132893562, 0.06029448116662048, 0.06514952970658254, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06030929969270755, 0.06201176563055445, 0.06050201132893562, 0.06050201132893562, 0.06088885377099409, 0.06061378562065109, 0.06050201132893562, 0.06050201132893562, 0.05979592551074399, 0.06231710802093271, 0.06050201132893562, 0.06050201132893562, 0.06086379020534394, 0.06109824623511206, 0.05906421007263216, 0.06050201132893562, 0.06028851134172113, 0.06185469469926672, 0.06050201132893562, 0.06050201132893562, 0.06004824625411104, 0.06050201132893562, 0.05975038477755461, 0.06050201132893562, 0.05967445309356023, 0.06110725575033905, 0.06050201132893562, 0.06050201132893562, 0.0597759872092163, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.0603308022264912, 0.06226450817009362, 0.06050201132893562, 0.06050201132893562, 0.06221092270785485, 0.06111759986749466, 0.06050201132893562, 0.06050201132893562, 0.06021148334668756, 0.06233052541273109, 0.06050201132893562, 0.06050201132893562, 0.06189712228382211, 0.06102483398043761,
        -0.000917027050763652, 0, -0.0001023161632857904, 0, -0.0006749786591149164, 0, -0.0007672342630328493, 0, 0.0001663165483582841, 0.003226331038366911, 0.0002329458906825826, 0.001478344266430803, -0.0007717087483218283, 0.001182194915445247, 5.012452717852837e-05, 0, -0.0008628047136134319, 0, 0.001732900826582379, 0, -0.0006706854733602141, 0, 3.435305076754058e-05, 0, 0.0005609281733780663, 0.0007760228657202333, 0.001473320309238871, 0.0004394853178986445, 2.199843094468903e-05, 0.0005916955731173785, -0.0006557269432687956, 0, 7.976131087115272e-05, 0, 0.0002605133215158231, 0, -0.001032667716450859, 0, -0.0004968056518123862, 0, -0.0004714617299453535, 0.001760351586397863, 0.001073909387324209, 0.002123790209328512, -0.001448511372609406, 0, -0.0007268539533859603, 0.0005825210508545048, -0.0003529758214693614, 0, 0.001056730442594447, 0, -0.0009167520939934676, 0, -0.001098800568124502, 0, 0.001872449541173461, 0.001656261283980514, 0.0006601236074602712, 0.002449939671961454, -0.0003088540056609102, 0.001341961855967867, 0.0002397258267875516, 0
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



