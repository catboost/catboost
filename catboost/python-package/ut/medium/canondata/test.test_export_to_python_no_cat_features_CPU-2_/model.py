###  Model data
class catboost_model(object):
    float_features_index = [
        1, 7, 15, 16, 20, 26, 31, 38, 46, 47,
    ]
    float_feature_count = 50
    cat_feature_count = 0
    binary_feature_count = 10
    tree_count = 2
    float_feature_borders = [
        [0.161994487],
        [0.5],
        [0.645990014],
        [0.102054507],
        [0.272549003, 0.552941501],
        [0.5],
        [0.0950040966],
        [0.726830006],
        [0.163893014],
        [0.387494028, 0.767975509],
    ]
    tree_depth = [6, 6]
    tree_split_border = [1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1]
    tree_split_feature_index = [6, 5, 3, 4, 8, 2, 9, 0, 4, 9, 1, 7]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = []
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.05946996308189739, 0.06050201132893562, 0.0597759872092163, 0.06036483399876603, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06036571608229641, 0.06050201132893562, 0.06029448116662048, 0.06514952970658254, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06009674506236463, 0.06089071607279693, 0.06050201132893562, 0.06050201132893562, 0.0605746798716745, 0.06193475821392059, 0.06050201132893562, 0.06050201132893562, 0.0594607193831092, 0.06165612771390118, 0.06050201132893562, 0.06050201132893562, 0.06022814215791031, 0.06268669740952665, 0.05906421007263216, 0.06050201132893562, 0.06035650779951425, 0.06185469469926672, 0.06050201132893562, 0.06050201132893562, 0.05975863778762609, 0.06050201132893562, 0.05975038477755461, 0.06050201132893562, 0.05969502605322529, 0.06110725575033905, 0.06050201132893562, 0.06050201132893562, 0.05946483401505088, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.0601506083272138, 0.06178352478588342, 0.06050201132893562, 0.06050201132893562, 0.06104073085673532, 0.06222657144513391, 0.06050201132893562, 0.06050201132893562, 0.05978753017388105, 0.06192195767467341, 0.06050201132893562, 0.06050201132893562, 0.06095108994090935, 0.06237451352671954,
        -0.0007230111882521906, 0, 0.0009948666723278347, 0, -0.0007900164519951704, 0, -0.0002659289693480023, 0, 0.0001254563983881354, 0.001428628754144297, 0.001078632236203841, 0.0009796141431045787, -0.0008358364702347598, 0.00160007905108467, 0.000592866816661922, 0, -0.0005340867153276784, 0, 0.0003537692229000622, 0, -0.0007821319066561396, 0, -0.0003675843939963945, 0, 0.001604622291796156, 0.001675244171601597, 0.0008848058784688241, 0.004456863924052284, -0.001148912729734933, 0, -0.0005602905556206702, 0, -0.0005737073891526625, 0, -0.001100966225028356, 0, -0.0001166522483816535, 0, -0.00106129749074719, 0, 0.0001540051217899349, 0.005564426303322108, -0.001032946050067701, 0, 0.0001473638233077871, 0.0003772364785429609, -0.0009327486391357833, 0, -0.0005709572214822599, 0, -0.0002979036512760183, 0, -0.001296188877843233, 0, -0.000560651596176989, 0, -0.0002341298353139103, 0.001675639834827854, 0.0006904073493944664, 0.001700468864075095, -0.0004692910482660644, 0.001468376451942307, 0.0001621911364306871, 0
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



