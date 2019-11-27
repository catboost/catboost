###  Model data
class catboost_model(object):
    float_features_index = [
        1, 15, 16, 20, 31, 32, 33, 35, 37, 39,
    ]
    float_feature_count = 50
    cat_feature_count = 0
    binary_feature_count = 10
    tree_count = 2
    float_feature_borders = [
        [0.156479999],
        [0.748417497],
        [0.0504557006],
        [0.398038983],
        [0.0867389515],
        [0.5],
        [0.185759991],
        [0.318073004, 0.423586994],
        [0.00513814017],
        [0.00177894998, 0.005463365],
    ]
    tree_depth = [6, 6]
    tree_split_border = [1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1]
    tree_split_feature_index = [4, 0, 9, 1, 8, 7, 5, 9, 3, 7, 2, 6]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = []
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.05960256437951928, 0.06070146909282823, 0.06015918889076437, 0.05959448117928647, 0.0599721206163218, 0.06062821605892021, 0.0605089520658085, 0.06288086010413954, 0.05955195210314157, 0.06099448115395449, 0.06229438070265529, 0.06050201132893562, 0.06192215820207864, 0.06190509985324542, 0.06176463823760008, 0.06162483396958104, 0.06005946608783778, 0.06062279438329973, 0.06004824625411104, 0.06064448116028749, 0.06103598718641752, 0.06270338815639177, 0.06267781612256605, 0.06172957858253918, 0.05938985894333002, 0.06132279437063375, 0.0613444811476215, 0.06109824623511206, 0.0616944811412885, 0.0620775132274143, 0.06146759860387622, 0.06253466764112585, 0.05988602370841774, 0.0604175986228752, 0.05969421006123277, 0.06050201132893562, 0.06009357756463582, 0.0597648340096226, 0.06094687741927594, 0.06145321607326258, 0.06024776317952554, 0.06096483398790947, 0.06303508723080734, 0.06050201132893562, 0.0608775135811513, 0.06275863771560382, 0.06071954670341606, 0.06099448115395449, 0.05989259863237469, 0.0648667073624649, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06221517819009029, 0.06050201132893562, 0.06404448115730725, 0.05957821609654565, 0.06004824625411104, 0.06050201132893562, 0.06050201132893562, 0.06050201132893562, 0.06240996302870024, 0.06057324624461156, 0.06019598720161671,
        -0.0009068310388559039, 0.0005823790048818268, 0.0003022397759053274, 0.001236387425717793, -0.001110814547534013, 0, -0.0009955282825560336, 0.0005900790335013008, -0.0007270945767319858, -0.0004503618368394611, -0.0004830951003004704, 0, -0.0007312046291846942, 0, -0.0003228709852496245, 0.001331784484289174, -0.0009001571778393699, 0.0008336580731940341, 0.0001846719011632318, 0.001297891181080824, -0.0009618816738515486, 0, -8.710280354874731e-05, 0.000935069827523146, -0.000469122101802195, 0.000927062975956497, 0.0009596543982336384, 0.0001628772571593989, -0.0005596775005835593, 6.159951384984342e-05, -0.0003414199319814777, 0.002115943356260227, 0, -0.001142222399221946, 0, 0.001164318208439542, -0.000457458598429662, 0.0005900790335013008, 0, 9.584290627186962e-05, 0.001870434369045688, 0.0005927637363117251, 0.0005925413825713574, 0.0005833861550049742, 0.0002303670248576997, 0.005328653757534604, -0.000235488818658018, 0, 0, -0.0003494213597650448, -0.0003075385840448801, 0.001911465205838799, 0, 0, 0.0003968806892349656, 0.001932862827397617, -7.592085855706009e-05, 0, 0.001140556113073425, 0.002650395860214851, 0.0002143034913390684, 0, -0.0003199517355933474, 0.002342043877120708
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



