###  Model data
class catboost_model(object):
    float_features_index = [
        1, 2, 14, 15, 16, 20, 31, 33, 35, 39,
    ]
    float_feature_count = 50
    cat_feature_count = 0
    binary_feature_count = 10
    tree_count = 2
    float_feature_borders = [
        [0.156479999],
        [0.937812984],
        [0.398038983],
        [0.748417497],
        [0.0504557006, 0.564610481],
        [0.398038983],
        [0.134183004],
        [0.185759991],
        [0.318073004, 0.423586994],
        [0.005463365],
    ]
    tree_depth = [6, 6]
    tree_split_border = [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1]
    tree_split_feature_index = [6, 0, 9, 3, 4, 8, 1, 2, 5, 8, 4, 7]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = []
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
    ]
    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        [-0.01476383881461352], [-0.001527162562859686], [-0.006224901143771906], [-0.008500670393308004], [-0.005548771821939555], [0.03321244932711125], [-0.0007866994090765327], [0.02974564745090902], [-0.01629468016139227], [0.01505178487614581], [0.01007350760379008], [0.009937248658388853], [0.01932053081691265], [0.02920025381548651], [0.02495628258062376], [0.03457912147045136], [0.02889959737658501], [0], [0], [0], [-0.02016733710964521], [0], [0.02151997769251466], [0], [-0.007286288908549717], [0], [0.07163563167506998], [0], [0.03654116361091534], [0], [0.008546827094895499], [0.00702062388882041], [-0.010533983425406], [0.06907871986428897], [-0.01257979418886335], [0], [-0.006533451605497337], [0.02355278163616146], [0.00832707500306978], [0.0532265424051068], [-0.005099951297420414], [0.0023744972422719], [0.006521995941346342], [0], [0.01014725991559249], [0.03873599536324802], [0.002457431196395693], [0.0023744972422719], [0], [0], [-0.007562751416116953], [0], [0], [0], [-0.005775703676044941], [-0.007562751416116953], [-0.007562751416116953], [0], [0.07546812086366117], [0], [-0.005100402235984802], [0], [0.008899597823619843], [0],
        [-0.005185098429306783], [0], [-0.001709653076446105], [0], [-0.006138443525332251], [0], [-0.01215968660400086], [0], [-0.01519701123714509], [0], [-0.003332020375386257], [0], [-0.007397120564829815], [0], [-0.005090657595602796], [0], [0.001331868529870192], [0.001844042242737487], [0.0006912374866847305], [-0.002530957310227677], [0.0003234795604749375], [0], [-0.003460162245464885], [0], [0.006208794650302458], [0], [0.01026425088720871], [0], [-0.009062705858377137], [0], [-0.003170142447814982], [0.00509524923798285], [-0.01921837785302248], [0], [0.01186028514135008], [-0.007371856095759492], [0], [0], [-0.008735171039180272], [0.008055775548870627], [0.02558814778354052], [0], [0.01813888668502663], [0], [0.00976561973448551], [0.05585603478054205], [0.01800648072426705], [0], [0.005122852583468298], [0], [0.01407679499998793], [0.01682312171287297], [0], [0], [0.0002649408239403998], [0.06453569420613349], [0.005155574898665143], [0.1034671237030998], [0.01886628208918604], [0.006703344076586683], [-0.008603635791500676], [0], [-0.003121383204795378], [0.04060238017912277]
    ]
    scale = 1
    biases = [0.06050201133]
    dimension = 1


cat_features_hashes = {
}


def hash_uint64(string):
    return cat_features_hashes.get(str(string), 0x7fFFffFF)


### Applicator for the CatBoost model
def apply_catboost_model_multi(float_features, cat_features=[], ntree_start=0, ntree_end=catboost_model.tree_count):
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
    predictions : list of formula values for the model and the features

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
        if len(model.float_feature_borders[i]) > 0:
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
            if len(model.one_hot_hash_values[i]) > 0:
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
    results = [0.0] * model.dimension
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
        results = [result + delta for result, delta in zip(results, model.leaf_values[current_tree_leaf_values_index + index])]
        tree_splits_index += current_tree_depth
        current_tree_leaf_values_index += (1 << current_tree_depth)
    return [model.scale * res + bias for res, bias in zip(results, model.biases)]


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
    predictions : single (first) formula value for the model and the features

    """
    return apply_catboost_model_multi(float_features, cat_features, ntree_start, ntree_end)[0]
