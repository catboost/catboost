### Types to hold CTR's data

class catboost_model_ctr(object):
    def __init__(self, base_hash, base_ctr_type, target_border_idx, prior_num, prior_denom, shift, scale):
        self.base_hash = base_hash
        self.base_ctr_type = base_ctr_type
        self.target_border_idx = target_border_idx
        self.prior_num = prior_num
        self.prior_denom = prior_denom
        self.shift = shift
        self.scale = scale

    def calc(self, count_in_class, total_count):
        ctr = (count_in_class + self.prior_num) / float(total_count + self.prior_denom)
        return (ctr + self.shift) * self.scale


class catboost_bin_feature_index_value(object):
    def __init__(self, bin_index, check_value_equal, value):
        self.bin_index = bin_index
        self.check_value_equal = check_value_equal
        self.value = value


class catboost_ctr_mean_history(object):
    def __init__(self, sum, count):
        self.sum = sum
        self.count = count


class catboost_ctr_value_table(object):
    def __init__(self, index_hash_viewer, target_classes_count, counter_denominator, ctr_mean_history, ctr_total):
        self.index_hash_viewer = index_hash_viewer
        self.target_classes_count = target_classes_count
        self.counter_denominator = counter_denominator
        self.ctr_mean_history = ctr_mean_history
        self.ctr_total = ctr_total

    def resolve_hash_index(self, hash):
        try:
            return self.index_hash_viewer[hash]
        except KeyError:
            return None


class catboost_ctr_data(object):
    def __init__(self, learn_ctrs):
        self.learn_ctrs = learn_ctrs


class catboost_projection(object):
    def __init__(self, transposed_cat_feature_indexes, binarized_indexes):
        self.transposed_cat_feature_indexes = transposed_cat_feature_indexes
        self.binarized_indexes = binarized_indexes


class catboost_compressed_model_ctr(object):
    def __init__(self, projection, model_ctrs):
        self.projection = projection
        self.model_ctrs = model_ctrs


class catboost_model_ctrs_container(object):
    def __init__(self, used_model_ctrs_count, compressed_model_ctrs, ctr_data):
        self.used_model_ctrs_count = used_model_ctrs_count
        self.compressed_model_ctrs = compressed_model_ctrs
        self.ctr_data = ctr_data


###  Model data
class catboost_model(object):
    float_features_index = [
        0, 2, 3,
    ]
    float_feature_count = 6
    cat_feature_count = 11
    binary_feature_count = 10
    tree_count = 2
    float_feature_borders = [
        [45.5, 61.5],
        [13.5],
        [3280],
    ]
    tree_depth = [6, 6]
    tree_split_border = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1]
    tree_split_feature_index = [0, 9, 1, 2, 4, 3, 5, 1, 6, 0, 7, 8]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
        [0.2531880140304565],
        [-0.009802941232919693],
        [0.6861486434936523],
        [0.3333666622638702],
        [0.8007199168205261],
        [0.8007199168205261],
        [0.578125]
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01928571425378323, 0.01200000010430813, 0.02769230678677559, 0.02210526168346405, 0, 0.004999999888241291, 0.006000000052154064, 0.007499999832361937, 0, 0, 0, 0, 0, 0, 0, 0,
        0.007310356944799423, 0, -3.749999814317562e-05, 0, 0.01944969967007637, 0.02751540020108223, 0.003624374745413661, 0.01489500049501657, 0, 0, 0, 0, 0, 0.01173473615199327, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    model_ctrs = catboost_model_ctrs_container(
        used_model_ctrs_count = 7,
        compressed_model_ctrs = [
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 5, 7],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493035932493, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 7],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 10041049327410906822, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791580653471472, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 6],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493260528854, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 6, 8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791582220620454, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 6, 8, 10],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493297533872, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [7],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791580653471474, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            )
        ],
        ctr_data = catboost_ctr_data(
            learn_ctrs = {
                768791580653471472 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 13987540656699198946 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18089724839685297862 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10353740403438739754 : 1, 3922001124998993866 : 2, 13686716744772876732 : 4, 18293943161539901837 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 13), catboost_ctr_mean_history(sum = 3.08286e-44, count = 20), catboost_ctr_mean_history(sum = 0, count = 37), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 3)],
                    ctr_total = [0, 13, 22, 20, 0, 37, 0, 2, 0, 4, 0, 3]
                ),
                768791580653471474 :
                catboost_ctr_value_table(
                    index_hash_viewer = {3607388709394294015 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18356215166324018775 : 2, 18365206492781874408 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 14559146096844143499 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 11416626865500250542 : 3, 5549384008678792175 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 22), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 14), catboost_ctr_mean_history(sum = 2.66247e-44, count = 17), catboost_ctr_mean_history(sum = 0, count = 22), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [0, 22, 2, 3, 0, 14, 19, 17, 0, 22, 1, 1]
                ),
                768791582220620454 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 8729380230485332353 : 7, 9977784445157143938 : 44, 10895282230322158083 : 41, 11761827888484752132 : 2, 12615796933138713349 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16179912168674497673 : 12, 11168821652508931466 : 34, 18446744073709551615 : 0, 10475827037446056716 : 27, 14750448552479345421 : 20, 16495354791085375886 : 28, 10135854469738880143 : 48, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12233930600554179099 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3208547115700824735 : 23, 18277252057819687584 : 43, 11380194846102552353 : 29, 18446744073709551615 : 0, 14030234294679959587 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14597065438263107115 : 15, 1433532977505263916 : 11, 17401263168096565421 : 31, 15971769898056770222 : 8, 2808237437298927023 : 47, 1256940733300575664 : 1, 18446744073709551615 : 0, 9689317982178109362 : 42, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6069959757871852856 : 6, 7318363972543664441 : 10, 18446744073709551615 : 0, 5876843741908031419 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5859361731318044735 : 16, 6636790901455000384 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 10322519660234049603 : 32, 7078520115819519811 : 37, 8011495361248663491 : 33, 9259899575920475076 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9323661606633862475 : 40, 18146214905751188428 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3907755348917795664 : 49, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8720774373489072856 : 14, 6896376012912388953 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10662309976865883491 : 24, 9111013272867532132 : 39, 10359417487539343717 : 46, 17543390521745065830 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 13529097553057277673 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 2350285447658856812 : 3, 16689654767293439084 : 38, 18446744073709551615 : 0, 6176161755113003119 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 346329515989321719 : 45, 6353480505666359544 : 17, 4263979015577900536 : 30, 9547190628529608953 : 35, 9583115984936959099 : 36, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 8.40779e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 9.80909e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 3, 0, 1, 0, 3, 0, 3, 0, 1, 1, 1, 6, 1, 2, 4, 0, 1, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 0, 4, 0, 1, 0, 3, 0, 1, 0, 1, 0, 1, 2, 5, 0, 1, 0, 2, 7, 1, 0, 5, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 1, 0, 1, 0, 2, 0, 1, 1, 0, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
                ),
                10041049327410906822 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 14339393822756684802 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9042729150784344201 : 33, 18446744073709551615 : 0, 1434197551787351435 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 15505904404980462094 : 9, 17132136727440490127 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2690081920877379861 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1532562665111458202 : 31, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14397741423195249570 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18052491238949695525 : 29, 18446744073709551615 : 0, 18446744073709551615 : 0, 5961989641064476328 : 26, 777303952308747305 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15890374780837199661 : 35, 16738422394153010094 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 11699844009042731185 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5970980967522331961 : 38, 1590910265550022970 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11601902557128801344 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14909972007605802568 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5867491582615089871 : 22, 2743913003832016080 : 23, 18446744073709551615 : 0, 7716892253132515538 : 27, 18446744073709551615 : 0, 8557324777698838228 : 18, 18446744073709551615 : 0, 4383219007951416278 : 14, 5231266621267226711 : 4, 10600672353715374294 : 21, 7399805521932916569 : 30, 18446744073709551615 : 0, 2482461723210813787 : 37, 2164920571584601052 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7725883579590371171 : 16, 16967431379427980772 : 32, 18446744073709551615 : 0, 18446744073709551615 : 0, 4392210334409271911 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 13356805169196840554 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10871179537331551727 : 7, 18446744073709551615 : 0, 3402816234720019185 : 17, 2724972351271196914 : 36, 8122374639275138803 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11414809869912342394 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 15496913078522606461 : 2, 18446744073709551615 : 0, 17469145413950259711 : 34},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 5.60519e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 11), catboost_ctr_mean_history(sum = 7.00649e-45, count = 2), catboost_ctr_mean_history(sum = 1.26117e-44, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6), catboost_ctr_mean_history(sum = 8.40779e-45, count = 5), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.26117e-44, count = 2), catboost_ctr_mean_history(sum = 9.80909e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [1, 1, 4, 1, 1, 11, 5, 2, 9, 2, 1, 6, 6, 5, 2, 2, 1, 1, 9, 2, 7, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 3, 1, 1, 1, 1]
                ),
                17677952493035932493 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 5463648634093988865 : 17, 10335157342305182722 : 32, 6738126486361565313 : 4, 10791805669449703300 : 37, 2663236581938862725 : 29, 18446744073709551615 : 0, 8062072459088193799 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11155470568209245195 : 41, 18446744073709551615 : 0, 18446744073709551615 : 0, 1131617383599204750 : 13, 4832104736055316622 : 38, 8806694637943861776 : 33, 14839272646072616465 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 15228474649950355220 : 43, 13076441876572047509 : 1, 15233604573492176790 : 9, 13439225070635064087 : 0, 9152574924415960088 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2585548803766369181 : 44, 3108819379344245150 : 15, 13701985375152518815 : 20, 16363877839430637216 : 3, 15404114944395482785 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11497229771350565927 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 2435711933190352810 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1798551784282379183 : 27, 18446744073709551615 : 0, 18446744073709551615 : 0, 5796622587947976498 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 6185824591825715253 : 40, 18446744073709551615 : 0, 4290540494726515383 : 35, 3557046681785410616 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 4265003389608318523 : 23, 18446744073709551615 : 0, 5967132958851282493 : 12, 4654205393486057278 : 19, 18446744073709551615 : 0, 6356334962729021248 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 10716948865937726787 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16961813080521916615 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17235223807996045521 : 28, 6812569332418330194 : 39, 18446744073709551615 : 0, 12566808769950761940 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6127862249290423643 : 42, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5151892753385378284 : 36, 15470067739584089325 : 30, 15822331420583681518 : 34, 13574783642484889455 : 16, 13134404219107339119 : 45, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13938448541244431350 : 7, 18446744073709551615 : 0, 11602785020767681144 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7799312154570739071 : 31},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 5.60519e-45, count = 7), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 4, 7, 5, 0, 0, 2, 0, 2, 0, 7, 1, 1, 0, 1, 0, 6, 0, 3, 0, 5, 0, 2, 0, 1, 1, 0, 0, 1, 3, 6, 0, 2, 0, 1, 1, 0, 0, 7, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1, 3, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
                ),
                17677952493260528854 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17985374731566054150 : 22, 18446744073709551615 : 0, 4969880389554839688 : 30, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1883285504791108373 : 14, 14139902777924824981 : 23, 18446744073709551615 : 0, 18446744073709551615 : 0, 17540248381108153753 : 15, 18446744073709551615 : 0, 2120068639763588379 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 1277857586923739550 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9915512646490226338 : 9, 18446744073709551615 : 0, 5780999427119446436 : 37, 15493676505554854693 : 31, 14453653496344422438 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3622512433858345389 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9415440463389949361 : 35, 18446744073709551615 : 0, 15689261734764374707 : 7, 17838331352489460532 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18403625549429831228 : 24, 18446744073709551615 : 0, 16192880425411659454 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6383791411268594626 : 3, 18033916581698980546 : 33, 18446744073709551615 : 0, 11961955270333222981 : 13, 18446744073709551615 : 0, 11191788834073534919 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17618698769621849933 : 28, 5730630563994427981 : 32, 16620451033949360975 : 19, 647125264645798733 : 38, 7150295984444125389 : 29, 18446744073709551615 : 0, 12157540499542742995 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1072059942279220057 : 34, 10177020748048094298 : 1, 18446744073709551615 : 0, 9494950831378731228 : 8, 18446744073709551615 : 0, 518361807174415198 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 592499207252901221 : 18, 4098784705883188966 : 36, 10062654256758136807 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3618574749222493677 : 21, 18446744073709551615 : 0, 13088729798727729263 : 27, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2625225542620233849 : 20, 6645299512826462586 : 2, 5651789874985220091 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 8.40779e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 7), catboost_ctr_mean_history(sum = 9.80909e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 4, 2, 2, 0, 5, 0, 3, 0, 1, 1, 1, 6, 1, 2, 4, 0, 1, 0, 2, 0, 5, 0, 1, 0, 2, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 2, 7, 7, 1, 0, 7, 0, 1, 0, 1, 0, 1, 0, 3, 0, 2, 0, 4, 0, 1, 0, 1, 0, 2, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
                ),
                17677952493297533872 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 228832412018222341 : 29, 18446744073709551615 : 0, 11579036573410064263 : 37, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2142920538933900555 : 49, 18446744073709551615 : 0, 11420714090427158285 : 12, 18446744073709551615 : 0, 17720405802426315535 : 24, 3215834049561110672 : 51, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 346575239343974036 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 13139983920087306647 : 41, 14860408764928037144 : 6, 286844492446271769 : 47, 18446744073709551615 : 0, 18446744073709551615 : 0, 10925792178412610972 : 3, 12726869934920056605 : 22, 11945848411936959644 : 39, 18446744073709551615 : 0, 11343638620497380128 : 44, 9857611124702919969 : 20, 15541558334966787106 : 36, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10990677728635501222 : 8, 4919457811166910375 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 4237122415554814250 : 52, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 339035928827901487 : 42, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8200830002684883256 : 14, 6893797804197340345 : 28, 1058988547593232698 : 43, 11714417785040418747 : 23, 18446744073709551615 : 0, 6067291172676902717 : 35, 16636473811085647678 : 18, 18446744073709551615 : 0, 483329372556896832 : 26, 3198032362459766081 : 31, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12661894127993305031 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4340360739111205579 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1471101928894068943 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 464994231589622356 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5278641733246315480 : 17, 14915048362378503384 : 30, 1537907742216832473 : 48, 5054839022797264859 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6888411174261376229 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16939687026671270763 : 50, 14120581721888279787 : 46, 18080292852670312173 : 5, 7952734526884932333 : 27, 15038402360561546607 : 32, 9875412811804264560 : 1, 8723830392309106799 : 38, 16771855022716002162 : 34, 8813616260402002415 : 11, 7006154001587127924 : 40, 5933240490959917807 : 33, 18446744073709551615 : 0, 5540766610232480247 : 15, 18446744073709551615 : 0, 16586264761736307193 : 4, 18446744073709551615 : 0, 6712598941894663547 : 45, 17585370940655764860 : 9, 9392162505557741693 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 7.00649e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 9.80909e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [0, 3, 0, 1, 0, 3, 0, 3, 0, 1, 1, 1, 5, 1, 2, 3, 0, 1, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 1, 0, 3, 0, 1, 0, 1, 0, 1, 2, 5, 0, 1, 0, 2, 7, 1, 0, 5, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
                )
            }
        )
    )


### Routines to compute CTRs

def calc_hash(a, b):
    max_int = 0xffFFffFFffFFffFF
    MAGIC_MULT = 0x4906ba494954cb65
    return (MAGIC_MULT * ((a + MAGIC_MULT * b) & max_int)) & max_int


def calc_hashes(binarized_features, hashed_cat_features, transposed_cat_feature_indexes, binarized_feature_indexes):
    result = 0
    for cat_feature_index in transposed_cat_feature_indexes:
        result = calc_hash(result, hashed_cat_features[cat_feature_index])
    for bin_feature_index in binarized_feature_indexes:
        binary_feature = binarized_features[bin_feature_index.bin_index]
        if not(bin_feature_index.check_value_equal):
            result = calc_hash(result, 1 if (binary_feature >= bin_feature_index.value) else 0)
        else:
            result = calc_hash(result, 1 if (binary_feature == bin_feature_index.value) else 0)
    return result


def calc_ctrs(model_ctrs, binarized_features, hashed_cat_features, result):
    ctr_hash = 0
    result_index = 0

    for i in range(len(model_ctrs.compressed_model_ctrs)):
        proj = model_ctrs.compressed_model_ctrs[i].projection
        ctr_hash = calc_hashes(binarized_features, hashed_cat_features, proj.transposed_cat_feature_indexes, proj.binarized_indexes)
        for j in range(len(model_ctrs.compressed_model_ctrs[i].model_ctrs)):
            ctr = model_ctrs.compressed_model_ctrs[i].model_ctrs[j]
            learn_ctr = model_ctrs.ctr_data.learn_ctrs[ctr.base_hash]
            ctr_type = ctr.base_ctr_type
            bucket = learn_ctr.resolve_hash_index(ctr_hash)
            if bucket is None:
                result[result_index] = ctr.calc(0, 0)
            else:
                if ctr_type == "BinarizedTargetMeanValue" or ctr_type == "FloatTargetMeanValue":
                    ctr_mean_history = learn_ctr.ctr_mean_history[bucket]
                    result[result_index] = ctr.calc(ctr_mean_history.sum, ctr_mean_history.count)
                elif ctr_type == "Counter" or ctr_type == "FeatureFreq":
                    ctr_total = learn_ctr.ctr_total
                    denominator = learn_ctr.counter_denominator
                    result[result_index] = ctr.calc(ctr_total[bucket], denominator)
                elif ctr_type == "Buckets":
                    ctr_history = learn_ctr.ctr_total
                    target_classes_count = learn_ctr.target_classes_count
                    total_count = 0
                    good_count = ctr_history[bucket * target_classes_count + ctr.target_border_idx];
                    for class_id in range(target_classes_count):
                        total_count += ctr_history[bucket * target_classes_count + class_id]
                    result[result_index] = ctr.calc(good_count, total_count)
                else:
                    ctr_history = learn_ctr.ctr_total;
                    target_classes_count = learn_ctr.target_classes_count;

                    if target_classes_count > 2:
                        good_count = 0
                        total_count = 0
                        for class_id in range(ctr.target_border_idx + 1):
                            total_count += ctr_history[bucket * target_classes_count + class_id]
                        for class_id in range(ctr.target_border_idx + 1, target_classes_count):
                            good_count += ctr_history[bucket * target_classes_count + class_id]
                        total_count += good_count;
                        result[result_index] = ctr.calc(good_count, total_count);
                    else:
                        result[result_index] = ctr.calc(ctr_history[bucket * 2 + 1], ctr_history[bucket * 2] + ctr_history[bucket * 2 + 1])
            result_index += 1



cat_features_hashes = {
    "Female": -2114564283,
    "Protective-serv": -2075156126,
    "Assoc-voc": -2029370604,
    "Married-civ-spouse": -2019910086,
    "Federal-gov": -1993066135,
    "Transport-moving": -1903253868,
    "Farming-fishing": -1888947309,
    "Prof-school": -1742589394,
    "Self-emp-inc": -1732053524,
    "?": -1576664757,
    "Handlers-cleaners": -1555793520,
    "0": -1438285038,
    "Philippines": -1437257447,
    "Male": -1291328762,
    "11th": -1209300766,
    "Unmarried": -1158645841,
    "Local-gov": -1105932163,
    "Divorced": -993514283,
    "Some-college": -870577664,
    "Asian-Pac-Islander": -787966085,
    "Sales": -760428919,
    "Self-emp-not-inc": -661998850,
    "Widowed": -651660490,
    "Masters": -453513993,
    "State-gov": -447941100,
    "Doctorate": -434936054,
    "White": -218697806,
    "Own-child": -189887997,
    "Amer-Indian-Eskimo": -86031875,
    "Exec-managerial": -26537793,
    "Husband": 60472414,
    "Italy": 117615621,
    "Not-in-family": 143014663,
    "n": 239748506,
    "Married-spouse-absent": 261588508,
    "Prof-specialty": 369959660,
    "Assoc-acdm": 475479755,
    "Adm-clerical": 495735304,
    "Bachelors": 556725573,
    "HS-grad": 580496350,
    "Craft-repair": 709691013,
    "Other-relative": 739168919,
    "Other-service": 786213683,
    "9th": 840896980,
    "Separated": 887350706,
    "10th": 888723975,
    "Mexico": 972041323,
    "Hong": 995245846,
    "1": 1121341681,
    "Tech-support": 1150039955,
    "Black": 1161225950,
    "Canada": 1510821218,
    "Wife": 1708186408,
    "United-States": 1736516096,
    "Never-married": 1959200218,
    "Machine-op-inspct": 2039859473,
    "7th-8th": 2066982375,
    "Private": 2084267031,
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



