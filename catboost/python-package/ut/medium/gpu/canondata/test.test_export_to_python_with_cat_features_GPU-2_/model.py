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
        0, 2,
    ]
    float_feature_count = 3
    cat_feature_count = 11
    binary_feature_count = 10
    tree_count = 2
    float_feature_borders = [
        [61.5, 68.5],
        [12.5, 13.5]
    ]
    tree_depth = [6, 6]
    tree_split_border = [2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
    tree_split_feature_index = [0, 8, 1, 2, 3, 9, 4, 1, 5, 0, 6, 7]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
        [0.212610587477684],
        [0.2531880140304565],
        [0.6642736196517944],
        [1.000899910926819],
        [0.8007199168205261],
        [0.8007199168205261],
        [0.6542679071426392],
        [0.4607843160629272]
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007499999832361937, 0, 0, 0, 0, 0, 0, 0, 0.006000000052154064, 0, 0.02400000020861626, 0.007499999832361937, 0.007499999832361937, 0, 0.01200000010430813, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007499999832361937, 0, 0, 0, 0, 0, 0, 0, 0.0169565211981535, 0, 0.02571428567171097, 0, 0.003000000026077032, 0, 0.02285714261233807, 0,
        0.01847253181040287, 0.02758516371250153, 0.002167582279071212, 0.01473392825573683, 0, 0, 0, 0, -0.0001017391259665601, 0.01181785762310028, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    model_ctrs = catboost_model_ctrs_container(
        used_model_ctrs_count = 8,
        compressed_model_ctrs = [
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 7],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 2)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 5819498284355557857, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 7, 8],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 2)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 5819498284603408945, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
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
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 2)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 12627245789391619613, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 8405694746487331109, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            )
        ],
        ctr_data = catboost_ctr_data(
            learn_ctrs = {
                768791580653471472 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 13987540656699198946 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18089724839685297862 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10353740403438739754 : 1, 3922001124998993866 : 0, 13686716744772876732 : 4, 18293943161539901837 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 37), catboost_ctr_mean_history(sum = 3.08286e-44, count = 20), catboost_ctr_mean_history(sum = 0, count = 13), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 3)],
                    ctr_total = [0, 37, 22, 20, 0, 13, 0, 2, 0, 4, 0, 3]
                ),
                768791582220620454 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 8729380230485332353 : 34, 9977784445157143938 : 47, 10895282230322158083 : 22, 11761827888484752132 : 4, 12615796933138713349 : 42, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16179912168674497673 : 19, 11168821652508931466 : 17, 18446744073709551615 : 0, 10475827037446056716 : 45, 14750448552479345421 : 11, 16495354791085375886 : 13, 10135854469738880143 : 48, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12233930600554179099 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3208547115700824735 : 14, 18277252057819687584 : 40, 11380194846102552353 : 6, 18446744073709551615 : 0, 14030234294679959587 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14597065438263107115 : 46, 1433532977505263916 : 37, 17401263168096565421 : 24, 15971769898056770222 : 43, 2808237437298927023 : 20, 1256940733300575664 : 21, 18446744073709551615 : 0, 9689317982178109362 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6069959757871852856 : 1, 7318363972543664441 : 41, 18446744073709551615 : 0, 5876843741908031419 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5859361731318044735 : 36, 6636790901455000384 : 30, 18446744073709551615 : 0, 18446744073709551615 : 0, 7078520115819519811 : 28, 10322519660234049603 : 49, 8011495361248663491 : 18, 9259899575920475076 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9323661606633862475 : 32, 18146214905751188428 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3907755348917795664 : 33, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8720774373489072856 : 0, 6896376012912388953 : 27, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10662309976865883491 : 5, 9111013272867532132 : 15, 10359417487539343717 : 29, 17543390521745065830 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 13529097553057277673 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 2350285447658856812 : 23, 16689654767293439084 : 44, 18446744073709551615 : 0, 6176161755113003119 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 346329515989321719 : 39, 4263979015577900536 : 10, 6353480505666359544 : 12, 9547190628529608953 : 31, 9583115984936959099 : 38, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 8.40779e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 9.80909e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 4, 6, 1, 0, 3, 0, 2, 0, 3, 0, 5, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 2, 5, 0, 1, 0, 1, 7, 1, 2, 1, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 3, 0, 3, 1, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 4, 0, 1, 0, 3, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                ),
                5819498284355557857 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16651102300929268102 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17241896836922419469 : 14, 18446744073709551615 : 0, 10511965914255575823 : 39, 9263222292810378768 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6908362001823204373 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10984690017285232025 : 34, 18446744073709551615 : 0, 13013334951445741211 : 25, 18446744073709551615 : 0, 11118050854346541341 : 38, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1178507314092247330 : 7, 18124759156733634467 : 19, 11481715753106083236 : 10, 5594842188654002339 : 29, 13183845322349047206 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15878007324930404523 : 30, 18446744073709551615 : 0, 5342579366432390957 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13092564086442952000 : 0, 12955372608910449601 : 32, 11197279989343752130 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2759029251343736904 : 15, 11560944888103294025 : 9, 863745154244537034 : 24, 13263074457346257995 : 31, 6835357266805764556 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1227410053004078929 : 16, 5421808501429601746 : 2, 2929539622247042899 : 33, 18446744073709551615 : 0, 5303738068466106581 : 4, 18446744073709551615 : 0, 7005867637709070551 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 13535017740039938266 : 27, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6084421232781469407 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 1416006194736185826 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14197630471391805927 : 18, 17162667701925208680 : 23, 18446744073709551615 : 0, 9529215433346522346 : 36, 18273958486853833579 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 2701066908808445294 : 11, 13605543448808549998 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17744431985855206516 : 21, 11659615095675342580 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3609240935860829562 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 18182721499268926077 : 37, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 9), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 5.60519e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 4.2039e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [0, 9, 2, 0, 0, 1, 4, 7, 0, 7, 0, 6, 0, 5, 1, 1, 0, 1, 0, 6, 3, 6, 0, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 4, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 2, 0, 1, 2, 0, 0, 1, 0, 1, 5, 0, 0, 1, 0, 1, 0, 2, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
                ),
                5819498284603408945 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 401572100562674692 : 2, 18446744073709551615 : 0, 15483923052928748550 : 39, 12879637026568809095 : 41, 793550578637923848 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11367447619088669583 : 11, 5265189619257386128 : 7, 4243019055446252944 : 43, 7714913839382636178 : 16, 18446744073709551615 : 0, 2395930809040249492 : 13, 116261182353282069 : 47, 6322089685280714644 : 38, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14730630066036795803 : 46, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13687336289042331679 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5278875108181112996 : 26, 4457098312139029797 : 12, 12062999459536534438 : 6, 18446744073709551615 : 0, 2409616297963976360 : 3, 18446744073709551615 : 0, 6401305903214724138 : 22, 18446744073709551615 : 0, 13010046892757689900 : 30, 18446744073709551615 : 0, 18446744073709551615 : 0, 2457936645703814959 : 34, 11036119054636294576 : 21, 9928946807531223473 : 33, 2486846435837533490 : 24, 18035421909196229939 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14292590550055890113 : 36, 13993739391568110402 : 27, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17239079498146583879 : 15, 18446744073709551615 : 0, 1488087840841630025 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 10635501563021545804 : 49, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1670208868805258833 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 5733202716806891092 : 18, 12366794655858091989 : 42, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2328590962904621532 : 25, 18446744073709551615 : 0, 7642631318464954334 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 2966914650778391649 : 23, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16422441931114309222 : 37, 3772880582916128230 : 35, 18446744073709551615 : 0, 8454321268541353705 : 40, 13553183120897172586 : 0, 6965341922180312555 : 32, 18446744073709551615 : 0, 18446744073709551615 : 0, 12647497933473231982 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8531551406414910835 : 8, 3948120307064453107 : 31, 1150935252797962357 : 20, 18446744073709551615 : 0, 1078861496847316471 : 9, 780010338359536760 : 48, 18446744073709551615 : 0, 12321202553328987770 : 29, 13267872202687779963 : 45, 18341677509001906300 : 44, 12802406888695251965 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 5.60519e-45, count = 5), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 4.2039e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [0, 7, 2, 0, 0, 1, 4, 5, 0, 5, 0, 5, 0, 5, 0, 1, 0, 1, 0, 5, 3, 6, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 4, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 2, 0, 0, 1, 0, 1, 0, 1, 5, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0]
                ),
                8405694746487331109 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 8473802870189803490 : 0, 7071392469244395075 : 3, 18446744073709551615 : 0, 8806438445905145973 : 2, 619730330622847022 : 1, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.14906e-43, count = 12), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6)],
                    ctr_total = [82, 12, 1, 6]
                ),
                12627245789391619613 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9226604805100152147 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 1034166431492604838 : 1, 13302932820562179799 : 2, 6203552979315789704 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 1113395566489815627 : 0, 13957701839509617452 : 7, 15316838452862012827 : 3, 18446744073709551615 : 0, 5765263465902070143 : 6},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 22), catboost_ctr_mean_history(sum = 2.52234e-44, count = 17), catboost_ctr_mean_history(sum = 0, count = 22), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 13), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 22, 18, 17, 0, 22, 1, 1, 0, 13, 2, 3, 1, 0, 0, 1]
                ),
                17677952493260528854 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17985374731566054150 : 24, 18446744073709551615 : 0, 4969880389554839688 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1883285504791108373 : 36, 14139902777924824981 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 17540248381108153753 : 27, 18446744073709551615 : 0, 2120068639763588379 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 1277857586923739550 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9915512646490226338 : 3, 18446744073709551615 : 0, 5780999427119446436 : 30, 15493676505554854693 : 29, 14453653496344422438 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3622512433858345389 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9415440463389949361 : 19, 18446744073709551615 : 0, 15689261734764374707 : 26, 17838331352489460532 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18403625549429831228 : 12, 18446744073709551615 : 0, 16192880425411659454 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6383791411268594626 : 20, 18033916581698980546 : 34, 18446744073709551615 : 0, 11961955270333222981 : 8, 18446744073709551615 : 0, 11191788834073534919 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5730630563994427981 : 23, 647125264645798733 : 37, 16620451033949360975 : 10, 17618698769621849933 : 38, 7150295984444125389 : 17, 18446744073709551615 : 0, 12157540499542742995 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1072059942279220057 : 25, 10177020748048094298 : 14, 18446744073709551615 : 0, 9494950831378731228 : 33, 18446744073709551615 : 0, 518361807174415198 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 592499207252901221 : 7, 4098784705883188966 : 31, 10062654256758136807 : 32, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3618574749222493677 : 5, 18446744073709551615 : 0, 13088729798727729263 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2625225542620233849 : 13, 6645299512826462586 : 4, 5651789874985220091 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 8.40779e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 9.80909e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 5, 6, 1, 0, 4, 0, 2, 0, 5, 0, 7, 0, 3, 0, 1, 0, 2, 0, 2, 2, 7, 0, 1, 0, 1, 7, 1, 2, 2, 0, 2, 0, 2, 0, 1, 0, 2, 0, 1, 0, 3, 0, 4, 1, 1, 0, 1, 0, 1, 0, 1, 2, 4, 0, 3, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                ),
                17677952493297533872 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 228832412018222341 : 6, 18446744073709551615 : 0, 11579036573410064263 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2142920538933900555 : 20, 18446744073709551615 : 0, 11420714090427158285 : 19, 18446744073709551615 : 0, 17720405802426315535 : 5, 3215834049561110672 : 33, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 346575239343974036 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 13139983920087306647 : 32, 14860408764928037144 : 1, 286844492446271769 : 41, 18446744073709551615 : 0, 18446744073709551615 : 0, 10925792178412610972 : 23, 12726869934920056605 : 27, 11945848411936959644 : 46, 18446744073709551615 : 0, 11343638620497380128 : 42, 9857611124702919969 : 11, 15541558334966787106 : 50, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10990677728635501222 : 45, 4919457811166910375 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 4237122415554814250 : 37, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 339035928827901487 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8200830002684883256 : 0, 6893797804197340345 : 13, 1058988547593232698 : 16, 11714417785040418747 : 14, 18446744073709551615 : 0, 6067291172676902717 : 31, 16636473811085647678 : 26, 18446744073709551615 : 0, 483329372556896832 : 30, 3198032362459766081 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12661894127993305031 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4340360739111205579 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1471101928894068943 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 464994231589622356 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14915048362378503384 : 10, 5278641733246315480 : 12, 1537907742216832473 : 29, 5054839022797264859 : 38, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6888411174261376229 : 34, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16939687026671270763 : 51, 14120581721888279787 : 36, 18080292852670312173 : 25, 7952734526884932333 : 47, 8723830392309106799 : 28, 9875412811804264560 : 21, 15038402360561546607 : 52, 16771855022716002162 : 17, 5933240490959917807 : 18, 7006154001587127924 : 15, 8813616260402002415 : 39, 18446744073709551615 : 0, 5540766610232480247 : 48, 18446744073709551615 : 0, 16586264761736307193 : 44, 18446744073709551615 : 0, 6712598941894663547 : 49, 17585370940655764860 : 3, 9392162505557741693 : 43, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 7.00649e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 9.80909e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 3, 5, 1, 0, 3, 0, 2, 0, 3, 0, 5, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 2, 5, 0, 1, 0, 1, 7, 1, 2, 1, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 3, 0, 3, 1, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 1, 0, 0, 3, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
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



