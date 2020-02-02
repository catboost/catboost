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
    binary_feature_count = 12
    tree_count = 2
    float_feature_borders = [
        [60.5],
        [9.5],
        [3280],
    ]
    tree_depth = [6, 6]
    tree_split_border = [1, 1, 1, 1, 1, 255, 1, 1, 1, 1, 1, 1]
    tree_split_feature_index = [9, 4, 6, 0, 2, 3, 5, 10, 7, 8, 1, 11]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0]
    cat_features_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    one_hot_cat_feature_index = [9]
    one_hot_hash_values = [
        [-1291328762]
    ]
    ctr_feature_borders = [
        [0.599734068],
        [0.0539215691],
        [0.471354157],
        [0.142156869],
        [0.771959484],
        [0.925000072],
        [0.460784316],
        [0.0147058833]
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0, 0.02722772210836411, 0, 0, -0.09777227789163589, 0.04001649841666222, 0.05445544421672821, 0.09335219115018845, 0, 0, 0, 0, 0, 0.02722772210836411, 0, 0.02722772210836411, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.156435638666153, 0, -0.2234794944524765, 0.0622347928583622, -0.1496149599552155, 0.0622347928583622, 0.003150312695652246, 0.09529702365398407, 0, 0, 0, 0, 0, 0, 0, 0.02722772210836411, -0.09777227789163589, 0, 0, 0, -0.09777227789163589, 0, -0.156435638666153, 0, 0, 0, 0, 0, 0, 0, -0.156435638666153, 0,
        0, 0, 0, 0, 0, 0, 0, 0.02683393657207489, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.07907040417194366, 0, 0, 0, 0, 0, -0.07821781933307648, 0, 0, 0, 0, 0.01531559228897095, 0, 0, 0, 0.02382425963878632, 0.02382425963878632, 0, 0.05775443837046623, 0, 0.01555869728326797, -0.1624610275030136, 0.0333411805331707, 0, 0.03022748790681362, 0, 0.04391272366046906, 0, 0.02450494840741158, 0.02222566306591034, 0.05160301178693771, 0, 0.03189532831311226, -0.1618926376104355, -0.1863919943571091, 0, 0.04293429851531982, -0.08555074036121368, -0.05279784649610519, 0, 0.03630362451076508, 0, 0.03461810201406479, 0, 0.03063118457794189, 0.01555869728326797, 0.06042798608541489
    ]
    scale = 1
    bias = 0.7821782231
    model_ctrs = catboost_model_ctrs_container(
        used_model_ctrs_count = 8,
        compressed_model_ctrs = [
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791580653471478, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 11154939068695130472, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791580653471473, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 11154939068695130479, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791580653471472, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 768791580653471472, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 11154939068695130483, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [10],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 11154939068695130481, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
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
                768791580653471473 :
                catboost_ctr_value_table(
                    index_hash_viewer = {7537614347373541888 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5903587924673389870 : 2, 18278593470046426063 : 6, 10490918088663114479 : 3, 18446744073709551615 : 0, 407784798908322194 : 7, 5726141494028968211 : 1, 1663272627194921140 : 10, 8118089682304925684 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15431483020081801594 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 1403990565605003389 : 5, 3699047549849816830 : 11, 14914630290137473119 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 9.80909e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 5.60519e-45, count = 24), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 5.60519e-45, count = 16), catboost_ctr_mean_history(sum = 4.2039e-45, count = 16), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0)],
                    ctr_total = [7, 3, 1, 3, 4, 24, 0, 3, 4, 16, 3, 16, 0, 4, 0, 5, 1, 1, 0, 1, 0, 3, 2, 0]
                ),
                768791580653471478 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 15379737126276794113 : 5, 18446744073709551615 : 0, 14256903225472974739 : 3, 18048946643763804916 : 1, 2051959227349154549 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7024059537692152076 : 6, 18446744073709551615 : 0, 15472181234288693070 : 2, 8864790892067322495 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.4013e-44, count = 58), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 4.2039e-45, count = 6), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0)],
                    ctr_total = [10, 58, 0, 4, 1, 6, 1, 5, 3, 6, 2, 0, 5, 0]
                ),
                11154939068695130472 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 15379737126276794113 : 5, 18446744073709551615 : 0, 14256903225472974739 : 3, 18048946643763804916 : 1, 2051959227349154549 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7024059537692152076 : 6, 18446744073709551615 : 0, 15472181234288693070 : 2, 8864790892067322495 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 9.52883e-44, count = 4), catboost_ctr_mean_history(sum = 9.80909e-45, count = 6), catboost_ctr_mean_history(sum = 1.26117e-44, count = 2)],
                    ctr_total = [68, 4, 7, 6, 9, 2, 5]
                ),
                11154939068695130479 :
                catboost_ctr_value_table(
                    index_hash_viewer = {7537614347373541888 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5903587924673389870 : 2, 18278593470046426063 : 6, 10490918088663114479 : 3, 18446744073709551615 : 0, 407784798908322194 : 7, 5726141494028968211 : 1, 1663272627194921140 : 10, 8118089682304925684 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15431483020081801594 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 1403990565605003389 : 5, 3699047549849816830 : 11, 14914630290137473119 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.4013e-44, count = 4), catboost_ctr_mean_history(sum = 3.92364e-44, count = 3), catboost_ctr_mean_history(sum = 2.8026e-44, count = 19), catboost_ctr_mean_history(sum = 5.60519e-45, count = 5), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 2)],
                    ctr_total = [10, 4, 28, 3, 20, 19, 4, 5, 2, 1, 3, 2]
                ),
                11154939068695130481 :
                catboost_ctr_value_table(
                    index_hash_viewer = {2136296385601851904 : 0, 7428730412605434673 : 2, 9959754109938180626 : 6, 14256903225472974739 : 1, 8056048104805248435 : 3, 18446744073709551615 : 0, 12130603730978457510 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10789443546307262781 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.30321e-43, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [93, 1, 2, 2, 1, 1, 1]
                ),
                11154939068695130483 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 8473802870189803490 : 0, 7071392469244395075 : 2, 18446744073709551615 : 0, 8806438445905145973 : 3, 619730330622847022 : 1, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.14906e-43, count = 12), catboost_ctr_mean_history(sum = 8.40779e-45, count = 1)],
                    ctr_total = [82, 12, 6, 1]
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
    return model.scale * result + model.bias



