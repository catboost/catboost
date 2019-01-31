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
        0, 1, 2, 3, 4, 5,
    ]
    float_feature_count = 6
    cat_feature_count = 11
    binary_feature_count = 22
    tree_count = 40
    float_feature_borders = [
        [17.5, 36.5, 46.5, 51.5, 58.5, 59.5],
        [98122.5, 121010, 122181, 332801, 337225.5],
        [10.5, 11.5, 12.5, 14.5],
        [1087, 3280, 7493, 11356, 17537.5],
        [808.5, 1738, 1881.5, 2189.5],
        [17, 35.5, 36.5, 42, 46.5, 55]
    ]
    tree_depth = [2, 5, 3, 6, 0, 2, 3, 4, 1, 5, 1, 2, 0, 2, 1, 1, 1, 4, 0, 3, 4, 0, 0, 1, 1, 3, 6, 3, 0, 5, 0, 2, 5, 1, 0, 1, 1, 2, 3, 3]
    tree_split_border = [3, 4, 3, 4, 2, 4, 255, 1, 2, 255, 5, 3, 2, 1, 4, 2, 4, 255, 1, 2, 2, 5, 2, 1, 255, 1, 3, 1, 3, 1, 2, 6, 3, 4, 4, 3, 2, 3, 2, 5, 4, 1, 2, 2, 4, 255, 2, 1, 2, 1, 4, 4, 3, 1, 5, 5, 1, 4, 2, 2, 4, 5, 1, 5, 5, 1, 1, 1, 6, 1, 1, 4, 2, 1, 3, 1, 2, 2, 4, 1, 2, 3, 3, 1, 2, 1, 1]
    tree_split_feature_index = [4, 4, 4, 2, 0, 0, 6, 5, 3, 6, 5, 3, 17, 20, 2, 5, 3, 6, 2, 2, 19, 0, 12, 3, 6, 3, 10, 11, 19, 12, 3, 5, 4, 12, 5, 4, 12, 5, 0, 3, 19, 19, 20, 3, 2, 6, 10, 9, 1, 1, 4, 19, 1, 10, 5, 12, 17, 1, 9, 7, 4, 5, 8, 1, 12, 8, 19, 15, 0, 14, 18, 12, 2, 4, 0, 0, 9, 15, 12, 13, 4, 12, 2, 7, 3, 21, 16]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 253, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 253, 0, 0, 0, 0, 0, 0, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    one_hot_cat_feature_index = [9]
    one_hot_hash_values = [
        [-2114564283, -1291328762]
    ]
    ctr_feature_borders = [
        [0.999998987, 4.99999905],
        [5.99999905],
        [5.99999905, 8.99999905],
        [9.99999905, 11.999999, 13.999999],
        [12.999999],
        [7.99999905, 9.99999905, 10.999999, 12.999999, 14.999999],
        [13.999999],
        [8.99999905],
        [5.99999905, 10.999999],
        [9.99999905],
        [1.99999905, 3.99999905],
        [9.99999905],
        [2.99999905, 6.99999905, 7.99999905, 11.999999],
        [8.99999905, 12.999999],
        [7.99999905]
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.02339999947696924, 0.00599999986588955, 0, 0,
        0.02591746097304156, 0, 0, 0, 0.01456124968433753, 0, 0, 0, 0, 0, 0, 0, 0.01831124960051849, 0, 0, 0, 0.02467631196266134, -4.499999798834327e-05, 0, 0, 0.01067962477404065, 0, -0.0001754999921545387, 0.005963999867498875, 0, 0, 0, 0, 0.01563574966424331, 0, 0, 0,
        0.007139427504624127, 0.02075110084135458, 0, -0.0007778093404988993, 0.01140819021960694, 0.02448511731501992, 0, 0,
        0.01111436881839412, 0, 0, 0, 0.006983794251792642, 0, 0, 0, 0, 0, 0, 0, 0.02418690523181335, 0, -0.0002869345420918655, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.009037572049722845, 0, -0.0003547387578144583, 0, 0.0112283141754463, 0, 0, 0, 0.02228867108298849, 0.01087909501099718, -0.0002497636062461807, -0.0002869345420918655, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00725463658353566, -0.0004241441967297746, 0, 0,
        0.02043235997617835,
        0.02289143292803418, 0, 0.01822658364148615, -0.000851455689718699,
        0.01511366164838846, 0, 0, 0.003406747109574633, 0.02108914384657858, 0.004635681037954684, 0, 0.01521795223447008,
        0.007858889129924454, 0, 0.02209274391205599, 0.01016522408207408, 0, 0, 0, 0, 0.008128892029464959, 0.006436322792163481, 0.02267830650897435, 0.014910040314477, -0.0008650392815979342, -0.0009022102174436189, 0.002727587741968511, -0.0005457739511476377,
        0.01948715928351234, 0.0009583100573257223,
        0.004171520369862856, 0, 0, 0, 0.009432528150727938, 0, 0, 0, 0.01428302755087084, 0, 0.009701492403765126, 0.006284278803157167, 0.01524056542306919, 0, 0.02127192318248277, 0.02126267902109579, 0, 0, 0, 0, -0.001414695822906869, 0, 0, 0, -0.0006952151234696101, 0, 0, 0, -0.001105874790727093, 0, 0, 0,
        0.01742074075534553, 0.004257077418715998,
        0.008733160290593308, 0.001658226635653925, 0.02178132941921542, 0,
        0.01631457008241752,
        0.01790528008539668, 0.009304756862027495, -0.001375191263159938, 0.002491935990709242,
        0.006250234427636975, 0.01867207450361046,
        0.01780095441556682, 0.01342522420472558,
        0.0181694804759518, 0.009673016565344677,
        0.009142626749623557, 0, 0, 0, 0.01435451054913756, 0, 0.01168934090442451, 0, 0, 0, 0, 0, 0.005074812048966111, 0, 0.01267349225629992, -0.001437333634787336,
        0.01373645213338972,
        0.01617712408301317, 0, 0, 0, 0.01340990550867907, -0.004275821977717628, 0.001482357256203301, 0,
        0, 0, 0.009052077346939717, 0.01090330320683135, 0, 0, 0, 0, 0, 0, 0.005098245059020249, 0.009312033959464018, -0.004042448895529994, 0, 0.009643379155080446, 0.01725992163458791,
        0.01262008785545702,
        0.01225240645788656,
        0.01228013677982428, -0.00308969413189335,
        0.01173778811466943, 0.01016337465283759,
        0.007731705637872936, 0.009278700253535499, 0.01356438611355865, 0.0147070653497445, -0.003954531629498777, -0.002806928383385902, 0, 0.01105751622086575,
        0, 0, -0.005355409202675855, 0.003955771913726066, 0, 0, 0, 0.004166271668881169, 0, 0.006954274903448629, -0.005384066136085221, 0.006625228280929289, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007076338852890007, 0.005717097808605223, 0.0149875503011289, 0, 0, 0.008187285859570988, 0.0127431524906184, 0, 0, -0.00220767782220901, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.001820891436741218, 0, 0, 0, 0, 0,
        -0.0006587072315434223, -0.004325506399169796, 0.01213134000541419, 0.002147524571983833, 0.004135024632062986, 0, 0.01357500452453212, 0.004943732320974344,
        0.01035408124412413,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0004911993639373012, 0.009420394417210914, -0.005841748804511894, 0.006303795729590181, 0.004842633316746738, 0.01460331756575529, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001654165793231211, 0.006952183945989408,
        0.009783249575601456,
        0.009142509839414895, 0, 0.003536889699980823, 0.01256136554412805,
        0, 0.005577471872807968, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007943471253337, 0.01298408708488334, -0.001039017039971373, 0.01187284969952187, 0.003872895912853333, 0, -0.004924573192460603, 0.003552515704648683, 0.0033738567556442, 0.01044970333739874, -0.006864783326224233, 0.008372782518141225, -0.002387607036979199, 0.00367727579718598, 0.004617431448074908, 0,
        -0.001898573626189399, 0.009856919273925553,
        0.008747477134906763,
        0.001964759393421337, 0.01118398558257042,
        0.00159996519672589, 0.01233001043830119,
        0.01230758221610171, 0.002327880697098037, 0.003333808293161403, -0.002972210189898365,
        -0.006671777910561982, 0.007707819414731939, -0.003615454739394619, -0.001974197498230626, 0.007350148256499722, 0.0115526946496258, -0.00571585757725445, 0.01028907689583457,
        0, -0.002492570277062747, 0.01112724259327089, 0, 0, -0.005076196691672892, 0.003078786210129259, -0.004516997950976405
    ]
    model_ctrs = catboost_model_ctrs_container(
        used_model_ctrs_count = 15,
        compressed_model_ctrs = [
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 14216163332699387099, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 14216163332699387099, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 14216163332699387099, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 14216163332699387101, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 14216163332699387101, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 14216163332699387101, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 16890222057671696978, base_ctr_type = "Counter", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [7],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 14216163332699387103, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 14216163332699387103, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 16890222057671696976, base_ctr_type = "Counter", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 14216163332699387072, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 14216163332699387072, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [10],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 14216163332699387074, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 14216163332699387074, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [10],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 3, check_value_equal = 0, value = 2)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 8661910707442234914, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = -0, scale = 15)
                ]
            )
        ],
        ctr_data = catboost_ctr_data(
            learn_ctrs = {
                8661910707442234914 :
                catboost_ctr_value_table(
                    index_hash_viewer = {15788301966321637888 : 0, 7852726667766477841 : 5, 11772109559350781439 : 6, 5942209973749353715 : 4, 18446744073709551615 : 0, 11840406731846624597 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2072654927021551577 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1608091623008396926 : 7, 8746426839392254111 : 2},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.10195e-44, count = 73), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [15, 73, 0, 2, 0, 2, 5, 0, 1, 0, 0, 1, 0, 1, 1, 0]
                ),
                14216163332699387072 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 8473802870189803490 : 2, 7071392469244395075 : 1, 18446744073709551615 : 0, 8806438445905145973 : 3, 619730330622847022 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 2.94273e-44, count = 61), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 12, 1, 5, 21, 61, 0, 1]
                ),
                14216163332699387074 :
                catboost_ctr_value_table(
                    index_hash_viewer = {2136296385601851904 : 0, 7428730412605434673 : 1, 9959754109938180626 : 3, 14256903225472974739 : 5, 8056048104805248435 : 2, 18446744073709551615 : 0, 12130603730978457510 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10789443546307262781 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.8026e-44, count = 73), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [20, 73, 0, 2, 0, 2, 1, 0, 0, 1, 0, 1, 1, 0]
                ),
                14216163332699387099 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 15379737126276794113 : 5, 18446744073709551615 : 0, 14256903225472974739 : 2, 18048946643763804916 : 4, 2051959227349154549 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7024059537692152076 : 6, 18446744073709551615 : 0, 15472181234288693070 : 1, 8864790892067322495 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.4013e-44, count = 58), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 4.2039e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0)],
                    ctr_total = [10, 58, 1, 6, 1, 5, 3, 6, 0, 4, 2, 0, 5, 0]
                ),
                14216163332699387101 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 13987540656699198946 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18089724839685297862 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10353740403438739754 : 2, 3922001124998993866 : 0, 13686716744772876732 : 1, 18293943161539901837 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 37), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 3.08286e-44, count = 20), catboost_ctr_mean_history(sum = 0, count = 13), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3)],
                    ctr_total = [0, 37, 0, 4, 22, 20, 0, 13, 0, 2, 0, 3]
                ),
                14216163332699387103 :
                catboost_ctr_value_table(
                    index_hash_viewer = {3607388709394294015 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18356215166324018775 : 0, 18365206492781874408 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 14559146096844143499 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 11416626865500250542 : 3, 5549384008678792175 : 2},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 14), catboost_ctr_mean_history(sum = 0, count = 22), catboost_ctr_mean_history(sum = 0, count = 22), catboost_ctr_mean_history(sum = 2.66247e-44, count = 17), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [0, 14, 0, 22, 0, 22, 19, 17, 2, 3, 1, 1]
                ),
                16890222057671696976 :
                catboost_ctr_value_table(
                    index_hash_viewer = {3607388709394294015 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18356215166324018775 : 0, 18365206492781874408 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 14559146096844143499 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 11416626865500250542 : 3, 5549384008678792175 : 2},
                    target_classes_count = 0,
                    counter_denominator = 36,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.96182e-44, count = 22), catboost_ctr_mean_history(sum = 3.08286e-44, count = 36), catboost_ctr_mean_history(sum = 7.00649e-45, count = 2)],
                    ctr_total = [14, 22, 22, 36, 5, 2]
                ),
                16890222057671696978 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 13987540656699198946 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18089724839685297862 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10353740403438739754 : 2, 3922001124998993866 : 0, 13686716744772876732 : 1, 18293943161539901837 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 42,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 5.1848e-44, count = 4), catboost_ctr_mean_history(sum = 5.88545e-44, count = 13), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3)],
                    ctr_total = [37, 4, 42, 13, 2, 3]
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



