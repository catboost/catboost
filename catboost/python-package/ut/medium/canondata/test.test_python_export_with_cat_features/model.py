from cityhash import CityHash64  # Available at https://github.com/Amper/cityhash

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



##  Model data
class catboost_model(object):
    float_features_count = 6
    cat_features_count = 11
    binary_feature_count = 23
    tree_count = 20
    float_feature_borders = [
        [45.5],
        [121010, 168783.5, 198094.5, 200721, 209752.5, 332801, 350449],
        [4.5, 8, 9.5, 13.5],
        [1087, 3280, 7493, 11356, 17537.5],
        [1738, 1881.5, 2189.5],
        [17, 31.5, 35.5, 36.5, 49]
    ]
    tree_depth = [3, 0, 1, 2, 2, 0, 6, 2, 5, 4, 6, 2, 0, 3, 2, 4, 0, 5, 6, 4]
    tree_split_border = [2, 3, 3, 1, 1, 5, 1, 4, 255, 1, 2, 1, 1, 4, 3, 2, 1, 1, 4, 2, 5, 4, 2, 1, 3, 2, 3, 1, 3, 1, 1, 3, 1, 2, 2, 3, 1, 5, 4, 1, 1, 2, 1, 1, 1, 1, 7, 1, 3, 1, 1, 2, 6, 2, 2, 1, 3]
    tree_split_feature_index = [4, 5, 4, 9, 22, 1, 2, 2, 6, 4, 5, 10, 17, 1, 10, 10, 5, 20, 15, 11, 3, 3, 15, 16, 5, 2, 2, 4, 20, 21, 2, 20, 15, 19, 11, 15, 0, 5, 5, 1, 12, 1, 11, 14, 18, 13, 1, 3, 1, 7, 8, 20, 1, 3, 21, 19, 3]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    one_hot_cat_feature_index = [9]
    one_hot_hash_values = [
        [-1291328762]
    ]
    ctr_feature_borders = [
        [1.999999046325684],
        [8.999999046325684],
        [12.99999904632568],
        [4.999999046325684, 5.999999046325684, 9.999999046325684],
        [6.999999046325684, 9.999999046325684],
        [5.999999046325684],
        [10.99999904632568],
        [11.99999904632568],
        [5.999999046325684, 9.999999046325684, 10.99999904632568, 13.99999904632568],
        [1.999999046325684],
        [9.999999046325684],
        [0.9999989867210388],
        [2.999999046325684, 3.999999046325684],
        [2.999999046325684, 9.999999046325684, 12.99999904632568],
        [5.999999046325684, 8.999999046325684],
        [6.999999046325684]
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.02538461481722502, 0, 0.02181818133050745, 0.00599999986588955, 0, 0, 0, 0,
        0.02215084809364415,
        0.02172708936280266, -0.0003297677133102541,
        0.02152793739154444, 0.01778877082128263, 0.01850400474626319, 0.02067978983786623,
        0.01567963725707506, 0.0220112634798453, 0, 0.005536589382590208,
        0.01974820773748438,
        0, 0, 0, 0, 0.00668422720013354, 0.006656183451483403, 0, -0.00121032712489911, 0.01630675349288336, 0.01040940023674122, 0, 0, 0.01853636369398814, 0.02130145929132601, 0.004606578054190895, -0.0006521362759071482, 0, 0, 0, 0, 0, -0.001305236211565434, 0, 0, 0, 0, 0, 0, 0, 0.009936563966338845, 0, 0, 0, 0, 0, 0, 0, 0.00668422720013354, 0, 0, 0.01739248942175277, 0.01489443623183825, 0, 0, 0.01961891635677588, 0.02056829642804693, 0, 0.006781180916326713, 0, 0, 0, 0, 0, -0.001674910548736258, 0, 0, 0, 0, 0, 0, 0, 0.01015745703339937, 0, 0,
        0.004397862157920605, 0, 0.0173169712303545, 0.01995834279965041,
        0, 0.006573278111265951, 0, 0.01056177845088946, 0, -0.0008389673263076185, 0, -0.002210365106577666, 0, 0, 0.01252797537745427, 0.02368052087424731, 0, 0.008373865623709641, 0, 0.01177294973233899, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.001190292731481934,
        0.006172078091740506, 0, 0.006076093315588495, -0.001181365536195358, 0.009704954610537917, 0, 0.02094713146553188, 0, 0.01213569811646915, 0, 0.006267968516740491, -0.001668938697132199, 0.02077356302989403, 0, 0.01404276168665576, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01194789205007901, 0, 0, 0, 0, 0, 0, 0, 0.00618866801776118, 0, 0, 0, 0, 0, 0, 0, 0.006030522616740166, 0.005918989833108543, 0, 0.00956997428119615, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01731715780704228, 0.0200484977359294, 0, 0.02034553074602548, 0, 0.003998183659885355, 0, 0, 0.009791197088019349, 0.01081763758489084, 0, 0.004753520117275608, 0, 0, 0, 0.002113073546331681,
        0.02069773494014405, 0, 0.01995277383978061, 0.008226800289584996,
        0.01642145822000003,
        0, 0.01187160900013913, 0, 0.01916886602566633, 0, -0.003213923551218401, 0.003200190713244948, 0.01618175634200034,
        0.01857021431146683, 0.011534044748459, 0.007379608573513029, 0.003764586258426566,
        0, 0, 0.005451847189699659, 0.005352498027267781, 0.01219957601974083, 0.01240237014666204, 0.005981022937292964, 0.009931963439227415, 0, 0, 0.00558713536217078, 0.01389804493124376, 0, 0, 0.01764949412991807, 0.01233841871780235,
        0.01471105694954564,
        0, 0.002625700658516242, 0, 0.009724031707401531, -0.003376025787578505, 0.006818563531257435, 0, 0.005202021368305096, 0, 0, 0, 0.009424305756923209, 0, 0, 0.01111812876323236, 0.01885070155215754, 0, 0.005795377395515639, 0, 0.005044936656386488, 0, 0.008755366118293646, 0, 0, 0, 0, 0, 0.00509932973029755, 0, 0, 0, 0.01410926625216536,
        -0.001874681421018266, 0, 0, 0, 0, 0, 0.004810577060605463, 0, 0.009180949868100793, -0.001543919024798, 0, 0, 0.01823204901206538, 0.005108850007640363, 0.0166589057282016, 0, -0.002717896811328025, 0, -0.00206289377896663, -0.001865102449209582, 0, 0, 0, 0, 0, 0, -0.002108374658957168, 0, 0.009355673941159435, -0.003960394503023597, 0.007698807248080879, -0.002161145929500197, 0, 0, 0.005017273182540766, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01553680271461355, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.008283936590973833, 0,
        0, 0, 0.005173900140458494, 0, 0.01430852627013582, 0, 0.0143925650164066, -0.003323831052914757, 0, 0, 0, 0, 0, 0, 0, -0.00412862012170337
    ]
    model_ctrs = catboost_model_ctrs_container(
        used_model_ctrs_count = 16,
        compressed_model_ctrs = [
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 14216163332699387099, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 14216163332699387099, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 14216163332699387100, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 14216163332699387100, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 14216163332699387100, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = -0, scale = 15)
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
                    transposed_cat_feature_indexes = [5],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 3, check_value_equal = 0, value = 4)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 12923321341810884916, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 4, check_value_equal = 0, value = 1),
                        catboost_bin_feature_index_value(bin_index = 5, check_value_equal = 0, value = 2),
                        catboost_bin_feature_index_value(bin_index = 6, check_value_equal = 1, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 8875426491869161292, base_ctr_type = "Counter", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [6],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 14216163332699387102, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 14216163332699387102, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [7],
                    binarized_indexes = []
                ),
                model_ctrs = [
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
                    catboost_model_ctr(base_hash = 16890222057671696975, base_ctr_type = "Counter", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15)
                ]
            )
        ],
        ctr_data = catboost_ctr_data(
            learn_ctrs = {
                14216163332699387072 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 8473802870189803490 : 2, 7071392469244395075 : 1, 18446744073709551615 : 0, 8806438445905145973 : 3, 619730330622847022 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 2.94273e-44, count = 61), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 12, 1, 5, 21, 61, 0, 1]
                ),
                16890222057671696975 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 8473802870189803490 : 2, 7071392469244395075 : 1, 18446744073709551615 : 0, 8806438445905145973 : 3, 619730330622847022 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 82,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.68156e-44, count = 6), catboost_ctr_mean_history(sum = 1.14906e-43, count = 1)],
                    ctr_total = [12, 6, 82, 1]
                ),
                16890222057671696976 :
                catboost_ctr_value_table(
                    index_hash_viewer = {3607388709394294015 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18356215166324018775 : 0, 18365206492781874408 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 14559146096844143499 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 11416626865500250542 : 3, 5549384008678792175 : 2},
                    target_classes_count = 0,
                    counter_denominator = 36,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.96182e-44, count = 22), catboost_ctr_mean_history(sum = 3.08286e-44, count = 36), catboost_ctr_mean_history(sum = 7.00649e-45, count = 2)],
                    ctr_total = [14, 22, 22, 36, 5, 2]
                ),
                8875426491869161292 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 12388642741946772418 : 2, 15554416449052025026 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10159611108399828747 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17281877160932429072 : 7, 4069608767902639184 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10879563727165159958 : 4, 18446744073709551615 : 0, 3893644761655887960 : 1, 18446744073709551615 : 0, 3836326012375860634 : 17, 1838769409751938715 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5822126489489576543 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 8894837686808871138 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8800705802312104489 : 10, 7657545707537307113 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15610660761574625263 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10553223523899041848 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 13625934721218336443 : 5, 8407093386812891388 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 17609291800955974271 : 8},
                    target_classes_count = 0,
                    counter_denominator = 31,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.26117e-44, count = 15), catboost_ctr_mean_history(sum = 2.8026e-45, count = 31), catboost_ctr_mean_history(sum = 7.00649e-45, count = 4), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 1.26117e-44, count = 1), catboost_ctr_mean_history(sum = 5.60519e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 8), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [9, 15, 2, 31, 5, 4, 3, 1, 9, 1, 4, 2, 1, 2, 2, 8, 1, 1]
                ),
                16890222057671696978 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 13987540656699198946 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18089724839685297862 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10353740403438739754 : 2, 3922001124998993866 : 0, 13686716744772876732 : 1, 18293943161539901837 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 42,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 5.1848e-44, count = 4), catboost_ctr_mean_history(sum = 5.88545e-44, count = 13), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3)],
                    ctr_total = [37, 4, 42, 13, 2, 3]
                ),
                12923321341810884916 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 3581428127016485793 : 3, 1236773280081879954 : 2, 16151796118569799858 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13125720576600207402 : 5, 5967870314491345259 : 4, 9724886183021484844 : 1, 18446744073709551615 : 0, 13605281311626526238 : 6, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 37), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 2.66247e-44, count = 20), catboost_ctr_mean_history(sum = 0, count = 13), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3)],
                    ctr_total = [0, 37, 0, 4, 19, 20, 0, 13, 3, 0, 0, 2, 0, 3]
                ),
                14216163332699387099 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 15379737126276794113 : 5, 18446744073709551615 : 0, 14256903225472974739 : 2, 18048946643763804916 : 4, 2051959227349154549 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7024059537692152076 : 6, 18446744073709551615 : 0, 15472181234288693070 : 1, 8864790892067322495 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.4013e-44, count = 58), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 4.2039e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0)],
                    ctr_total = [10, 58, 1, 6, 1, 5, 3, 6, 0, 4, 2, 0, 5, 0]
                ),
                14216163332699387100 :
                catboost_ctr_value_table(
                    index_hash_viewer = {7537614347373541888 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5903587924673389870 : 1, 18278593470046426063 : 6, 10490918088663114479 : 8, 18446744073709551615 : 0, 407784798908322194 : 10, 5726141494028968211 : 3, 1663272627194921140 : 0, 8118089682304925684 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15431483020081801594 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 1403990565605003389 : 2, 3699047549849816830 : 11, 14914630290137473119 : 7},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 5.60519e-45, count = 24), catboost_ctr_mean_history(sum = 4.2039e-45, count = 16), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 5.60519e-45, count = 16), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 9.80909e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0)],
                    ctr_total = [0, 3, 4, 24, 3, 16, 1, 3, 4, 16, 1, 1, 0, 4, 7, 3, 0, 3, 0, 1, 0, 5, 2, 0]
                ),
                14216163332699387101 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 13987540656699198946 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18089724839685297862 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10353740403438739754 : 2, 3922001124998993866 : 0, 13686716744772876732 : 1, 18293943161539901837 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 37), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 3.08286e-44, count = 20), catboost_ctr_mean_history(sum = 0, count = 13), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3)],
                    ctr_total = [0, 37, 0, 4, 22, 20, 0, 13, 0, 2, 0, 3]
                ),
                14216163332699387102 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 14452488454682494753 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1388452262538353895 : 5, 8940247467966214344 : 9, 4415016594903340137 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 41084306841859596 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8678739366408346384 : 4, 18446744073709551615 : 0, 4544226147037566482 : 12, 14256903225472974739 : 6, 16748601451484174196 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5913522704362245435 : 0, 1466902651052050075 : 3, 2942073219785550491 : 8, 15383677753867481021 : 2, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 11), catboost_ctr_mean_history(sum = 2.8026e-45, count = 9), catboost_ctr_mean_history(sum = 2.8026e-45, count = 14), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 9.80909e-45, count = 6), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 8.40779e-45, count = 10), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 8), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [0, 11, 2, 9, 2, 14, 0, 2, 0, 6, 7, 6, 1, 5, 6, 10, 0, 1, 2, 8, 0, 3, 1, 4, 1, 0]
                )
            }
        )
    )

# Routines to compute CTRs

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



def city_hash_uint64(string):
    out = CityHash64(string) & 0xffffffff
    if (out > 0x7fFFffFF):
        out -= 0x100000000
    return out


# Applicator for the CatBoost model

def apply_catboost_model(float_features, cat_features):
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

    model = catboost_model

    assert len(float_features) == model.float_features_count
    assert len(cat_features) == model.cat_features_count

    # Binarise features
    binary_features = [0] * model.binary_feature_count
    binary_feature_index = 0

    for i in range(len(model.float_feature_borders)):
        for border in model.float_feature_borders[i]:
            binary_features[binary_feature_index] += 1 if (float_features[i] > border) else 0
        binary_feature_index += 1
    transposed_hash = [0] * model.cat_features_count
    for i in range(model.cat_features_count):
        transposed_hash[i] = city_hash_uint64(cat_features[i])

    if len(model.one_hot_cat_feature_index) > 0:
        cat_feature_packed_indexes = {}
        for i in range(model.cat_features_count):
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
    for tree_id in range(model.tree_count):
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



