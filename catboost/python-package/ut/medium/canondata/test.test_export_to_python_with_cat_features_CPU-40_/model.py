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
    binary_feature_count = 24
    tree_count = 40
    float_feature_borders = [
        [18.5, 19.5, 34.5, 68.5, 71],
        [126119, 200721, 215061, 231641.5, 281044.5, 337225.5, 553548.5],
        [5.5, 6.5, 9.5, 12.5, 13.5, 14.5],
        [1087, 3280, 5842, 7493, 11356, 17537.5],
        [808.5, 1622.5, 1738, 1862, 1881.5, 1944.5, 2396],
        [36.5, 42, 70]
    ]
    tree_depth = [3, 6, 6, 6, 3, 3, 4, 0, 0, 0, 4, 2, 1, 5, 5, 6, 0, 3, 4, 5, 1, 1, 4, 6, 1, 6, 1, 0, 0, 2, 2, 2, 2, 1, 1, 0, 1, 0, 1, 1]
    tree_split_border = [5, 4, 6, 1, 1, 4, 8, 7, 2, 4, 3, 1, 6, 1, 1, 2, 6, 1, 6, 5, 1, 4, 2, 2, 7, 4, 2, 5, 3, 4, 5, 6, 1, 4, 1, 3, 1, 2, 3, 3, 2, 255, 3, 4, 1, 7, 5, 6, 7, 3, 2, 2, 8, 6, 2, 1, 1, 2, 1, 1, 3, 2, 3, 1, 1, 1, 2, 7, 1, 4, 3, 5, 1, 6, 2, 3, 4, 1, 8, 4, 5, 3, 1, 1, 1, 1, 3, 9, 5, 1, 2, 2, 4, 3, 5, 2, 7, 4, 5]
    tree_split_feature_index = [3, 4, 7, 8, 0, 1, 13, 7, 10, 13, 1, 4, 4, 3, 5, 12, 1, 14, 2, 1, 21, 14, 7, 3, 4, 0, 13, 3, 11, 12, 2, 3, 22, 7, 10, 0, 11, 20, 20, 5, 18, 6, 7, 2, 18, 4, 7, 13, 1, 2, 21, 2, 13, 3, 1, 14, 15, 11, 7, 21, 14, 5, 3, 13, 19, 12, 8, 13, 4, 13, 13, 3, 17, 3, 0, 14, 3, 23, 13, 3, 0, 4, 16, 9, 1, 20, 14, 13, 7, 2, 4, 14, 2, 12, 13, 17, 13, 14, 4]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    one_hot_cat_feature_index = [9]
    one_hot_hash_values = [
        [-2114564283]
    ]
    ctr_feature_borders = [
        [0.999998987, 1.99999905, 3.99999905, 6.99999905, 8.99999905, 10.999999, 12.999999],
        [2.99999905, 6.99999905],
        [8.99999905],
        [6.99999905, 8.99999905],
        [0.999998987, 7.99999905, 12.999999],
        [6.99999905, 8.99999905, 10.999999, 12.999999],
        [6.99999905, 7.99999905, 8.99999905, 9.99999905, 10.999999, 11.999999, 12.999999, 13.999999, 14.999999],
        [0.999998987, 1.99999905, 12.999999, 13.999999],
        [6.99999905],
        [11.999999],
        [12.999999, 13.999999],
        [2.99999905, 4.99999905],
        [10.999999],
        [10.999999, 11.999999, 14.999999],
        [0.999998987, 1.99999905],
        [12.999999],
        [5.99999905]
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.01999999955296516, 0, 0.004999999888241291, 0, 0.0247826081417177, 0, 0.00599999986588955, 0,
        0, 0, -0.0001499999932944775, 0.003470380363463545, 0, 0, -0.0001499999932944775, 0.01156760844676067, 0, 0.007314130275453562, 0, 0.02304875726047787, 0, 0.007314130275453562, 0, 0.0117026084407257, 0, 0, 0, -4.499999798834327e-05, 0, 0, 0, -0.0001858695569083744, 0, 0, 0, 0.01462826055090712, 0, 0, 0, 0, 0, 0, -3.749999832361938e-05, 0.01346434753710809, 0, 0, 0, 0.01247329165524577, 0, 0, 0, 0.02416112479182612, 0, 0, 0, 0.02413580509986337, 0, 0, 0, 0.009504347624726918, 0, 0, 0, 0.007314130275453562, 0, 0.007314130275453562, 0, 0.02149234735704245, 0, 0.007314130275453562, 0, 0.007314130275453562,
        0, 0.02203329959438511, 0.01445557903672388, 0.02090051891739597, 0, 0.00729380723279871, 0, 0.007259274299613786, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0001611926015748829, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01202224912856518, 0.02519934044647189, 0.009296993208359132, 0.02221987841480803, -0.000205247925650846, 0, 0, 0.007220580590130218, 0, 0, 0, 0, -8.059727904824544e-05, 0, 0, 0.007375742772626784, -0.0002868521611795479, 0.003876930736150383, -0.0002571521625005358, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0.01131705219977818, 0.009131405919731996, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0003064845022101746, 0, 0, 0.00705971162972376, 0, 0, 0, 0, 0, 0, 0, 0, 0.00694392679444102, 0.006976014856599559, 0, 0, 0.01327883733386207, 0.02343944511749118, 0, 0, 0, 0, 0, 0, -0.0003242219493700392, 0, 0, 0, 0, 0, 0, 0, 0.004397972187707204, 0.01173991380755557, 0.01624559285904699, 0.02105239301807314, 0, 0, 0, 0, 0, 0, 0, 0.007320424703068537,
        0.01713816872353237, -0.00101015110586199, 0.02594558349416469, 0.01660501053748116, 0, -0.0004294048680491817, 0, -0.0008169148706745303,
        0.01079041022645152, 0, 0, 0, 0.0225548573429452, -0.000185029282570031, 0.005030479022316899, 0,
        0.01822020916108734, -0.0006713420243358168, 0, 0, 0.01880095670709041, 0, 0.02380558714888851, 0, 0.001016917894030677, -0.0003968415563199083, 0, 0, 0.006448589981753908, 0, 0.01031087564815217, 0,
        0.01898745826649682,
        0.01843426598513581,
        0.01789719074776575,
        0.003626164206099096, 0, 0.004123476059396431, 0, 0.01456064847424626, 0, 0.01391736939318713, 0, 0.007175222476750585, 0, 0, 0, 0.01974902256669395, 0, 0.01185448186361029, -0.0007772380101248539,
        0.01185945775793138, 0.01195241593405586, 0.02004545284761915, 0.01234895501775566,
        0.01527028129781319, 0.01721777215972029,
        0, 0.005793334092104274, 0, 0, -0.003267594832604089, 0, 0, 0, 0, 0.005832489972446285, 0, 0, 0.007280922462546561, 0.005710425653552725, 0, 0, 0, 0.005621030028584721, 0, 0, 0.01494774093086465, 0.01422939839545966, -0.0009225094290911208, 0, 0, 0.005592167556776282, 0, 0, 0.01925234309298835, 0.01503979076092118, 0, 0,
        0, 0, 0.0003863817991255305, 0.0007474773296608814, 0, 0, 0, 0, 0, 0, 0.01311749469323431, -0.0007109599610244532, 0, 0, 0, -0.0009155906085275851, 0.01139257112470219, 0, 0.01389686556604557, 0.01032920541091394, 0, 0, 0, 0, 0, 0.005578872304312634, 0.01982624716374476, 0.01629702279451762, 0, 0, 0, 0,
        0, 0, 0, 0, 0.005612829102526628, 0, 0, 0, 0.005669068941309504, 0, -0.002606908836528224, 0, 0.01061311920064227, 0, 0.004480053186381886, 0, 0.00896671715823245, 0, 0, 0, 0.01498684932222408, 0, 0, 0, 0.005464782019512482, 0, 0.005297583763136753, 0, 0.01783260039940101, 0.008305039985800648, 0.01933907115132087, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.001222107327861213, 0,
        0.0148392595791752,
        0, 0, 0.004105107418691836, 0, 0.008163211477585839, 0, 0.0151995635234467, 0.01444822488487993,
        0.005410760471687652, 0, 0.0083333543295834, 0.00503256071756729, 0.008415827235429293, 0.008226817711936436, 0.01272815382610973, 0.01839492637265765, 0, 0, 0.002841451847506272, -0.001809399723544643, -0.003171913592586823, -0.003547323033035719, 0.008179852565058206, 0.01042743776402821,
        0.005796650788699248, 0.002692987287766075, 0, 0, 0, 0.005493297219148043, -0.001813419916642705, 0, 0.002869765075874072, 0, 0, 0, 0.008596719468130386, 0, 0, 0, 0, 0, 0, 0, 0.01723908915402649, 0.006092324227009916, 0, 0, 0, 0, 0, 0, 0.01549819505598307, 0.006372982395348424, -0.003046820126714731, -0.001364968436561024,
        0.001096351010433316, 0.014152553874154,
        0.005603526527597412, 0.01722920563235135,
        0.005196358775439976, -0.003340520693045765, 0, 0, 0.007089145630978554, -0.001441788993578166, 0.01610700788538342, 0.01137615551467872, 0, 0, 0, 0, 0, 0, -0.003011475138625672, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.001382394190892858, 0, 0, 0, 0, 0.01143028454967799, 0, 0, 0.004443048908259466, 0.01654964369981381, 0, 0, 0, 0, 0, 0, 0.006519587515475378, 0.001982080052685937, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.002041951231623278, -0.001377552766169704, 0, -0.001689507561771382,
        0.005107116137383578, 0.01585163020787507,
        -0.00142395170697102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00881514938107701, -0.001405524490828592, 0, 0, -0.00183492725007898, 0, 0, 0, 0.01212449211578207, 0, 0.004336758318876998, 0, 0, 0, 0, 0, -0.001607402132489359, -0.002064939967902643, 0, 0, 0, 0, 0, 0, 0.007098120247080991, 0, 0, 0, 0, 0, 0, 0, 0.01004718114540055, -0.002564842467288145, 0, 0, 0.002477421733919167, 0, 0, 0, 0.01491703120771285, 0, 0, 0, 0, 0, 0, 0,
        0.01145660192587347, 0.01076895500564115,
        0.01090754352091524,
        0.01058975644274552,
        0, 0.003856719423080162, 0.01443359608522935, 0.001705163252909523,
        0.008127758385080027, 0.009034179229127411, 0.004592230418904769, 0.01066678917785803,
        0.0109106895535775, 0.003980974553640618, 0.009769372557730071, 0.0005892245494980584,
        0.007979979713054886, -0.002690460576320364, 0.01288174062248397, 0.0111351686241983,
        0.002966763715415119, 0.01235475487698945,
        0.003872195066324391, 0.01178306226981817,
        0.008673887277552682,
        0.002668892048570584, 0.01189125063927196,
        0.008181141734920867,
        0.01162672140433912, 0.002337625654666836,
        0.008095517950130804, -0.001003802595541243
    ]
    model_ctrs = catboost_model_ctrs_container(
        used_model_ctrs_count = 17,
        compressed_model_ctrs = [
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 14216163332699387099, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 14216163332699387099, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 3, check_value_equal = 0, value = 4),
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 5),
                        catboost_bin_feature_index_value(bin_index = 4, check_value_equal = 0, value = 3)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 425524955817535461, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 16890222057671696979, base_ctr_type = "Counter", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15)
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
                        catboost_bin_feature_index_value(bin_index = 1, check_value_equal = 0, value = 2)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 692698791827290762, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [7],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 14216163332699387103, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 14216163332699387103, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 14216163332699387072, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 14216163332699387072, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 14216163332699387072, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = -0, scale = 15)
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
                        catboost_bin_feature_index_value(bin_index = 3, check_value_equal = 0, value = 6),
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 2),
                        catboost_bin_feature_index_value(bin_index = 3, check_value_equal = 0, value = 4)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 13033542383760369867, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = -0, scale = 15)
                ]
            )
        ],
        ctr_data = catboost_ctr_data(
            learn_ctrs = {
                425524955817535461 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 12653652018840522049 : 4, 15085635429984554305 : 8, 14668331998361267939 : 0, 6371181315837451172 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 2365187170603376679 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6366972096803066445 : 11, 7160913711096172174 : 9, 725004896848258735 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12915814276774330580 : 3, 12850586852799469109 : 13, 14440651936148345046 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8119489818390003804 : 10, 8913431432683109533 : 12, 952684959061181628 : 6, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 8.40779e-45, count = 57), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 4.2039e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [6, 57, 0, 5, 1, 5, 2, 0, 3, 6, 0, 3, 2, 1, 1, 1, 0, 1, 1, 0, 3, 0, 1, 0, 1, 0, 1, 0]
                ),
                692698791827290762 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 3581428127016485793 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14455983217430950149 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13125720576600207402 : 8, 5967870314491345259 : 6, 9724886183021484844 : 1, 2436149079269713547 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1236773280081879954 : 2, 16151796118569799858 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8312525161425951098 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13605281311626526238 : 9, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 18), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.68156e-44, count = 8), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 1.4013e-44, count = 12), catboost_ctr_mean_history(sum = 0, count = 19), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3)],
                    ctr_total = [0, 18, 0, 2, 12, 8, 0, 7, 0, 2, 0, 6, 10, 12, 0, 19, 0, 2, 0, 3]
                ),
                13033542383760369867 :
                catboost_ctr_value_table(
                    index_hash_viewer = {8127566760675494400 : 3, 16133203970344820352 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 17493291581550525284 : 2, 11079641284750479812 : 8, 3155078433189509382 : 7, 1373856113935573863 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 15470940414085713834 : 1, 8124029294275766379 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3126373835522511222 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16842020874822597533 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.38221e-44, count = 67), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [17, 67, 0, 2, 0, 2, 0, 6, 1, 0, 2, 0, 1, 0, 0, 1, 0, 1, 1, 0]
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
                16890222057671696978 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 13987540656699198946 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18089724839685297862 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10353740403438739754 : 2, 3922001124998993866 : 0, 13686716744772876732 : 1, 18293943161539901837 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 42,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 5.1848e-44, count = 4), catboost_ctr_mean_history(sum = 5.88545e-44, count = 13), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3)],
                    ctr_total = [37, 4, 42, 13, 2, 3]
                ),
                16890222057671696979 :
                catboost_ctr_value_table(
                    index_hash_viewer = {7537614347373541888 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5903587924673389870 : 1, 18278593470046426063 : 6, 10490918088663114479 : 8, 18446744073709551615 : 0, 407784798908322194 : 10, 5726141494028968211 : 3, 1663272627194921140 : 0, 8118089682304925684 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15431483020081801594 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 1403990565605003389 : 2, 3699047549849816830 : 11, 14914630290137473119 : 7},
                    target_classes_count = 0,
                    counter_denominator = 28,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 4.2039e-45, count = 28), catboost_ctr_mean_history(sum = 2.66247e-44, count = 4), catboost_ctr_mean_history(sum = 2.8026e-44, count = 2), catboost_ctr_mean_history(sum = 5.60519e-45, count = 10), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 7.00649e-45, count = 2)],
                    ctr_total = [3, 28, 19, 4, 20, 2, 4, 10, 3, 1, 5, 2]
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



