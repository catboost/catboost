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
    binary_feature_count = 25
    tree_count = 40
    float_feature_borders = [
        [19.5, 45.5, 57.5],
        [167392.5, 208500.5, 218145.5, 313025.5],
        [6.5, 10.5, 13.5],
        [1087, 3280, 7493, 11356],
        [1738, 1881.5, 2189.5],
        [19, 44.5, 46.5, 49]
    ]
    tree_depth = [4, 4, 4, 3, 3, 4, 3, 1, 6, 0, 0, 3, 1, 2, 4, 0, 6, 0, 0, 3, 4, 5, 2, 6, 2, 6, 2, 1, 3, 0, 2, 3, 1, 3, 1, 0, 1, 1, 0, 2]
    tree_split_border = [1, 3, 4, 6, 5, 2, 3, 2, 1, 2, 5, 2, 3, 1, 2, 2, 2, 2, 1, 6, 1, 3, 1, 1, 2, 3, 4, 3, 1, 3, 1, 2, 2, 3, 2, 3, 6, 2, 2, 3, 4, 3, 2, 4, 3, 2, 4, 2, 4, 3, 4, 2, 1, 1, 4, 4, 5, 4, 3, 1, 1, 5, 3, 2, 1, 1, 3, 1, 4, 1, 2, 4, 2, 1, 1, 1, 2, 4, 6, 4, 1, 1, 7, 4, 1, 1, 2, 1, 2, 1, 2, 3, 7, 3, 2, 2]
    tree_split_feature_index = [3, 4, 3, 6, 21, 5, 6, 4, 4, 0, 6, 10, 2, 4, 18, 4, 7, 21, 1, 21, 9, 18, 12, 18, 3, 2, 21, 7, 0, 3, 10, 6, 19, 1, 9, 21, 15, 2, 3, 8, 5, 19, 7, 3, 0, 1, 5, 15, 7, 10, 15, 17, 21, 19, 1, 14, 8, 3, 19, 24, 14, 15, 5, 18, 6, 15, 15, 22, 14, 8, 4, 6, 3, 11, 20, 17, 8, 8, 15, 14, 2, 23, 15, 7, 5, 13, 12, 7, 18, 16, 8, 14, 15, 4, 14, 7]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    one_hot_cat_feature_index = []
    one_hot_hash_values = [
    ]
    ctr_feature_borders = [
        [0.999998987, 1.99999905, 3.99999905, 7.99999905, 8.99999905, 10.999999],
        [1.99999905, 3.99999905, 5.99999905, 12.999999],
        [3.99999905, 4.99999905, 7.99999905, 8.99999905, 9.99999905],
        [10.999999, 12.999999],
        [5.99999905, 8.99999905, 9.99999905],
        [5.99999905],
        [9.99999905, 10.999999],
        [6.99999905],
        [8.99999905, 10.999999, 12.999999, 14.999999],
        [2.99999905, 5.99999905, 7.99999905, 10.999999, 11.999999, 12.999999, 13.999999],
        [12.999999],
        [2.99999905, 12.999999],
        [2.99999905, 8.99999905, 10.999999],
        [1.99999905, 4.99999905, 7.99999905],
        [9.99999905],
        [0.999998987, 2.99999905, 3.99999905, 6.99999905, 7.99999905, 13.999999],
        [5.99999905],
        [5.99999905],
        [6.99999905]
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.01999999955296516, 0.007499999832361937, 0, 0, 0, 0, 0, 0, 0.02558823472217602, 0, 0, 0, 0, 0, 0, 0,
        0.01175999974250793, 0.01052727250050415, 0, -0.0001999999910593034, 0.02339779360812595, 0.02482017149086524, 0, 0.01591102906929, 0, 0, 0, 0, 0, 0, 0, 0.004772058721960467,
        0.00727104529707845, 0, 0.00337329538745213, 0, 0, 0, 0, 0, 0.01920425557155548, 0, 0.01176522696351214, 0.007314209399452734, 0.02416195997085191, -0.0004311573337764974, 0.01837508512323088, 0.005668917083679816,
        0.02660164906914413, 0.01029058144199528, 0.005663535933629778, 0, 0.01388382919908924, -0.0008798325473358468, 0, 0.0037991806891614,
        0, 0, 0.02221365900106278, 0, 0.003057396215551764, -4.247651855279913e-05, 0.0208357883877005, 0.004580730289323383,
        0.0177676995127538, 0.01665647711433886, 0.008488333600305155, 0.02182628839249728, 0.01066902785992093, 0.01317055026004484, 0.02111902255995688, 0.02271955636349854, 0, 0.004722583587441623, -0.0004571558986205873, 0.0086016263064681, 0, 0.0067240603373029, 0.0101001543574297, 0.009297037110251066,
        0.009337141719259394, 0.006670991987717867, 0.01556157202571188, 0.02442179794494355, 0, -0.0004643040113271255, -0.001428868156373164, 0,
        0.02063374026640057, 0.005310179839258373,
        0, 0, 0, 0, 0, -0.0008737859232070884, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004697871563523307, 0.01020639038100292, 0.006299288824691354, 0, 0, 0, 0, 0, -0.0003594076361730832, 0, -0.0005006480792242566, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.006114100857623254, 0.01520692407697443, 0, 0.006359117997278379, 0.0194824680979193, 0.01910641439582282, 0, 0, 0, 0, 0, 0, 0, -0.001316541324195777,
        0.01785100500998277,
        0.01733092285641282,
        0.01362210791393915, 0.01564412373335999, 0, 0.01694396154384863, 0, -0.0009267696960860561, 0, 0.009149846609688895,
        0.01848367515080994, 0.01524310971272575,
        0.02050462604828848, 0.01226047406753023, 0.01956529552946539, 3.607003330176642e-05,
        0, 0, 0.01557196416287685, 0, 0, 0, 0, 0, 0.00408371146007968, 0, 0.01850888948614398, -0.003322282424949401, -0.0008324767314414416, -0.0008525013029736559, 0.007602816895440651, 0,
        0.01504561537293027,
        0, 0.0147366878679351, 0, 0, 0, 0.01206911710107181, 0, 0, 0, 0.01421391442828344, 0, 0, 0, 0.0051216768795085, 0, 0, 0, 0.008443359011477329, 0, 0, 0, 0.005468688204778539, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.001976413361722452, 0.01625331972067124, 0, -0.001080607437870202, 0, 0.003075263007677012, 0, -0.00137798939648387, 0.003523713929683426, 0.01541603939670028, 0, 0, 0, -0.001809692375124854, 0, 0, -0.0009390752688699447, 0.002025285672893997, 0, 0, 0, 0, -0.0009589496561190243, 0, 0, 0.001601057357002395, 0, 0, 0, 0.005221679716313761, 0, 0,
        0.01425474566556058,
        0.01383943914247179,
        0, 0, 0.01553349164616098, 0.0153003443298752, 0.003182323338779773, 0, 0.01151251234732036, 0.01261814884776815,
        0.005015747404625367, 0, 0.009779941976381134, 0, 0, 0, 0.005898978646648359, 0.01759233439754007, 0, 0, 0, 0, 0, 0, 0.008636525925830774, 0.0142778632137181,
        0, 0, 0, 0.007698530512540223, 0, 0, 0, 0, 0, 0, 0.0003370130033674283, 0.007306410846780814, -0.001230573678577815, 0, -0.001413795443153852, -0.001717239310695781, 0, 0, 0.00492305526729598, 0.01109591847182253, 0, 0, 0, 0, -0.003310791787873644, 0.009995152277908721, 0.006985119145686737, 0.0173288413860494, 0, 0, 0, 0,
        0, 0.01602007889091014, 0.003198299956474914, 0.01386156434716355,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004558527804716961, 0, 0, 0.004774759324256685, 0, 0, 0, 0.0116976266335935, 0.007210009325158229, 0, 0, 0.00462386436765158, 0, 0, 0, 0.01372370821708051, 0.004718406076547615, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004236834782710379, 0, -0.003194375334008935, -0.002746001043995107, 0.01604593143402171, 0.007224593515354898, 0.009562089274057699, -0.00338218351898242,
        -0.002031460904419858, 0, 0.005050832866548528, 0.01605648152861459,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001225977779805077, -0.001334680520938713, 0, 0, 0, 0, 0, 0, 0.009173905251755154, 0.005331566780729114, 0.0139504951007888, -0.002683556589604022, -0.002335522846733781, 0, -0.004870528302619962, 0,
        -0.002006431157679476, 0.002412094801210576, 0, 0.01181683668751778,
        0.01523986386446615, 0.003934695436762752,
        0, 0.00656476282986982, -0.0003399521213203889, 0.009581360246738295, 0.004453415671425282, 0.01085989132132214, 0.003779907719972477, 0.01408466626564897,
        0.01020874161144901,
        0.01377525653968163, 0.002162718773543298, 0.01320479559206011, 0.004562703357425574,
        0, 0.005542670004969496, 0, 0.005882889861783709, 0, 0, 0.008008989511294964, 0.01333745034673167,
        -0.002293981418202208, 0.009678623704756265,
        -0.001487184469830165, -0.001345882582418679, 0, 0, 0.008073999639507634, 0.002951245844992038, 0.0124615169971586, 0.001249634009923163,
        0.002810535349346243, 0.0124975916093244,
        0.008600535082124553,
        0.01219885463833705, 0.002490378672171031,
        0.008417866370986569, -0.002991238260014547,
        0.007873505675664748,
        -0.004738938937063349, 0.003893793303985091, 0.003216700877146734, 0.01056651309048785
    ]
    model_ctrs = catboost_model_ctrs_container(
        used_model_ctrs_count = 19,
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
                    transposed_cat_feature_indexes = [3, 8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 4230580741181273963, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 4230580741181273963, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 10],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 4230580741181273965, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = -0, scale = 15)
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
                    transposed_cat_feature_indexes = [5, 7],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 4230580741057408764, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [7],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 14216163332699387103, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = -0, scale = 15),
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
                    catboost_model_ctr(base_hash = 14216163332699387072, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = -0, scale = 15),
                    catboost_model_ctr(base_hash = 16890222057671696975, base_ctr_type = "Counter", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [8],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 5, check_value_equal = 0, value = 3)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17362096607299135043, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [10],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 2, check_value_equal = 0, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 13402660528553184684, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [10],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 3, check_value_equal = 0, value = 4)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 12923321341810884907, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15)
                ]
            )
        ],
        ctr_data = catboost_ctr_data(
            learn_ctrs = {
                4230580741057408764 :
                catboost_ctr_value_table(
                    index_hash_viewer = {12653400145582130496 : 4, 13035191669214674561 : 8, 7188816836410779808 : 14, 9634357275635952003 : 3, 3254436053539040417 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16061267211184267017 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18140574223860629292 : 13, 707921246595766797 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 9130812135695277968 : 10, 4844161989476173969 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13514752404240993397 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 3490899219630952952 : 5, 18446744073709551615 : 0, 1155235699154202746 : 7, 15274270191700277019 : 6, 18446744073709551615 : 0, 12264198141704391741 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 21), catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.66247e-44, count = 17), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 4, 0, 21, 0, 12, 0, 2, 19, 17, 0, 5, 0, 2, 2, 3, 0, 2, 0, 2, 0, 6, 0, 1, 1, 0, 0, 1, 0, 1]
                ),
                4230580741181273963 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 1799168355831033313 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11936664559898134054 : 6, 14666845749088704071 : 10, 18429784838380727208 : 7, 17027374437435318793 : 13, 2862173265672040777 : 0, 16080065667299791243 : 5, 14677655266354382828 : 12, 12391839889973628461 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4082592020331177586 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 315676340386518075 : 3, 18446744073709551615 : 0, 10716245805238997245 : 2, 9313835404293588830 : 1, 17603450378469852574 : 11},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-44, count = 46), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 2.8026e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 8, 0, 4, 10, 46, 0, 1, 1, 3, 2, 6, 0, 2, 1, 4, 0, 2, 0, 2, 2, 0, 5, 0, 1, 0, 0, 1]
                ),
                4230580741181273965 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5599158075310182468 : 6, 8329339264500752485 : 7, 12092278353792775622 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 9742559182711839657 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 9671173347654628428 : 1, 12202197044987374381 : 8, 10298491039854442190 : 3, 1290122454378893647 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17719764915181305303 : 12, 13031886481356456536 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 4378739320651045659 : 0, 15662310901915236188 : 10, 11265943893881900988 : 9, 18446744073709551615 : 0, 13908405944952633343 : 4},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.26117e-44, count = 54), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 2.8026e-45, count = 5), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [9, 54, 0, 2, 1, 6, 0, 1, 1, 5, 2, 5, 0, 3, 2, 0, 1, 0, 5, 0, 0, 1, 0, 1, 0, 1, 1, 0]
                ),
                12923321341810884907 :
                catboost_ctr_value_table(
                    index_hash_viewer = {15788301966321637888 : 0, 7852726667766477841 : 5, 11772109559350781439 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 11840406731846624597 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2072654927021551577 : 3, 1211112939339888410 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1608091623008396926 : 7, 8746426839392254111 : 2},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.38221e-44, count = 73), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [17, 73, 0, 2, 0, 2, 3, 0, 1, 0, 0, 1, 0, 1, 1, 0]
                ),
                13402660528553184684 :
                catboost_ctr_value_table(
                    index_hash_viewer = {15788301966321637888 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12583823702175943146 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16571503766256089902 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5942209973749353715 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6339188657417862231 : 8, 13477523873801719416 : 5, 16503206593760246744 : 7, 2072654927021551577 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8746426839392254111 : 3},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 9), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-44, count = 64), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [0, 9, 0, 2, 20, 64, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
                ),
                14216163332699387072 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 8473802870189803490 : 2, 7071392469244395075 : 1, 18446744073709551615 : 0, 8806438445905145973 : 3, 619730330622847022 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 2.94273e-44, count = 61), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 12, 1, 5, 21, 61, 0, 1]
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
                16890222057671696978 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 13987540656699198946 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18089724839685297862 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10353740403438739754 : 2, 3922001124998993866 : 0, 13686716744772876732 : 1, 18293943161539901837 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 42,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 5.1848e-44, count = 4), catboost_ctr_mean_history(sum = 5.88545e-44, count = 13), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3)],
                    ctr_total = [37, 4, 42, 13, 2, 3]
                ),
                17362096607299135043 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14282620878612260867 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 12420782654419932198 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9743509310306231593 : 3, 9551523844202795562 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10742856347075653999 : 1},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 1.54143e-44, count = 54), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-44, count = 7)],
                    ctr_total = [0, 12, 1, 5, 11, 54, 0, 1, 10, 7]
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



