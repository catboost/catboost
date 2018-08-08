try:
    from cityhash import CityHash64  # Available at https://github.com/Amper/cityhash #4f02fe0ba78d4a6d1735950a9c25809b11786a56
except ImportError:
    from cityhash import hash64 as CityHash64  # ${catboost_repo_root}/library/python/cityhash


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
    float_feature_count = 6
    cat_feature_count = 11
    binary_feature_count = 74
    tree_count = 20
    float_feature_borders = [
        [17.5, 36.5, 61.5, 68.5],
        [38811, 51773, 59723, 204331, 449128.5, 553548.5],
        [10.5, 12.5, 13.5, 14.5],
        [3280],
        [1738, 1881.5, 2189.5],
        [46.5]
    ]
    tree_depth = [6, 6, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    tree_split_border = [4, 1, 2, 1, 1, 1, 2, 3, 1, 3, 1, 1, 4, 1, 1, 1, 1, 1, 2, 5, 1, 1, 4, 1, 4, 1, 2, 3, 1, 1, 2, 2, 1, 1, 1, 1, 3, 1, 1, 1, 255, 2, 2, 2, 1, 1, 2, 1, 2, 4, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 1, 2, 2, 2, 255, 1, 1, 6, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 4, 1, 2, 1, 1, 1, 1, 1]
    tree_split_feature_index = [0, 63, 2, 32, 33, 68, 36, 2, 46, 0, 49, 51, 1, 68, 64, 27, 60, 70, 25, 1, 28, 36, 2, 57, 0, 39, 1, 24, 1, 37, 38, 23, 53, 67, 2, 55, 1, 59, 42, 31, 6, 4, 8, 1, 41, 10, 4, 13, 9, 0, 4, 40, 12, 24, 29, 38, 7, 72, 30, 61, 23, 20, 21, 37, 58, 9, 22, 26, 14, 17, 0, 38, 58, 71, 62, 26, 56, 58, 45, 72, 9, 48, 52, 4, 43, 15, 47, 35, 0, 4, 44, 8, 24, 26, 1, 6, 3, 4, 1, 25, 34, 5, 11, 18, 19, 38, 16, 67, 0, 69, 54, 2, 50, 8, 5, 73, 65, 66, 25]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    one_hot_cat_feature_index = [9]
    one_hot_hash_values = [
        [-2114564283]
    ]
    ctr_feature_borders = [
        [0.2640624940395355],
        [0.25, 0.5],
        [0.02941176667809486, 0.05392156913876534],
        [0.2750875055789948],
        [0.6001399755477905],
        [0.2179407030344009],
        [0.5110822319984436],
        [0.5110822319984436],
        [0.5001167058944702],
        [0.9591957330703735],
        [0.2165127396583557],
        [0.9591957330703735],
        [0.5110822319984436],
        [0.3768093883991241],
        [0.8005866408348083],
        [0.5202279686927795],
        [0.4583333432674408, 0.859375],
        [0.2942708432674408, 0.4270833134651184, 0.4713541567325592],
        [0.625, 0.7000000476837158],
        [0.01470588333904743, 0.2352941334247589],
        [0.9554044604301453],
        [0.917491614818573],
        [0.1897294521331787],
        [-0.009802941232919693],
        [0.8007199168205261],
        [0.212610587477684],
        [0.2531880140304565],
        [0.9099090099334717],
        [0.6689189076423645],
        [0.4801520109176636, 0.6642736196517944],
        [0.4642857313156128, 0.6071428656578064],
        [0.02450980432331562, 0.3872548937797546],
        [0.9873741865158081],
        [0.9738485813140869],
        [0.4019205868244171],
        [0.7717800140380859],
        [0.689197838306427],
        [0.3921176493167877],
        [0.1780280321836472],
        [1.000899910926819],
        [0.3210945427417755],
        [-0.009802941232919693],
        [0.8007199168205261],
        [0.06862059235572815],
        [0.8007199168205261],
        [0.8256924748420715],
        [1.000899910926819],
        [0.3343569934368134],
        [0.9714616537094116],
        [0.9861807227134705],
        [0.375],
        [0.01470588333904743, 0.1421568691730499],
        [1.000899910926819],
        [0.9383437037467957],
        [0.7045454978942871],
        [0.7321428060531616],
        [0.6542679071426392],
        [0.8101000189781189],
        [0.5002500414848328],
        [0.5286458134651184],
        [0.6875],
        [0.4607843160629272],
        [0.9402392506599426],
        [0.9554044604301453],
        [0.05514705926179886],
        [0.6875],
        [0.4656862616539001]
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007499999832361937, 0, 0, 0, 0, 0, 0, 0, 0.006000000052154064, 0, 0.02400000020861626, 0.007499999832361937, 0.007499999832361937, 0, 0.01200000010430813, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007499999832361937, 0, 0, 0, 0, 0, 0, 0, 0.0169565211981535, 0, 0.02571428567171097, 0, 0.003000000026077032, 0, 0.02285714261233807, 0,
        0.01847253181040287, 0.02758516371250153, 0.002167582279071212, 0.01473392825573683, 0, 0, 0, 0, -0.0001017391259665601, 0.01181785762310028, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0.01171976048499346, 0.005697329062968493, 0.008425761945545673, 0.01433677691966295, 0.02086312137544155, 0.0199800506234169, 0.02266760356724262, 0.02407447062432766, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002714372472837567, 0.01595684327185154, 0, 0, 0, 0, 0, 0, 0.01601459085941315, 0.02370902337133884, 0, 0.0158926397562027, 0, 0, 0, 0, -0.0001019500778056681, 0, 0, 0, 0, 0, 0, 0, 0, 0.005765947513282299, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01656651869416237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.006752429530024529, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005176381673663855, 0, 0, 0, -0.0004485874378588051, 0, 0, 0, 0.02199704572558403, 0.005424461793154478, 0, 0,
        0.006587451323866844, 0, 0, 0, 0, 0, 0, 0, 0.02450917102396488, 0.0159892700612545, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02330418303608894, 0.003080242080613971, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0.01144003495573997, 0, 0, 0.00825794879347086, 0.02251898869872093, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01130221504718065, 0, 0, 0, 0.02336438000202179, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004327624104917049, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0009059818112291396, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007223065011203289, 0, 0, 0, 0, 0, 0, -0.0008991869399324059, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.001266310107894242, 0, 0, 0.004663164261728525, 0, 0.02390770427882671, 0, 0, 0, 0.01376825850456953, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004262709058821201,
        0, 0.005097512621432543, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004652826115489006, 0.02035207860171795, 0.006441782228648663, -0.0004930952563881874, -0.0008924430003389716, -0.0002275376318721101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0002729316474869847, -0.0005511741037480533, 0, -0.001586188096553087, 0, 0, 0.009997958317399025, 0.0226378683000803, 0, 0.002931951079517603, -0.000225831099669449, 0.01264820899814367,
        -0.0002708846295718104, 0, 0.004903013817965984, 0, -0.0004062593798153102, 0, 0.01041687931865454, 0, 0.004601195454597473, 0, -0.002038163132965565, -0.0008738532196730375, 0, 0, 0.006326347123831511, 0.006288318429142237, 0, 0, 0, 0, 0, 0, 0, 0, 0.01234117057174444, 0.005861080251634121, 0.005976180545985699, 0.01634914427995682, 0, 0.01638057455420494, 0, 0.02053526230156422, 0, 0, -0.0006894372054375708, 0, -0.001100746681913733, 0, 0.01786068081855774, 0, -0.0006467309431172907, 0, -0.0009685574914328754, 0, -0.000224137373152189, 0, 0, 0.006267681252211332, 0, 0, 0, 0, 0, 0, -0.0005842586397193372, 0, 0.003975818399339914, 0.005853517912328243, 0, 0.01158045418560505, 0, 0, 0, 0.01816232316195965,
        0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0008687424124218524, 0, 0, 0, -0.0008672993280924857, 0, 0.006264025811105967, 0, 0, 0, 0.007347192149609327, 0, 0, 0, 0, -0.000581549305934459, -0.0006842664442956448, 0, 0.01822366379201412, 0, 0, -0.001395853934809566, 0.01868292689323425, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0005798767087981105, 0, 0, 0, 0, 0, 0, 0, 0.0072466223500669, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0.006581226829439402, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00109198538120836, 0, 0.003929005935788155, 0, 0, 0.006169311702251434, 0.009158238768577576, 0, 0, 0.01284539978951216, 0, 0.005574367009103298, 0, 0.02088439837098122, 0.005921437870711088, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.001200675731524825, 0, -0.0008630763622932136, 0, 0, 0, 0, 0, 0.004742668475955725, 0, 0.009196614846587181, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007538412697613239, 0, 0, 0, 0, 0, 0, 0.003559011034667492, 0.01925214752554893,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00823186244815588, 0, -0.002024621702730656, 0, 0, 0, 0.007248329930007458, 0, 0, 0, 0.004668214358389378, 0, 0, 0, 0.01198064628988504, -0.001166831003502011, 0.01629480719566345, 0.004230910912156105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01953712664544582, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005415867082774639, 0, -0.001351873273961246, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0003595627786125988, 0, 0.005832122638821602, -0.0004880440246779472, -0.001394068705849349, 0, 0, 0, -0.0008550564525648952, 0, 0.01903237029910088, 0, 0.009482814930379391, 0, 0, 0, 0, 0, 0.005301141645759344, 0, 0, 0, 0, 0, 0, 0, 0.01694683730602264, -0.0009529921808280051, 0.01124119199812412, 0,
        0.008881010115146637, 0.004034095909446478, -0.0005869531887583435, 0, -0.001084911287762225, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01546689867973328, 0.01814763993024826, -0.003182707587257028, 0, 0.00406634621322155, 0.003149079624563456, 0, 0, 0.004973651375621557, 0.004905948415398598, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004733889829367399, 0.004846477881073952, 0, 0, 0, 0, 0, -0.0008636185084469616, 0.005787669215351343, 0, 0, 0, 0, 0, -0.001241611083969474, 0.005106562748551369, 0.01692573353648186, 0.002728525316342711, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0.005751724354922771, 0, -0.0009804083965718746, 0, 0, 0, 0, 0, 0.01701867394149303, 0.004851416684687138, 0.01527677290141582, 0, 0, 0, 0, 0, 0, 0, -0.001531464862637222, 0, 0, 0, 0, 0, 0, 0, 0.006127183325588703, 0, 0, 0, 0, 0, 0, 0, 0.006999356672167778, 0, 0, 0, 0, 0, 0, 0, -0.002223898656666279, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005480255465954542, -0.0004544197872746736, -0.001068010460585356, 0, 0, 0, 0, -0.0006257341592572629, 0.004451120737940073, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005006264429539442, 0, 0, 0, 0.009330349043011665, 0, 0, 0, -0.001663518138229847, 0, 0, 0, 0.009476527571678162, -0.001506267930381, -0.0002902768610510975, 0, 0.009738394059240818, 0, 0, -0.0009527977090328932, 0.01667401939630508, 0, 0.01057973131537437
    ]
    model_ctrs = catboost_model_ctrs_container(
        used_model_ctrs_count = 67,
        compressed_model_ctrs = [
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791580653471478, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 768791580653471478, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 8405694746487331134, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 1, check_value_equal = 0, value = 2)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 3001583246656978020, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 5, check_value_equal = 0, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 4544184825393173621, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 4, 5],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 4),
                        catboost_bin_feature_index_value(bin_index = 4, check_value_equal = 0, value = 3)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 4414881145659723684, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 4, 5],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 1, check_value_equal = 0, value = 2)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 3001583246125931243, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 4, 7, 8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952491747546147, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 5],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 4, check_value_equal = 0, value = 2)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 2790902285321313792, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 5],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 5, check_value_equal = 0, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 13902559248212744134, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 5, 8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791582259504189, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 6],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 5, check_value_equal = 0, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 13902559248212744135, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 6, 7],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 5, check_value_equal = 0, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 4544184825161771334, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 7],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493224740167, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 17677952493224740167, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493224740166, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791580653471473, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 768791580653471473, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 768791580653471473, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 8405694746487331129, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 1, check_value_equal = 0, value = 4)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 9867321491374199501, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 5],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493261641996, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 17677952493261641996, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 10041049327410906820, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 6],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 1, check_value_equal = 0, value = 3)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 5840538188647484189, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 7],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 4)
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
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 4)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 5819498284603408945, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 10],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493261641995, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791580653471472, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 768791580653471472, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 768791580653471472, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 8405694746487331128, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 4)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 12627245789391619615, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 4),
                        catboost_bin_feature_index_value(bin_index = 4, check_value_equal = 0, value = 3)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 4414881145133934893, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
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
                    catboost_model_ctr(base_hash = 6317293569456956330, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 1, check_value_equal = 0, value = 3)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 12606205885276083426, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 4, check_value_equal = 0, value = 2)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 15655841788288703925, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 4, check_value_equal = 0, value = 3)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 8628341152511840406, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 6],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493260528854, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 17677952493260528854, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 6],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 4, check_value_equal = 0, value = 2)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 2790902285205833619, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 6, 7],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 8405694746995314031, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
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
                    transposed_cat_feature_indexes = [5, 6, 8],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 2)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 3863811882172310855, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 6, 8, 10],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493297533872, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 17677952493297533872, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493260528848, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 8],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 2)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 13000966989535245561, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 10],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493260528850, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 17677952493260528850, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [6],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791580653471475, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 8405694746487331131, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [6],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 1, check_value_equal = 0, value = 3)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 12606205885276083425, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [6, 8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493263343771, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [7],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791580653471474, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 768791580653471474, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [7],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 4)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 12627245789391619613, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [7],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 1, check_value_equal = 0, value = 4)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 9867321491374199502, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [7],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 5, check_value_equal = 0, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 4544184825393173617, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791580653471469, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 768791580653471469, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 8405694746487331109, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [8],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 2)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 5445777084271881924, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [8, 10],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493265578087, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [10],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791580653471471, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 768791580653471471, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 8405694746487331111, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            )
        ],
        ctr_data = catboost_ctr_data(
            learn_ctrs = {
                768791580653471469 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 8473802870189803490 : 0, 7071392469244395075 : 3, 18446744073709551615 : 0, 8806438445905145973 : 2, 619730330622847022 : 1, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.94273e-44, count = 61), catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5)],
                    ctr_total = [21, 61, 0, 12, 0, 1, 1, 5]
                ),
                768791580653471471 :
                catboost_ctr_value_table(
                    index_hash_viewer = {2136296385601851904 : 0, 7428730412605434673 : 5, 9959754109938180626 : 2, 14256903225472974739 : 3, 8056048104805248435 : 1, 18446744073709551615 : 0, 12130603730978457510 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10789443546307262781 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.8026e-44, count = 73), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [20, 73, 0, 2, 1, 0, 0, 1, 1, 0, 0, 2, 0, 1]
                ),
                768791580653471472 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 13987540656699198946 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18089724839685297862 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10353740403438739754 : 1, 3922001124998993866 : 0, 13686716744772876732 : 4, 18293943161539901837 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 37), catboost_ctr_mean_history(sum = 3.08286e-44, count = 20), catboost_ctr_mean_history(sum = 0, count = 13), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 3)],
                    ctr_total = [0, 37, 22, 20, 0, 13, 0, 2, 0, 4, 0, 3]
                ),
                768791580653471473 :
                catboost_ctr_value_table(
                    index_hash_viewer = {7537614347373541888 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5903587924673389870 : 4, 18278593470046426063 : 9, 10490918088663114479 : 8, 18446744073709551615 : 0, 407784798908322194 : 5, 5726141494028968211 : 6, 1663272627194921140 : 7, 8118089682304925684 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15431483020081801594 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 1403990565605003389 : 0, 3699047549849816830 : 1, 14914630290137473119 : 2},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 4.2039e-45, count = 16), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 9.80909e-45, count = 3), catboost_ctr_mean_history(sum = 5.60519e-45, count = 16), catboost_ctr_mean_history(sum = 5.60519e-45, count = 24), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [3, 16, 2, 0, 7, 3, 4, 16, 4, 24, 0, 5, 1, 3, 0, 3, 0, 3, 0, 4, 0, 1, 1, 1]
                ),
                768791580653471474 :
                catboost_ctr_value_table(
                    index_hash_viewer = {3607388709394294015 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18356215166324018775 : 4, 18365206492781874408 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 14559146096844143499 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 11416626865500250542 : 1, 5549384008678792175 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 22), catboost_ctr_mean_history(sum = 2.66247e-44, count = 17), catboost_ctr_mean_history(sum = 0, count = 22), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 14), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3)],
                    ctr_total = [0, 22, 19, 17, 0, 22, 1, 1, 0, 14, 2, 3]
                ),
                768791580653471475 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 14452488454682494753 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1388452262538353895 : 8, 8940247467966214344 : 2, 4415016594903340137 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 41084306841859596 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8678739366408346384 : 1, 18446744073709551615 : 0, 4544226147037566482 : 11, 14256903225472974739 : 5, 16748601451484174196 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5913522704362245435 : 3, 1466902651052050075 : 7, 2942073219785550491 : 12, 15383677753867481021 : 6, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 8.40779e-45, count = 10), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 2.8026e-45, count = 8), catboost_ctr_mean_history(sum = 0, count = 11), catboost_ctr_mean_history(sum = 2.8026e-45, count = 9), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 2.8026e-45, count = 14), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 9.80909e-45, count = 6), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [6, 10, 0, 6, 2, 8, 0, 11, 2, 9, 1, 5, 2, 14, 0, 2, 7, 6, 1, 4, 0, 3, 1, 0, 0, 1]
                ),
                768791580653471478 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 15379737126276794113 : 5, 18446744073709551615 : 0, 14256903225472974739 : 3, 18048946643763804916 : 6, 2051959227349154549 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7024059537692152076 : 4, 18446744073709551615 : 0, 15472181234288693070 : 1, 8864790892067322495 : 2},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 4.2039e-45, count = 6), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6), catboost_ctr_mean_history(sum = 1.4013e-44, count = 58), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 4)],
                    ctr_total = [3, 6, 1, 6, 10, 58, 1, 5, 5, 0, 2, 0, 0, 4]
                ),
                768791582220620454 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 8729380230485332353 : 34, 9977784445157143938 : 47, 10895282230322158083 : 22, 11761827888484752132 : 4, 12615796933138713349 : 42, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16179912168674497673 : 19, 11168821652508931466 : 17, 18446744073709551615 : 0, 10475827037446056716 : 45, 14750448552479345421 : 11, 16495354791085375886 : 13, 10135854469738880143 : 48, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12233930600554179099 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3208547115700824735 : 14, 18277252057819687584 : 40, 11380194846102552353 : 6, 18446744073709551615 : 0, 14030234294679959587 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14597065438263107115 : 46, 1433532977505263916 : 37, 17401263168096565421 : 24, 15971769898056770222 : 43, 2808237437298927023 : 20, 1256940733300575664 : 21, 18446744073709551615 : 0, 9689317982178109362 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6069959757871852856 : 1, 7318363972543664441 : 41, 18446744073709551615 : 0, 5876843741908031419 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5859361731318044735 : 36, 6636790901455000384 : 30, 18446744073709551615 : 0, 18446744073709551615 : 0, 7078520115819519811 : 28, 10322519660234049603 : 49, 8011495361248663491 : 18, 9259899575920475076 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9323661606633862475 : 32, 18146214905751188428 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3907755348917795664 : 33, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8720774373489072856 : 0, 6896376012912388953 : 27, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10662309976865883491 : 5, 9111013272867532132 : 15, 10359417487539343717 : 29, 17543390521745065830 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 13529097553057277673 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 2350285447658856812 : 23, 16689654767293439084 : 44, 18446744073709551615 : 0, 6176161755113003119 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 346329515989321719 : 39, 4263979015577900536 : 10, 6353480505666359544 : 12, 9547190628529608953 : 31, 9583115984936959099 : 38, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 8.40779e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 9.80909e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 4, 6, 1, 0, 3, 0, 2, 0, 3, 0, 5, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 2, 5, 0, 1, 0, 1, 7, 1, 2, 1, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 3, 0, 3, 1, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 4, 0, 1, 0, 3, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                ),
                768791582259504189 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 4639344346382560065 : 0, 6768655180658783362 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17601732372345076103 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9253120901934657613 : 11, 18446744073709551615 : 0, 12494120118534626831 : 23, 18446744073709551615 : 0, 18446744073709551615 : 0, 15104639456961940114 : 1, 10170507820794899987 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 17626484575418309142 : 27, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10540782073424112667 : 3, 5606650437257072540 : 18, 18446744073709551615 : 0, 14838774965469232670 : 17, 18446744073709551615 : 0, 16546754159773737760 : 20, 8171065581604191777 : 28, 8376012298141440672 : 10, 17449294303896545953 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2686709533857156199 : 15, 8500597432574416232 : 21, 4462546031259207335 : 25, 12885436920358718506 : 2, 6984702425902276202 : 12, 17008555610512316647 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 8962398883312995119 : 7, 10515720428538797616 : 19, 18446744073709551615 : 0, 11572918221740308402 : 29, 3982985296232888499 : 6, 646524893007459764 : 22, 582150902654165941 : 9, 5031364380791762038 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7009060838202480955 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5478643861907335871 : 8},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 1.4013e-44, count = 15), catboost_ctr_mean_history(sum = 0, count = 19), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 4, 1, 2, 0, 7, 10, 15, 0, 19, 0, 1, 0, 2, 0, 2, 0, 1, 0, 2, 5, 0, 2, 0, 0, 1, 0, 1, 0, 4, 0, 3, 0, 1, 0, 1, 0, 3, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1]
                ),
                2790902285205833619 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 2188507106676080001 : 5, 637210402677728642 : 19, 15993786133470765187 : 11, 9069587651555262340 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 5055294682867474183 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 12323226651178604938 : 22, 8215851897103635594 : 37, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14236920219097648662 : 9, 17912585353630038166 : 27, 18446744073709551615 : 0, 1109313114747155609 : 32, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8703931276011015453 : 28, 18446744073709551615 : 0, 255577360295528863 : 29, 18446744073709551615 : 0, 18446744073709551615 : 0, 3288025018294948642 : 4, 4141994062948909859 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7706109298484694183 : 18, 2695018782319127976 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 6276645682289541931 : 10, 8021551920895572396 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 2327253922091514671 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11721681120454478260 : 33, 18446744073709551615 : 0, 18446744073709551615 : 0, 2894085065674662199 : 26, 18446744073709551615 : 0, 3760127730364375609 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13181488319220572861 : 13, 9803449187629884094 : 34, 2906391975912748863 : 6, 18446744073709551615 : 0, 5556431424490156097 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6123262568073303625 : 39, 3404434568201661641 : 38, 8927460297906761931 : 23, 7497967027866966732 : 36, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10275551899699311061 : 20, 16042900961391600982 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 15849784945427779545 : 7, 18446744073709551615 : 0, 5368307437087193947 : 14, 18446744073709551615 : 0, 15832302934837792861 : 30, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17051461319339267937 : 25, 9516124139116033121 : 40, 1848716790044246113 : 41, 17984436564768411617 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9672412035561384938 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 246971503299269366 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 813802646882416894 : 31, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 8.40779e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 7.00649e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 5, 1, 0, 0, 4, 0, 2, 0, 5, 0, 7, 0, 3, 0, 1, 0, 2, 0, 2, 2, 7, 0, 1, 0, 1, 6, 1, 1, 0, 0, 2, 0, 2, 0, 1, 0, 2, 1, 2, 0, 1, 5, 1, 0, 3, 0, 4, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 2, 4, 0, 3, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                ),
                2790902285321313792 :
                catboost_ctr_value_table(
                    index_hash_viewer = {8975491433706742463 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14435487234778955461 : 22, 26794562384612742 : 19, 18446744073709551615 : 0, 4411634050168915016 : 2, 11361933621181601929 : 1, 15118949489711741514 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 488596013123191629 : 6, 2041917558348994126 : 16, 18446744073709551615 : 0, 3099115351550504912 : 23, 13955926499752636625 : 5, 6798076237643774482 : 17, 10555092106173914067 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4633306462361102487 : 11, 18446744073709551615 : 0, 16982002041722229081 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14612285549902308191 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 779318031744854123 : 10, 18446744073709551615 : 0, 4020317248344823341 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 6630836586772136624 : 13, 18446744073709551615 : 0, 15927023829150890738 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2066979203234309177 : 3, 16388825279889469625 : 15, 18446744073709551615 : 0, 6364972095279429180 : 12, 18446744073709551615 : 0, 18348953501661188798 : 9, 18144006785123939903 : 21},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 1.26117e-44, count = 18), catboost_ctr_mean_history(sum = 0, count = 26), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 5.60519e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2)],
                    ctr_total = [0, 4, 1, 1, 0, 8, 9, 18, 0, 26, 0, 2, 0, 2, 0, 1, 0, 3, 4, 0, 2, 0, 1, 0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 2, 1, 0, 0, 1, 0, 3, 1, 0, 3, 0, 0, 1, 0, 2]
                ),
                3001583246125931243 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 3628762670213083650 : 21, 4152033245790959619 : 48, 18446744073709551615 : 0, 17407091705877351685 : 20, 6882106995078381574 : 36, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12587226841112876431 : 1, 18317959509575665424 : 25, 742970608404689295 : 15, 16344700011017827602 : 32, 2035992302241612690 : 40, 18446744073709551615 : 0, 12338705865783753109 : 23, 18446744073709551615 : 0, 18446744073709551615 : 0, 2314852681173712664 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 16662434637249573787 : 51, 12814906903325799324 : 38, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2339754841350749476 : 39, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11760162732384441256 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 10190834038247912619 : 29, 18446744073709551615 : 0, 808135643011091501 : 37, 18446744073709551615 : 0, 18446744073709551615 : 0, 15248885571486568624 : 8, 18446744073709551615 : 0, 16951015140729532594 : 10, 4483191194562101811 : 50, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12966982428170399929 : 47, 18446744073709551615 : 0, 18446744073709551615 : 0, 7619784579124933820 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14111457067786581057 : 18, 6827846041496707650 : 30, 14168626759096197955 : 24, 1232811508289654594 : 34, 16285601543817880901 : 19, 15931517322100805958 : 46, 8622225544353820353 : 0, 6088165380785275845 : 14, 12273342661996998085 : 49, 12095984437040160455 : 41, 18446744073709551615 : 0, 12637007560756539980 : 11, 8792735915257126348 : 35, 14339137129999503950 : 3, 18395773067821135182 : 33, 18446744073709551615 : 0, 18446744073709551615 : 0, 16513281606030803794 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 6497871174082847701 : 31, 18446744073709551615 : 0, 18446744073709551615 : 0, 6558771518591634648 : 42, 18446744073709551615 : 0, 18446744073709551615 : 0, 14981662407691145819 : 9, 18446744073709551615 : 0, 4209995588326251229 : 27, 18446744073709551615 : 0, 8877606263321985375 : 52, 18446744073709551615 : 0, 10579735832564949345 : 43, 18446744073709551615 : 0, 10636905523874566243 : 4, 8359469528752380003 : 16, 8842526021017453540 : 2, 7781340352808279782 : 44, 18446744073709551615 : 0, 18446744073709551615 : 0, 14110664416560906345 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 9105286325534908268 : 5, 18446744073709551615 : 0, 10807415894777872238 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9608040095590038645 : 45, 3924353183578036726 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15648596808374320252 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 3, 1, 0, 0, 2, 2, 6, 0, 7, 0, 8, 0, 5, 0, 1, 0, 1, 0, 4, 0, 1, 1, 7, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3, 0, 2, 0, 1, 0, 1, 2, 1, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 3, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1]
                ),
                3001583246656978020 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 12337359831519453058 : 0, 18446744073709551615 : 0, 6973539969458659060 : 2, 13860744542689514389 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 16503206593760246744 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 2242442935049193755 : 7, 18446744073709551615 : 0, 8193958724117795869 : 6, 10924139913308365886 : 5, 14687079002600389023 : 1},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 4.2039e-45, count = 6), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6), catboost_ctr_mean_history(sum = 1.26117e-44, count = 57), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [3, 6, 1, 6, 9, 57, 1, 5, 5, 0, 2, 0, 0, 4, 1, 1]
                ),
                3863811882172310855 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16539280100125922053 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 2933836577635514888 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 6624153332048651 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15584109416824463631 : 5, 1079537663959258768 : 37, 10795630651492496657 : 40, 18446744073709551615 : 0, 3653789196400846099 : 17, 16657022927451673748 : 7, 14309218433848032148 : 15, 5255101148688962965 : 55, 784530386183709463 : 29, 12724112379326185240 : 41, 3078130021364510233 : 33, 5792833011267379482 : 49, 18446744073709551615 : 0, 8789495792810759068 : 25, 9809552026335107740 : 53, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6935161387918818980 : 8, 18446744073709551615 : 0, 8854381343033649318 : 51, 2783161425565058471 : 10, 4065902577701682344 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7873442382053928881 : 13, 17509849011186116785 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6064533617083031352 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9578121399438566843 : 35, 18446744073709551615 : 0, 3930994787075050813 : 34, 9483211823068989630 : 38, 18446744073709551615 : 0, 18446744073709551615 : 0, 1061735976857914177 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2228349427768373958 : 48, 10525597742391453127 : 9, 8528041139767531208 : 19, 18446744073709551615 : 0, 17730612602998780490 : 57, 919911597814063947 : 54, 18446744073709551615 : 0, 9600954650394741325 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16775441919697322068 : 39, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12778751976776651480 : 11, 17848355430324532185 : 32, 18446744073709551615 : 0, 2918542637195412955 : 42, 13003457108638579292 : 43, 18446744073709551615 : 0, 18446744073709551615 : 0, 3568846733801557215 : 14, 14173837222217677664 : 44, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4752114788659524325 : 52, 14015514739234771686 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14803390641069418859 : 58, 18446744073709551615 : 0, 2941375888151587437 : 4, 15943996467068460269 : 28, 6587534006707254895 : 31, 7739116426202412656 : 23, 15734784568894920048 : 36, 14635558637114150258 : 18, 6602984835177365875 : 45, 4869857615985276020 : 27, 12902105974959694703 : 59, 17455209413735650545 : 1, 15321670583727670006 : 30, 3404470224630628343 : 56, 13938439269304993529 : 46, 12452411773510533370 : 12, 14449968376134455289 : 50, 15449074555053912956 : 3, 7255866119955889789 : 47, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 4.2039e-45, count = 5), catboost_ctr_mean_history(sum = 4.2039e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 9.80909e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 7), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 5.60519e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [3, 5, 3, 2, 1, 5, 2, 2, 1, 1, 2, 1, 7, 1, 1, 7, 2, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1]
                ),
                4414881145133934893 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15744222165914032517 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5649296009765293714 : 4, 12515044663515391091 : 1, 7217896454495632755 : 8, 18446744073709551615 : 0, 16950186414058081238 : 6, 16498401743253028919 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 10434762244723145786 : 0, 7783947629105925786 : 3, 6133087514440113244 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 37), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 13), catboost_ctr_mean_history(sum = 2.66247e-44, count = 20), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 37, 2, 0, 0, 13, 19, 20, 0, 2, 0, 4, 0, 2, 1, 0, 0, 1]
                ),
                4414881145659723684 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 1968438200869838210 : 51, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17679689593011713800 : 19, 17770863836821556104 : 0, 5873407800382985098 : 52, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15354859179876249743 : 29, 18446744073709551615 : 0, 18446744073709551615 : 0, 8241769915270219922 : 49, 18446744073709551615 : 0, 4781221601667622548 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 13259958884694735255 : 41, 18446744073709551615 : 0, 18446744073709551615 : 0, 4656770217989207578 : 33, 3123998877235734427 : 44, 2880537555826393628 : 14, 1045758839855420444 : 50, 6453812626951551388 : 20, 7373521406379420317 : 10, 15208993820173816478 : 47, 5036837526151520545 : 46, 18446744073709551615 : 0, 18446744073709551615 : 0, 785637260644879140 : 39, 18446744073709551615 : 0, 18446744073709551615 : 0, 14201924189220883111 : 1, 1883760134919382184 : 45, 7171489145492152617 : 3, 2203248159541751849 : 34, 5114067664025077675 : 7, 7763215270077623596 : 24, 18433555596902063917 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7248719283365709747 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14392356253141541821 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3334767067226870465 : 37, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9822303761109372617 : 6, 1918034629091685706 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2657477670942228686 : 15, 14131880866619031375 : 13, 9892630029936032464 : 23, 18446744073709551615 : 0, 6432081716333435090 : 38, 12606536426880398291 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1240745232807043927 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 9144531774667173594 : 30, 6557251021933574875 : 27, 1262915927860373596 : 21, 18446744073709551615 : 0, 7116775155420360158 : 53, 12404504165993130591 : 11, 10606002133962586720 : 48, 63527192270722015 : 36, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 133853461097381862 : 43, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11727214769708488812 : 32, 9928712737677944941 : 31, 18048189026166942061 : 35, 15146535830480538223 : 25, 17409370781001408239 : 40, 303226289080229489 : 12, 9082896331950655090 : 4, 12760211465443864178 : 17, 18446744073709551615 : 0, 7611634590235858933 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4722706790762200317 : 42, 18446744073709551615 : 0, 15055318781610350591 : 5},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 3, 1, 0, 0, 2, 3, 6, 0, 7, 0, 8, 0, 5, 0, 1, 0, 1, 0, 4, 0, 1, 1, 7, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3, 0, 2, 0, 1, 0, 1, 1, 0, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 3, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 2, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1]
                ),
                4544184825161771334 :
                catboost_ctr_value_table(
                    index_hash_viewer = {158999094665252608 : 33, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8367117909645166341 : 19, 18446744073709551615 : 0, 8702450991728868615 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6262126705356135693 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 14946871803847873040 : 34, 13913159826293879825 : 29, 3752585126949001232 : 64, 18446744073709551615 : 0, 18446744073709551615 : 0, 14375335943966472213 : 32, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6045632691652965145 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12440274563092365353 : 46, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6619783738664723247 : 23, 18446744073709551615 : 0, 4905776570447084337 : 37, 18446744073709551615 : 0, 8130996685331913523 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14732992977275059767 : 56, 18446744073709551615 : 0, 2585940615564665401 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 3128199045106796348 : 47, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7735182141653903170 : 7, 17619157723446594371 : 44, 11241408283717132868 : 48, 13574756925474066500 : 53, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18314399080722540106 : 27, 4146295242583377226 : 43, 18446744073709551615 : 0, 3172219588709525325 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17343663287863212120 : 20, 174679536037619032 : 24, 18446744073709551615 : 0, 79769959668041819 : 51, 16685972765223635547 : 54, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16843846084406882659 : 11, 518059473081761380 : 52, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6182897570358924904 : 45, 18446744073709551615 : 0, 18446744073709551615 : 0, 6510311249307667563 : 21, 12533704194145800556 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11009716383731513464 : 41, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13808447517377610367 : 42, 18446744073709551615 : 0, 7824087488121779841 : 1, 13795416998968128130 : 55, 7469564332260859522 : 59, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12443445925107193993 : 50, 6540554603667512458 : 30, 18446744073709551615 : 0, 18123001185196525196 : 13, 18446744073709551615 : 0, 8051767550334702734 : 40, 2891096590926338447 : 62, 18446744073709551615 : 0, 6116316705811167633 : 0, 9269497864691699089 : 63, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8490057063748422295 : 58, 18446744073709551615 : 0, 4919885718948249 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 522424025960426143 : 57, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17603342537294645418 : 8, 16803678464185371818 : 61, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7617439314328151475 : 6, 18446744073709551615 : 0, 3320670515925237429 : 26, 13992388961291090614 : 4, 18446744073709551615 : 0, 1385219671401702328 : 31, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6662268631026693053 : 5, 16764616949409671870 : 12, 6124861826650175934 : 14, 9498428910012038848 : 36, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7162655135050725840 : 35, 12072581429775906513 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 11977671853406329300 : 39, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 197512786993793514 : 49, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10586639264769695215 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 635802300407513075 : 25, 18446744073709551615 : 0, 6475377084227676405 : 60, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12735534006750400250 : 38, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 1, 1, 0, 0, 1, 0, 2, 0, 2, 0, 6, 0, 4, 0, 3, 0, 1, 0, 3, 0, 2, 0, 3, 0, 7, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 2, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 2, 1, 0, 1, 2, 0, 0, 2, 0, 1, 0, 1, 1, 1, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 2, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
                ),
                4544184825393173617 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5844492600899280932 : 0, 18446744073709551615 : 0, 1034166431492604838 : 2, 18446744073709551615 : 0, 6203552979315789704 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 1113395566489815627 : 3, 13957701839509617452 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9226604805100152147 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13302932820562179799 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15316838452862012827 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5765263465902070143 : 1},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-44, count = 1), catboost_ctr_mean_history(sum = 1.26117e-44, count = 16), catboost_ctr_mean_history(sum = 0, count = 18), catboost_ctr_mean_history(sum = 0, count = 22), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 2)],
                    ctr_total = [0, 4, 10, 1, 9, 16, 0, 18, 0, 22, 1, 1, 0, 12, 2, 3, 0, 2]
                ),
                4544184825393173621 :
                catboost_ctr_value_table(
                    index_hash_viewer = {11772109559350781439 : 4, 18446744073709551615 : 0, 12337359831519453058 : 0, 18446744073709551615 : 0, 3462861689708330564 : 10, 6193042878898900581 : 7, 9955981968190923718 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 7606262797109987753 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6973539969458659060 : 2, 13860744542689514389 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2242442935049193755 : 3, 9129647508280049084 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 14687079002600389023 : 1},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 7.00649e-45, count = 5), catboost_ctr_mean_history(sum = 7.00649e-45, count = 53), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 5), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 4)],
                    ctr_total = [1, 1, 1, 1, 5, 5, 5, 53, 1, 5, 0, 5, 2, 0, 2, 0, 2, 5, 3, 0, 0, 4]
                ),
                5445777084271881924 :
                catboost_ctr_value_table(
                    index_hash_viewer = {17151879688829397503 : 2, 18446744073709551615 : 0, 14474606344715696898 : 3, 14282620878612260867 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 12420782654419932198 : 4, 18446744073709551615 : 0, 15473953381485119304 : 6, 18446744073709551615 : 0, 9551523844202795562 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10742856347075653999 : 5},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.8026e-45, count = 40), catboost_ctr_mean_history(sum = 2.66247e-44, count = 21), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [2, 40, 19, 21, 0, 6, 0, 1, 0, 6, 1, 4, 0, 1]
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
                5840538188647484189 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14437050659463898499 : 15, 13712861078413872003 : 27, 18446744073709551615 : 0, 10471866573136752518 : 39, 3339193297886510343 : 33, 18446744073709551615 : 0, 18446744073709551615 : 0, 6522474812938725258 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2143292629466310926 : 41, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7089305956521872786 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 6692985183842031253 : 38, 18446744073709551615 : 0, 6568726319216336023 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 14184963790483298970 : 17, 8066257757148410395 : 12, 17298463301634926620 : 2, 5557686758182214811 : 50, 6932391217975877918 : 5, 151985887108509214 : 25, 8634520787218841888 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17067110762478402855 : 34, 18446744073709551615 : 0, 322496258011815209 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 905552284316133676 : 32, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1455867562654053300 : 21, 18446744073709551615 : 0, 9563528327934404534 : 4, 15234196598318321335 : 47, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11590587107156578237 : 13, 18446744073709551615 : 0, 8031909129594746559 : 16, 6922172069111294656 : 48, 9734038698837710529 : 30, 18446744073709551615 : 0, 18446744073709551615 : 0, 12449993409784492100 : 20, 18446744073709551615 : 0, 14152122979027456070 : 42, 8600131001622206919 : 45, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8140346942904631631 : 24, 12703712228892337104 : 51, 18446744073709551615 : 0, 18446744073709551615 : 0, 9181895285552204755 : 29, 18446744073709551615 : 0, 18446744073709551615 : 0, 8504011841664173526 : 9, 18446744073709551615 : 0, 10206141410907137496 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9505565985399112924 : 23, 2305054716417383901 : 19, 9352405656455510750 : 6, 15202963607546217823 : 31, 7650276087212546780 : 44, 13923650265588858465 : 46, 13307679510447017953 : 49, 12613343166795003362 : 8, 5168754957168326627 : 1, 1511139700538854501 : 36, 18446744073709551615 : 0, 18446744073709551615 : 0, 13671344409206559848 : 10, 5002941664428245224 : 43, 15373473978449523818 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3661062170767697137 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 17506660474440175860 : 37, 15791627448755489013 : 52, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2129442972428039162 : 14, 18446744073709551615 : 0, 3831572541671003132 : 22, 18446744073709551615 : 0, 8194820753884735230 : 26, 6592600901344044030 : 53},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.8026e-45, count = 5), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 5.60519e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [2, 5, 2, 0, 2, 3, 0, 2, 0, 2, 0, 4, 1, 2, 0, 2, 0, 1, 0, 1, 1, 4, 1, 1, 0, 1, 0, 1, 2, 2, 4, 0, 1, 3, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 5, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 5, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
                ),
                6317293569456956330 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 3581428127016485793 : 7, 1236773280081879954 : 6, 17856817611009672707 : 3, 18446744073709551615 : 0, 14455983217430950149 : 4, 18446744073709551615 : 0, 18336378346035991543 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 8312525161425951098 : 2, 5967870314491345259 : 1, 2436149079269713547 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 5.1848e-44, count = 41), catboost_ctr_mean_history(sum = 1.68156e-44, count = 2), catboost_ctr_mean_history(sum = 5.60519e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [37, 41, 12, 2, 4, 3, 1, 1]
                ),
                8405694746487331109 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 8473802870189803490 : 0, 7071392469244395075 : 3, 18446744073709551615 : 0, 8806438445905145973 : 2, 619730330622847022 : 1, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.14906e-43, count = 12), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6)],
                    ctr_total = [82, 12, 1, 6]
                ),
                8405694746487331111 :
                catboost_ctr_value_table(
                    index_hash_viewer = {2136296385601851904 : 0, 7428730412605434673 : 5, 9959754109938180626 : 2, 14256903225472974739 : 3, 8056048104805248435 : 1, 18446744073709551615 : 0, 12130603730978457510 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10789443546307262781 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.30321e-43, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2)],
                    ctr_total = [93, 2, 1, 1, 1, 2, 1]
                ),
                8405694746487331128 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 13987540656699198946 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18089724839685297862 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10353740403438739754 : 1, 3922001124998993866 : 0, 13686716744772876732 : 4, 18293943161539901837 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 5.1848e-44, count = 42), catboost_ctr_mean_history(sum = 1.82169e-44, count = 2), catboost_ctr_mean_history(sum = 5.60519e-45, count = 3)],
                    ctr_total = [37, 42, 13, 2, 4, 3]
                ),
                8405694746487331129 :
                catboost_ctr_value_table(
                    index_hash_viewer = {7537614347373541888 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5903587924673389870 : 4, 18278593470046426063 : 9, 10490918088663114479 : 8, 18446744073709551615 : 0, 407784798908322194 : 5, 5726141494028968211 : 6, 1663272627194921140 : 7, 8118089682304925684 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15431483020081801594 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 1403990565605003389 : 0, 3699047549849816830 : 1, 14914630290137473119 : 2},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.66247e-44, count = 2), catboost_ctr_mean_history(sum = 1.4013e-44, count = 20), catboost_ctr_mean_history(sum = 3.92364e-44, count = 5), catboost_ctr_mean_history(sum = 5.60519e-45, count = 3), catboost_ctr_mean_history(sum = 4.2039e-45, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2)],
                    ctr_total = [19, 2, 10, 20, 28, 5, 4, 3, 3, 4, 1, 2]
                ),
                8405694746487331131 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 14452488454682494753 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1388452262538353895 : 8, 8940247467966214344 : 2, 4415016594903340137 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 41084306841859596 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8678739366408346384 : 1, 18446744073709551615 : 0, 4544226147037566482 : 11, 14256903225472974739 : 5, 16748601451484174196 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5913522704362245435 : 3, 1466902651052050075 : 7, 2942073219785550491 : 12, 15383677753867481021 : 6, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.24208e-44, count = 6), catboost_ctr_mean_history(sum = 1.4013e-44, count = 11), catboost_ctr_mean_history(sum = 1.54143e-44, count = 6), catboost_ctr_mean_history(sum = 2.24208e-44, count = 2), catboost_ctr_mean_history(sum = 1.82169e-44, count = 5), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1)],
                    ctr_total = [16, 6, 10, 11, 11, 6, 16, 2, 13, 5, 3, 1, 1]
                ),
                8405694746487331134 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 15379737126276794113 : 5, 18446744073709551615 : 0, 14256903225472974739 : 3, 18048946643763804916 : 6, 2051959227349154549 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7024059537692152076 : 4, 18446744073709551615 : 0, 15472181234288693070 : 1, 8864790892067322495 : 2},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.26117e-44, count = 7), catboost_ctr_mean_history(sum = 9.52883e-44, count = 6), catboost_ctr_mean_history(sum = 7.00649e-45, count = 2)],
                    ctr_total = [9, 7, 68, 6, 5, 2, 4]
                ),
                8405694746995314031 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 14806117600143412865 : 45, 18446744073709551615 : 0, 18446744073709551615 : 0, 9012783753182299908 : 1, 1339560154066889221 : 11, 18446744073709551615 : 0, 174039779367851655 : 44, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16747653203520223500 : 5, 9579614896765447436 : 31, 18446744073709551615 : 0, 10954319356559110543 : 20, 18446744073709551615 : 0, 8837409026973740817 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8435628674313196821 : 25, 4328253920238227477 : 46, 13255493307163486358 : 22, 8244402790997920151 : 47, 2642294827352083864 : 3, 5465902517104623256 : 32, 13570935929574364571 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3313905660816128800 : 48, 18446744073709551615 : 0, 18446744073709551615 : 0, 7407438120481433891 : 43, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17847171115139092141 : 7, 15352833196308676269 : 41, 18446744073709551615 : 0, 11105815433168948272 : 2, 15759256038042246704 : 9, 12053837268177979184 : 27, 18446744073709551615 : 0, 15741774027452260020 : 49, 16471921548367724725 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 11672646576752095800 : 52, 18446744073709551615 : 0, 14476844306585554106 : 33, 18446744073709551615 : 0, 16788877028614866876 : 16, 18446744073709551615 : 0, 15961363380463923774 : 51, 18446744073709551615 : 0, 9163820133884450112 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18319273827208519108 : 10, 15824935908378103236 : 24, 18446744073709551615 : 0, 12525939980247406151 : 39, 18446744073709551615 : 0, 18446744073709551615 : 0, 17465538072756892362 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 11672204225795779405 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 4154101254308508496 : 30, 15065508147794825296 : 53, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5039862321041353814 : 26, 18446744073709551615 : 0, 2097978199290547160 : 23, 17693272547789792473 : 12, 15903257226085232346 : 13, 8979058744169729499 : 29, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12232697743793072097 : 50, 4691563912245177186 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 5796355511978061541 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11944704957972384744 : 37, 18446744073709551615 : 0, 555672821750051434 : 17, 6151371111011271787 : 15, 16407862886888389612 : 54, 14146391311712115821 : 21, 6363186655561209069 : 38, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14618971660234054515 : 18, 8613402368625482612 : 34, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3197496110909415801 : 36, 4051465155563377018 : 42, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2604489874933595135 : 19},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 5.60519e-45, count = 5), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6), catboost_ctr_mean_history(sum = 4.2039e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.26117e-44, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 7), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [4, 5, 2, 2, 1, 6, 3, 3, 1, 1, 2, 1, 9, 1, 1, 7, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 3, 1, 5, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
                ),
                8628341152511840406 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 3581428127016485793 : 2, 1236773280081879954 : 3, 16151796118569799858 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13125720576600207402 : 4, 5967870314491345259 : 1, 9724886183021484844 : 5, 18446744073709551615 : 0, 13605281311626526238 : 6, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 5.1848e-44, count = 2), catboost_ctr_mean_history(sum = 1.82169e-44, count = 40), catboost_ctr_mean_history(sum = 2.8026e-45, count = 4)],
                    ctr_total = [37, 2, 13, 40, 2, 4, 3]
                ),
                9867321491374199501 :
                catboost_ctr_value_table(
                    index_hash_viewer = {5321795528652759552 : 3, 1871794946608052991 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 2572990630596346628 : 14, 9755089559480497988 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14488270330580782411 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10281901548337195535 : 19, 18446744073709551615 : 0, 6052518548450009169 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12538518194927513684 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9204844949746414424 : 15, 10052892563062224857 : 6, 3493345142105552026 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 14486186593889963293 : 21, 7304087665005811933 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 772623291280696100 : 20, 18446744073709551615 : 0, 15587441985908139302 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10783615582859474474 : 4, 14429922730142217643 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8224442176515017331 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 5550804513927730230 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7807421160518048379 : 7, 14505127246450782459 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 4473747915336949119 : 10},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 4.2039e-45, count = 7), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 9), catboost_ctr_mean_history(sum = 0, count = 9), catboost_ctr_mean_history(sum = 2.8026e-45, count = 11), catboost_ctr_mean_history(sum = 1.4013e-45, count = 7), catboost_ctr_mean_history(sum = 7.00649e-45, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 13), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [3, 7, 1, 0, 2, 1, 3, 9, 0, 9, 2, 11, 1, 7, 5, 2, 0, 2, 2, 13, 0, 1, 0, 1, 0, 3, 0, 1, 0, 2, 1, 2, 0, 4, 0, 1, 0, 1, 1, 0, 0, 2, 1, 0]
                ),
                9867321491374199502 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 10934650013725255009 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 5844492600899280932 : 2, 18446744073709551615 : 0, 1034166431492604838 : 1, 18446744073709551615 : 0, 6203552979315789704 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 1113395566489815627 : 0, 13957701839509617452 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18034029854971645104 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 9226604805100152147 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13302932820562179799 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15316838452862012827 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5765263465902070143 : 7},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 10), catboost_ctr_mean_history(sum = 1.54143e-44, count = 7), catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 0, count = 10), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 1.12104e-44, count = 10), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 10, 11, 7, 0, 12, 0, 12, 0, 10, 1, 1, 0, 8, 8, 10, 2, 2, 0, 6, 0, 1]
                ),
                10041049327410906820 :
                catboost_ctr_value_table(
                    index_hash_viewer = {16259707375369223360 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13847085545544291780 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7654620248348398600 : 2, 18446744073709551615 : 0, 9243796653651753418 : 5, 18446744073709551615 : 0, 1681026541770505292 : 22, 1292491219513334285 : 21, 13677090684479491854 : 23, 6494991755595340494 : 15, 7494438315637327440 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18410271455579776277 : 14, 6336919059871405781 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9974519673449003035 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5899717636280359390 : 13, 18446744073709551615 : 0, 15904544917366469984 : 1, 18446744073709551615 : 0, 862592111642406882 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 18161161563956788133 : 11, 18446744073709551615 : 0, 3340544229935902247 : 12, 18446744073709551615 : 0, 14827488318775688873 : 16, 15675535932091499306 : 3, 18446744073709551615 : 0, 15230422751883885548 : 24, 18446744073709551615 : 0, 1662085889209686126 : 27, 18446744073709551615 : 0, 1062699037197581552 : 4, 14072903496117963889 : 17, 18446744073709551615 : 0, 15434641073738489523 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14277121817972567864 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 18160464660109825851 : 9, 16406258951888748923 : 18, 17480885798804750972 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.68156e-44, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 11), catboost_ctr_mean_history(sum = 1.54143e-44, count = 8), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 1.54143e-44, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 7), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 5.60519e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [12, 2, 3, 11, 11, 8, 1, 5, 11, 2, 1, 7, 2, 2, 3, 2, 2, 2, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1]
                ),
                12606205885276083425 :
                catboost_ctr_value_table(
                    index_hash_viewer = {14591795653440117248 : 7, 3812458928802352640 : 15, 14931585970071951136 : 3, 16031103881690819777 : 2, 18446744073709551615 : 0, 10918373804899154693 : 14, 2002444088171013702 : 9, 18446744073709551615 : 0, 11300006847281354472 : 13, 6619561440864924457 : 1, 3223795087593081450 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16939442126893167761 : 11, 18446744073709551615 : 0, 8668830525758017779 : 12, 18446744073709551615 : 0, 12990050366695140501 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16503206593760246744 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10128637724524112380 : 8, 13556881510278288029 : 10, 15649470839308619998 : 4, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 8.40779e-45, count = 10), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 1.4013e-45, count = 8), catboost_ctr_mean_history(sum = 0, count = 11), catboost_ctr_mean_history(sum = 2.8026e-45, count = 8), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 2.8026e-45, count = 13), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 9.80909e-45, count = 6), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [6, 10, 0, 6, 1, 8, 0, 11, 2, 8, 1, 5, 2, 13, 0, 2, 7, 6, 1, 4, 0, 3, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
                ),
                12606205885276083426 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 3581428127016485793 : 8, 18446744073709551615 : 0, 17856817611009672707 : 3, 18446744073709551615 : 0, 14455983217430950149 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5967870314491345259 : 1, 2436149079269713547 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1236773280081879954 : 7, 16151796118569799858 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18336378346035991543 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 8312525161425951098 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 36), catboost_ctr_mean_history(sum = 2.94273e-44, count = 20), catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 36, 21, 20, 0, 12, 0, 2, 0, 4, 0, 3, 0, 1, 1, 0, 0, 1]
                ),
                12627245789391619613 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9226604805100152147 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 1034166431492604838 : 1, 13302932820562179799 : 2, 6203552979315789704 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 1113395566489815627 : 0, 13957701839509617452 : 7, 15316838452862012827 : 3, 18446744073709551615 : 0, 5765263465902070143 : 6},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 22), catboost_ctr_mean_history(sum = 2.52234e-44, count = 17), catboost_ctr_mean_history(sum = 0, count = 22), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 13), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 22, 18, 17, 0, 22, 1, 1, 0, 13, 2, 3, 1, 0, 0, 1]
                ),
                12627245789391619615 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 3581428127016485793 : 2, 1236773280081879954 : 1, 16151796118569799858 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18336378346035991543 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 13125720576600207402 : 3, 5967870314491345259 : 6, 9724886183021484844 : 4, 18446744073709551615 : 0, 13605281311626526238 : 5, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 37), catboost_ctr_mean_history(sum = 2.94273e-44, count = 20), catboost_ctr_mean_history(sum = 0, count = 13), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 37, 21, 20, 0, 13, 0, 2, 0, 4, 0, 2, 1, 0, 0, 1]
                ),
                13000966989535245561 :
                catboost_ctr_value_table(
                    index_hash_viewer = {1757986209816306368 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14449357781033878404 : 5, 17335471473308721348 : 3, 15684611358642908806 : 14, 18446744073709551615 : 0, 11580098970816741768 : 2, 80059902472028169 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15655322029177927125 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6489083244225771673 : 13, 12786063218960790489 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 3619824434008635037 : 1, 2160949785446258526 : 8, 1968964319342822495 : 9, 4408800825433526368 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1539542015216389732 : 0, 3160296822215680932 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2730874518089248169 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6849001936407276463 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15200819853968089276 : 6, 6270639049625855037 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 23), catboost_ctr_mean_history(sum = 2.66247e-44, count = 8), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 2.8026e-45, count = 9), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 23, 19, 8, 0, 6, 2, 9, 0, 5, 0, 2, 0, 2, 0, 5, 0, 1, 0, 1, 0, 5, 0, 1, 0, 2, 0, 2, 0, 1, 0, 4, 1, 0, 0, 1, 0, 1]
                ),
                13902559248212744134 :
                catboost_ctr_value_table(
                    index_hash_viewer = {8975491433706742463 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14435487234778955461 : 25, 26794562384612742 : 22, 18446744073709551615 : 0, 4411634050168915016 : 5, 11361933621181601929 : 1, 15118949489711741514 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 488596013123191629 : 7, 2041917558348994126 : 19, 18446744073709551615 : 0, 3099115351550504912 : 26, 13955926499752636625 : 6, 6798076237643774482 : 20, 10555092106173914067 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4633306462361102487 : 17, 4428359745823853592 : 29, 16982002041722229081 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14612285549902308191 : 15, 18446744073709551615 : 0, 9142731084578380321 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 240279460452550314 : 28, 779318031744854123 : 11, 15286189140583379372 : 16, 4020317248344823341 : 23, 18446744073709551615 : 0, 18446744073709551615 : 0, 6630836586772136624 : 18, 18446744073709551615 : 0, 3266355002422142770 : 27, 15927023829150890738 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 896638510602221880 : 0, 2066979203234309177 : 3, 16388825279889469625 : 14, 18446744073709551615 : 0, 6364972095279429180 : 12, 18446744073709551615 : 0, 18348953501661188798 : 10, 18144006785123939903 : 24},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 7.00649e-45, count = 18), catboost_ctr_mean_history(sum = 0, count = 24), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [0, 1, 1, 1, 0, 1, 5, 18, 0, 24, 0, 7, 0, 1, 0, 2, 0, 1, 0, 2, 2, 0, 2, 0, 0, 2, 0, 1, 0, 1, 0, 3, 0, 1, 3, 0, 0, 1, 0, 2, 5, 0, 0, 1, 0, 3, 1, 0, 2, 0, 0, 1, 0, 2, 0, 2, 0, 1, 1, 0]
                ),
                13902559248212744135 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16479676762461049221 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14314906987178377226 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12130212770433783695 : 4, 18446744073709551615 : 0, 4054001010745510673 : 30, 18446744073709551615 : 0, 18446744073709551615 : 0, 9965442995151111700 : 13, 16548297050623529236 : 17, 1889231235462838678 : 20, 18446744073709551615 : 0, 11147526993393187224 : 28, 18446744073709551615 : 0, 14555653527613724826 : 8, 12522231453186850331 : 23, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10958843647676541603 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8794073872393869608 : 35, 8589127155856620713 : 36, 18446744073709551615 : 0, 18446744073709551615 : 0, 11748579728051583916 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18384113673385397171 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17769648050045596984 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13820256289847569724 : 2, 13621749364805718972 : 26, 1878905203052656190 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11450539798027648834 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15254761925720908613 : 10, 18446744073709551615 : 0, 2398222922681060807 : 29, 18446744073709551615 : 0, 18446744073709551615 : 0, 6227746613267076298 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18358311891254868174 : 37, 4062976837984404303 : 11, 3858030121447155408 : 32, 7449767364017353680 : 33, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13653016638975931866 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14219847782559079394 : 21, 9089159255438104419 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16861309804843249000 : 31, 6719442763618183529 : 18, 16481986878556141930 : 25, 9655990399021251947 : 34, 18446744073709551615 : 0, 18446744073709551615 : 0, 11030694858814915054 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2718670329607888375 : 3, 7719283207639011575 : 27, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4940085441777621244 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 9), catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 1, 1, 0, 2, 1, 0, 5, 0, 7, 0, 9, 0, 8, 2, 2, 1, 5, 0, 12, 0, 2, 1, 0, 1, 0, 1, 0, 2, 4, 0, 1, 0, 2, 1, 4, 1, 4, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 2, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
                ),
                15655841788288703925 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 3581428127016485793 : 2, 1236773280081879954 : 3, 16151796118569799858 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13125720576600207402 : 4, 5967870314491345259 : 1, 9724886183021484844 : 5, 18446744073709551615 : 0, 13605281311626526238 : 6, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 37), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 13), catboost_ctr_mean_history(sum = 2.66247e-44, count = 19), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 3)],
                    ctr_total = [0, 37, 3, 1, 0, 13, 19, 19, 0, 2, 0, 4, 0, 3]
                ),
                17677952491747546147 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17787954881284471813 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7454420046185256717 : 20, 16256335682944813838 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1636731659193698578 : 48, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5922800847845598742 : 22, 14182197490569975831 : 27, 7624930417088562712 : 52, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10422205982269444643 : 44, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3411314423057176877 : 30, 18446744073709551615 : 0, 18446744073709551615 : 0, 4522605207985801776 : 29, 18446744073709551615 : 0, 13192676729576349746 : 62, 16466569643076362291 : 8, 18300934243650069811 : 58, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4431368220400894274 : 60, 18446744073709551615 : 0, 18446744073709551615 : 0, 14233673023285815109 : 50, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2899749022061236299 : 53, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8023290181753164880 : 65, 9933882341717515345 : 66, 3233597379123467602 : 47, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8402263143377857370 : 37, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17651556054977644126 : 21, 15680080812126751838 : 55, 17708725746287261024 : 28, 18446744073709551615 : 0, 1780070264439091554 : 19, 15773274901763725923 : 0, 16328374789029446500 : 51, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16177106547947603049 : 13, 18446744073709551615 : 0, 17879236117190567019 : 3, 3489127981302646635 : 41, 14241655703424067948 : 56, 15943785272667031918 : 43, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9771448801094703501 : 67, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11530748061647284369 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 5994047302704556692 : 57, 18446744073709551615 : 0, 18446744073709551615 : 0, 10117199296271121559 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 9999128863307626394 : 5, 18446744073709551615 : 0, 11701258432550590364 : 6, 7854656800704835228 : 39, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 118997543255608737 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 10779812027622989220 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 6111396989577705639 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16127325828303939500 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 5576091289432675759 : 49, 14224606228188042159 : 59, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14966077412008197812 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 10025163577623610551 : 32, 1755789550731085240 : 64, 7501413217152384697 : 14, 16355005890516862393 : 46, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14797650915799523780 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13730933025438975688 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 724243645116964305 : 42, 18446744073709551615 : 0, 11702735195037717203 : 31, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16072394239031333591 : 45, 18446744073709551615 : 0, 11159883566315996889 : 34, 11603752796664724186 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16142728109259286750 : 10, 18446744073709551615 : 0, 17844857678502250720 : 12, 18446744073709551615 : 0, 9628264367976338914 : 16, 15813441649188061154 : 61, 18446744073709551615 : 0, 18446744073709551615 : 0, 2145056323740669926 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9516068082126538479 : 54, 18446744073709551615 : 0, 18446744073709551615 : 0, 10037970161273910770 : 38, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17560274819071548920 : 23, 11038948726666369272 : 25, 18446744073709551615 : 0, 8596718462362217979 : 63, 18446744073709551615 : 0, 10298848031605181949 : 33, 16215728555360712189 : 36, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 2, 1, 0, 0, 1, 3, 4, 0, 4, 0, 3, 0, 4, 0, 3, 0, 1, 0, 1, 0, 1, 0, 4, 0, 1, 1, 6, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
                ),
                17677952493224740166 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 1799168355831033313 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11936664559898134054 : 11, 14666845749088704071 : 7, 18429784838380727208 : 1, 17027374437435318793 : 13, 2862173265672040777 : 3, 16080065667299791243 : 0, 14677655266354382828 : 12, 12391839889973628461 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4082592020331177586 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 315676340386518075 : 5, 18446744073709551615 : 0, 10716245805238997245 : 2, 9313835404293588830 : 9, 17603450378469852574 : 6},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.8026e-45, count = 6), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 1.4013e-44, count = 46), catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [2, 6, 1, 4, 10, 46, 0, 8, 1, 3, 0, 1, 5, 0, 2, 0, 0, 2, 0, 4, 0, 2, 0, 2, 1, 0, 0, 1]
                ),
                17677952493224740167 :
                catboost_ctr_value_table(
                    index_hash_viewer = {7515733889724454912 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 2160905354121516547 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13659069800549444297 : 3, 7791826943727985930 : 2, 7884511582485373322 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18022007786552474063 : 19, 18446744073709551615 : 0, 6068383991325515601 : 25, 7524725216182310545 : 24, 17609669744399151123 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 11681580651965248598 : 15, 576145588900686679 : 22, 13155646805788779928 : 0, 18446744073709551615 : 0, 5849831644443487770 : 5, 3372332782322797723 : 17, 18446744073709551615 : 0, 9865453060805390877 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9800431588293194596 : 10, 9048109927352371876 : 11, 16801589031893337254 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2099530300070748010 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 4741992351141480365 : 21, 17321493568029573614 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2151914027663660914 : 6, 9012245698387122739 : 20, 3718664820244579636 : 23, 2925864759981622644 : 1, 15505365976869715893 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 0, count = 13), catboost_ctr_mean_history(sum = 1.4013e-44, count = 15), catboost_ctr_mean_history(sum = 0, count = 17), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 9), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 5.60519e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 4, 1, 2, 0, 13, 10, 15, 0, 17, 0, 1, 0, 9, 0, 2, 0, 1, 4, 0, 1, 0, 1, 0, 0, 2, 0, 2, 0, 3, 0, 1, 0, 1, 0, 2, 1, 0, 0, 1, 0, 1, 1, 0, 2, 0, 0, 1, 1, 0, 0, 1]
                ),
                17677952493260528848 :
                catboost_ctr_value_table(
                    index_hash_viewer = {3632340108106778112 : 12, 84580555217079201 : 5, 1856503610704726976 : 8, 12055230997206289283 : 2, 16771526449192646880 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3152779373080459276 : 4, 14225011642249373260 : 9, 18198689053211288334 : 6, 16796278652265879919 : 13, 4201158457639332815 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9710576150271683444 : 1, 6178854915050051732 : 0, 8308165749326275029 : 11, 4776444514104643317 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 28), catboost_ctr_mean_history(sum = 2.94273e-44, count = 17), catboost_ctr_mean_history(sum = 0, count = 11), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 28, 21, 17, 0, 11, 0, 2, 0, 2, 0, 1, 0, 2, 0, 5, 0, 3, 0, 2, 0, 4, 1, 0, 0, 1, 0, 1]
                ),
                17677952493260528850 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 5717724512618337697 : 2, 18446744073709551615 : 0, 5133782457465682915 : 12, 11196527390020060580 : 8, 11961955270333222981 : 9, 5761100149665496677 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15262016962202059306 : 3, 18446744073709551615 : 0, 11861182568623336748 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 12026216826389142735 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 3373069665683731858 : 1, 18288092504171651762 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13367377011060337464 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17153616595626919517 : 11, 15741577697228378142 : 6, 17780934287826733279 : 5},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 34), catboost_ctr_mean_history(sum = 2.8026e-44, count = 19), catboost_ctr_mean_history(sum = 0, count = 13), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 34, 20, 19, 0, 13, 0, 2, 0, 2, 0, 1, 0, 3, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
                ),
                17677952493260528854 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17985374731566054150 : 24, 18446744073709551615 : 0, 4969880389554839688 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1883285504791108373 : 36, 14139902777924824981 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 17540248381108153753 : 27, 18446744073709551615 : 0, 2120068639763588379 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 1277857586923739550 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9915512646490226338 : 3, 18446744073709551615 : 0, 5780999427119446436 : 30, 15493676505554854693 : 29, 14453653496344422438 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3622512433858345389 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9415440463389949361 : 19, 18446744073709551615 : 0, 15689261734764374707 : 26, 17838331352489460532 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18403625549429831228 : 12, 18446744073709551615 : 0, 16192880425411659454 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6383791411268594626 : 20, 18033916581698980546 : 34, 18446744073709551615 : 0, 11961955270333222981 : 8, 18446744073709551615 : 0, 11191788834073534919 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5730630563994427981 : 23, 647125264645798733 : 37, 16620451033949360975 : 10, 17618698769621849933 : 38, 7150295984444125389 : 17, 18446744073709551615 : 0, 12157540499542742995 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1072059942279220057 : 25, 10177020748048094298 : 14, 18446744073709551615 : 0, 9494950831378731228 : 33, 18446744073709551615 : 0, 518361807174415198 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 592499207252901221 : 7, 4098784705883188966 : 31, 10062654256758136807 : 32, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3618574749222493677 : 5, 18446744073709551615 : 0, 13088729798727729263 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2625225542620233849 : 13, 6645299512826462586 : 4, 5651789874985220091 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 8.40779e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 9.80909e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 5, 6, 1, 0, 4, 0, 2, 0, 5, 0, 7, 0, 3, 0, 1, 0, 2, 0, 2, 2, 7, 0, 1, 0, 1, 7, 1, 2, 2, 0, 2, 0, 2, 0, 1, 0, 2, 0, 1, 0, 3, 0, 4, 1, 1, 0, 1, 0, 1, 0, 1, 2, 4, 0, 3, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                ),
                17677952493261641995 :
                catboost_ctr_value_table(
                    index_hash_viewer = {7458091914254611456 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4709287016198198532 : 9, 11891385945082349892 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16624566716182634315 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8188814934051861073 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4569428324804022359 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 5629641527707403930 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11910326597643169058 : 11, 16012272658388189795 : 14, 7930141458505850467 : 19, 16604351646315406629 : 16, 17723738371509991206 : 4, 1862677700213432292 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16566219115744069547 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11902478327942383792 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 13377843633458007987 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 7687100899529582134 : 1, 10629038735401595063 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9943717546119900283 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6610044300938801023 : 6},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.8026e-45, count = 15), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 9.80909e-45, count = 3), catboost_ctr_mean_history(sum = 5.60519e-45, count = 15), catboost_ctr_mean_history(sum = 5.60519e-45, count = 22), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [2, 15, 2, 0, 7, 3, 4, 15, 4, 22, 0, 5, 0, 2, 0, 1, 0, 3, 0, 2, 0, 4, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
                ),
                17677952493261641996 :
                catboost_ctr_value_table(
                    index_hash_viewer = {16259707375369223360 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13847085545544291780 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7654620248348398600 : 2, 18446744073709551615 : 0, 9243796653651753418 : 5, 18446744073709551615 : 0, 1681026541770505292 : 22, 1292491219513334285 : 21, 13677090684479491854 : 23, 6494991755595340494 : 15, 7494438315637327440 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18410271455579776277 : 14, 6336919059871405781 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9974519673449003035 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5899717636280359390 : 13, 18446744073709551615 : 0, 15904544917366469984 : 1, 18446744073709551615 : 0, 862592111642406882 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 18161161563956788133 : 11, 18446744073709551615 : 0, 3340544229935902247 : 12, 18446744073709551615 : 0, 14827488318775688873 : 16, 15675535932091499306 : 3, 18446744073709551615 : 0, 15230422751883885548 : 24, 18446744073709551615 : 0, 1662085889209686126 : 27, 18446744073709551615 : 0, 1062699037197581552 : 4, 14072903496117963889 : 17, 18446744073709551615 : 0, 15434641073738489523 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14277121817972567864 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 18160464660109825851 : 9, 16406258951888748923 : 18, 17480885798804750972 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 5.60519e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 11), catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 5.60519e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 9.80909e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [0, 12, 2, 0, 0, 3, 4, 7, 0, 11, 0, 8, 0, 1, 0, 5, 4, 7, 0, 2, 0, 1, 7, 0, 0, 2, 0, 2, 0, 3, 0, 2, 1, 1, 0, 2, 3, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
                ),
                17677952493263343771 :
                catboost_ctr_value_table(
                    index_hash_viewer = {15330345801530070271 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13871343560304450565 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 17989766274143549768 : 22, 18334501489220455433 : 24, 17271881404906880906 : 17, 1327065643761606346 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5745149923951351887 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18147836298725285973 : 23, 11919737177904201494 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 220392991226246300 : 8, 11009125960592947549 : 19, 16732756202475478686 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1799168355831033313 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17299587346058626214 : 16, 945432601379406567 : 4, 18446744073709551615 : 0, 227547732142737705 : 3, 8878683662908522218 : 5, 18371399316525749547 : 15, 18446744073709551615 : 0, 12391839889973628461 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 4342739523005943472 : 21, 18446744073709551615 : 0, 10362267276645262642 : 1, 6966500923373419635 : 7, 9445514806491669746 : 18, 10820219266285332853 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17559172457516014783 : 14},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 8.40779e-45, count = 8), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 2.8026e-45, count = 5), catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 2.8026e-45, count = 8), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 2.8026e-45, count = 11), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 9.80909e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [6, 8, 0, 6, 2, 5, 0, 8, 2, 8, 0, 1, 1, 3, 2, 11, 0, 1, 7, 6, 0, 2, 1, 3, 0, 2, 0, 2, 0, 3, 0, 1, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                ),
                17677952493265578087 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18171586759681088672 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14557079040021784102 : 1, 1894223316800506727 : 9, 18446744073709551615 : 0, 11879805695908083497 : 2, 11687820229804647466 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12879152732677505903 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4426716004344559893 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8230941806183355321 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 1064533880431424572 : 5, 17607571949008043997 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.8026e-44, count = 58), catboost_ctr_mean_history(sum = 0, count = 11), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [20, 58, 0, 11, 0, 1, 0, 2, 0, 3, 1, 0, 0, 1, 1, 0, 0, 2, 0, 1]
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



def city_hash_uint64(string):
    if type(string) is not str:
        string = str(string)
    out = CityHash64(string) & 0xffffffff
    if (out > 0x7fFFffFF):
        out -= 0x100000000
    return out


### Applicator for the CatBoost model

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

    assert len(float_features) == model.float_feature_count
    assert len(cat_features) == model.cat_feature_count

    # Binarise features
    binary_features = [0] * model.binary_feature_count
    binary_feature_index = 0

    for i in range(len(model.float_feature_borders)):
        for border in model.float_feature_borders[i]:
            binary_features[binary_feature_index] += 1 if (float_features[i] > border) else 0
        binary_feature_index += 1
    transposed_hash = [0] * model.cat_feature_count
    for i in range(model.cat_feature_count):
        transposed_hash[i] = city_hash_uint64(cat_features[i])

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



