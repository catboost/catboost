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
    binary_feature_count = 17
    tree_count = 40
    float_feature_borders = [
        [20.5, 25, 29.5, 32, 34.5, 35.5, 36.5, 38.5, 45.5, 54.5, 57.5, 61.5],
        [132707.5, 159204.5, 168783.5, 185890.5, 197408.5, 203488.5, 218145.5, 243690.5, 303732.5, 325462, 350449, 421435.5, 678993],
        [4.5, 5.5, 6.5, 8, 10.5, 11.5, 12.5, 13.5, 15.5],
        [1087, 3280, 5842, 7493, 17537.5],
        [1622.5, 1738, 1862, 1881.5, 1944.5, 2189.5, 2396],
        [11.5, 19, 22, 27, 35.5, 36.5, 38.5, 42, 44.5, 46.5, 49, 70]
    ]
    tree_depth = [2, 6, 6, 6, 6, 6, 3, 6, 6, 6, 6, 6, 6, 2, 5, 6, 6, 0, 2, 1, 1, 2, 5, 1, 3, 6, 6, 6, 6, 6, 6, 6, 0, 3, 6, 0, 3, 1, 6, 2]
    tree_split_border = [2, 6, 3, 8, 9, 4, 10, 1, 8, 2, 2, 6, 12, 4, 5, 8, 7, 7, 4, 4, 9, 6, 10, 6, 1, 6, 4, 1, 1, 3, 8, 5, 2, 8, 5, 2, 4, 6, 9, 2, 9, 5, 6, 6, 7, 2, 4, 9, 3, 10, 5, 3, 4, 3, 1, 5, 1, 2, 4, 12, 255, 5, 2, 2, 12, 2, 3, 7, 11, 8, 6, 11, 1, 6, 4, 6, 7, 4, 12, 7, 2, 5, 12, 1, 3, 7, 5, 11, 1, 5, 4, 8, 11, 2, 6, 4, 1, 1, 2, 13, 10, 6, 6, 11, 11, 1, 11, 2, 7, 3, 6, 2, 7, 12, 3, 5, 4, 3, 2, 7, 11, 2, 9, 1, 2, 1, 3, 8, 1, 4, 8, 4, 3, 3, 5, 7, 12, 2, 1, 1, 6, 11, 1, 7, 3, 4, 5, 3, 3, 1, 2, 6, 2, 1, 3, 4, 10, 1, 1, 3, 5, 3, 4, 1, 4, 7, 9, 8]
    tree_split_feature_index = [3, 4, 3, 0, 1, 3, 5, 5, 2, 3, 2, 1, 11, 5, 3, 5, 11, 1, 2, 4, 0, 5, 1, 2, 2, 4, 1, 5, 11, 2, 5, 3, 3, 2, 4, 11, 11, 4, 5, 5, 2, 5, 0, 11, 4, 0, 4, 5, 11, 0, 1, 2, 4, 0, 0, 11, 3, 2, 4, 11, 6, 0, 9, 10, 1, 3, 9, 8, 0, 1, 10, 11, 3, 2, 14, 9, 8, 10, 11, 5, 8, 9, 5, 4, 2, 10, 3, 0, 15, 0, 10, 11, 11, 14, 8, 9, 3, 4, 7, 1, 11, 8, 1, 11, 1, 15, 5, 4, 10, 5, 2, 2, 2, 0, 4, 8, 9, 9, 3, 0, 0, 1, 0, 10, 0, 8, 7, 2, 14, 0, 2, 8, 3, 14, 3, 5, 11, 15, 16, 3, 2, 5, 4, 0, 10, 7, 2, 1, 7, 12, 12, 2, 14, 7, 4, 14, 11, 1, 9, 15, 10, 8, 4, 13, 7, 2, 11, 2]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    one_hot_cat_feature_index = [9]
    one_hot_hash_values = [
        [-2114564283]
    ]
    ctr_feature_borders = [
        [0.999998987, 2.99999905, 10.999999, 11.999999],
        [1.99999905, 3.99999905, 5.99999905, 8.99999905, 9.99999905, 11.999999, 13.999999],
        [3.99999905, 6.99999905, 8.99999905, 10.999999, 11.999999, 12.999999],
        [6.99999905, 8.99999905, 10.999999, 11.999999, 12.999999, 13.999999, 14.999999],
        [2.99999905, 3.99999905, 4.99999905, 5.99999905, 6.99999905, 7.99999905, 8.99999905, 9.99999905, 10.999999, 11.999999, 12.999999, 13.999999],
        [10.999999, 11.999999],
        [7.99999905],
        [1.99999905, 5.99999905, 7.99999905, 9.99999905],
        [8.99999905, 9.99999905, 12.999999],
        [10.999999]
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.02468749944819137, 0, 0, 0,
        0.007314843590639066, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02647276728040805, 0, 0.02129639376876518, 0, 0.02019951879200884, 0, 0.01448156218789984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01283958305738245, 0, 0.002555624952809885, 0, 0, 0, 0.009629687293036841, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0.0189767942412636, 0, 0, 0, 0, 0, 0, 0, 0.01639931989859468, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01142371603484725, 0, 0, 0, 0.01784897880188897, 0, 0, 0, 0.02449413747981692, 0.01144819302550946, 0, 0, 0.007163347203085207, 0, 0, 0, 0.0251866989363283, 0.007295676403921411, 0, 0, 0.007155120640943414, 0, 0, 0, 0.01493803990471317, -0.0005270731737590317, 0, 0, 0, 0, 0, 0, 0.01495451284813917, 0.003686149484403514, -0.0001588366001321735, 0,
        0.01863846400543752, 0, 0, 0, 0.01602156163210883, 0, 0.007029646482060887, 0, 0.007109622100262918, 0, 0, 0, 0.01123559304257172, 0, 0.007042961797089315, 0, 0.01956510065934012, 0, 0.01420709347856819, 0, 0.0201042203171601, 0, 0.008239907701306054, 0, 0.01747488880571386, 0, 0, 0, 0.02049746726122497, 0, 0.009046597757113313, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0002717582464768173, 0, -0.000234171361431056, 0, 0, 0, 0, 0, 0, 0, 0.00717858575614239, 0,
        0, 0, 0, 0.01578610857474072, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02250683049627574, 0.01369238901375851, 0.0213215063722589, 0.008544744434232528, 0.01093128101314685, 0, 0.01580837330144117, 0.01387498881707853, 0.01100553837525823, 0, 0.01585159744463972, 0.007976453829622865, 0.01108275545028801, 0, 0.008707705861519803, 0.01399792503867479, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0002697200596737982, 0, 0, 0, 0, 0, 0, 0, -1.345785283134202e-05, 0, 0, 0, 0,
        0, 0, 0.01082091042294551, 0.006937737138592066, 0, 0, 0.01345532825946575, 0.01357746900237084, 0, 0, 0.01343481239457261, 0.006781285540300392, 0, 0.006900714831452688, 0.01933589702792803, 0.02089515459773852, 0, 0, 0, 0, 0, 0, 0, 0.006817012840406221, 0, 0, 0.01086222137274153, 0, 0, 0, 0.008588194170420862, 0.01014698609445439, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0.0230622065633737, -0.0007389850185289174, 0.005138561794132496, -0.0003556242758615103, -0.000412716383739488, 0, 0.005483360177423291, 0,
        0, 0, 0, 0.01475099235693367, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0182476495557158, 0.02059214803198194, 0, 0.01923858902573345, 0, 0, 0, -0.0004096210109306288, 0.01045512824660739, 0.006959940624420446, 0, 0.009060512586790482, 0, 0, 0, -0.0001305845133562047, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.006902534334312409, 0, 0, 0, 0,
        0, 0, 0, 0, 0.02078290510629795, 0.01749171811583971, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.006254797180305098, 0.01673628750888337, 0.01004876401013248, 0.02084395237962923, 0.006366848053889308, 0.01930868101047137, 0.01073868393864335, 0.008413316161350709, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0005655352671070562, 0, 0.005155373179635901, 0, 0, 0, 0, 0, 0, 0, -0.0001296051295279241,
        0, 0, 0.01238554672873119, 0, 0.01240601198461524, 0, 0, 0, 0.006419550557894609, 0, 0.01006371668698282, 0.006393776974068205, 0, 0, 0, 0, 0.01540807997411586, 0.0101288312648348, 0.01751369496615774, 0.002479417266972542, 0.01249366501050774, 0, -0.0006341719506063576, 0.006548768743080029, 0.01529541521160124, 0, 0.01873051387008106, 0.008324516635950041, 0.009821312803700865, 0.006751411431551065, 0.004104169770109241, 0.003566785992872954, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0004452141514003497, -0.0005612937526985587, 0, 0, 0, 0, 0, 0, 0, 0.005346773551120776, 0, 0, 0, 0,
        0, 0, 0, 0.01411502093902556, 0.006060107097836049, 0, 0.006026533265227919, 0.006305234997782804, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0139987549739286, 0.02013091340256827, 0.01659235059475045, 0, 0.01988908598516424, 0.01391648495816914, 0, 0, 0, 0, 0, 0, 0, 0.001035226933120528, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.003202460709965662, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0.003419576790737231, 0, 0, 0, 0.01116390445283285, 0, 0, 0, 0.004165997478927516, 0, 0, 0, 0.001532831450266852, 0, -0.00046589350012737, 0, 0, 0, 0, 0, 0, 0, 0, 0.02004647840344661, 0.009753131139344872, 0.02006024279678595, 0.01208710847836926, 0.01776158996007026, 0.006542679952853596, 0.01793604460302113, -0.00117820525959095, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005838515961263281, 0, 0, 0, 0.01204961810927871, 0, 0.009406391103697623, 0,
        0.008608179488307987, -0.0007023731062800744, 0.007822624690305252, -0.00132723901103845, 0, 0, 0, 0, 0.006311550062226664, -0.0006601649773919133, 0, -0.0004056303481862837, 0, 0, 0, 0, 0.007293843909270122, 0, 0.01010961298306204, 0, 0, 0, 0, 0, 0, 0, -0.001345037874003682, 0, 0, 0, 0, 0, 0.0147058954758953, 0, 0.01953734887778817, 0, 0, 0, 0.01909315598759035, 0, 0, 0, 0.01498400494529643, -0.0002713651655436331, 0, 0, 0, 0, 0.00583218865766337, 0, 0.01638562399175868, 0, 0, 0, 0.01604615564211889, 0, 0, 0, 0.009610766765273953, 0, 0, 0, 0, 0,
        0.0214715581350155, 0.01048766307335349, 0.005619837288550077, -0.001628167677707009,
        0.005784893748970192, 0, 0.01151738729091962, 0.001908529396927527, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01109348170712972, 0.00890111974026524, 0.01627788555950123, 0.005586086976411244, 0.005356188301667429, 0, 0.01745798913564202, 0.01692423634402208, 0, 0, 0, 0, 0.005301137259130298, 0, 0.01785214387316218, 0.01533385993712314,
        0.005476350442175056, 0, 0.01110183636597215, 0.005675041802175975, 0.008517167341224841, 0.01122093696568546, 0.01111360503090515, 0.006788201943333229, 0, 0, 0, 0, 0.01777612284889007, 0, 0.01970971114286632, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005441554903043388, 0, 0, 0, 0, 0, 0, 0.004814402684443475, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0003993552313210477, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0.005435277814776788, 0.008430764410305316, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01100624661619311, 0.01067678807234482, 0.01878291287566486, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005398688998059978, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005928364905607377, 0.005853050250056947, 0.01093756259517355, 0.0166197838873686, 0, 0, 0, 0, 0, 0.001863695900974821, 0.008593987634620817, 0.01378352398889048, 0, 0, 0, -0.0003831713462859157,
        0.01503585574547744,
        0, 0.01801809108819051, 0.007709867638820007, 0.01730008459544116,
        0.01854745925478346, 0.007256784355642204,
        0.01121875377759088, 0.01355993943126421,
        0.008181633638042455, 0, 0.01397681705251495, 0.01709312055953389,
        0.01342162098676391, 0, 0, 0, 0.01490314927063936, 0, 0.004726022534425507, 0, 0, 0, 0, 0, 0.004494962806800468, 0, 0, 0, 0.00743298050290305, -0.002175723449116748, 0.002715738063591281, 0, 0.01533401642203459, 0.0006669419609643623, 0.001539419550810387, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0.009499213127585046, 0.01625558872329592,
        0.01577971732503594, 0.0147087365759171, 0.004113315724452522, 0.004817554365083441, 0, 0.01311095846577134, 0, 0.01108983042356618,
        0, 0.004750589809015993, 0, 0, 0, 0, 0, 0, 0, 0.01143670322302048, 0, 0, 0, 0, 0, 0, 0, 0.00953359732050472, 0, 0.002864732495708519, 0, -0.001282547009462461, 0, 0, 0.007039772707638463, 0.01494096223105057, 0, 0.007144687052904893, 0, 0.00450098539205263, 0, 0, 0, 0.005058750282783412, 0, 0, 0, 0, 0, 0, 0, 0.006911548019164175, 0, 0, 0, 0, 0, 0, 0, -0.002007563288879609, 0, -0.003299051196316456, 0, 0.005887244787957996, 0, -0.002561021505292733, 0.007256097207950777, 0.01220879099245144, 0, 0.01049822585489484, 0, 0, 0, 0,
        0.007773767969440824, 0.01017687202712697, 0, -0.001050522677134296, 0, -0.001339630732300674, 0, 0, 0, 0.00255544007620822, 0, 0.002360299926138286, 0, 0, 0, 0, 0, 0, 0, -0.001294291109292278, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007096015571718361, 0.006724747496318347, 0, 0, 0.004343175213457304, 0, 0, -0.001082430617952559, 0, 0, 0, 0, 0, 0, 0, 0, 0.006718208897307288, 0.01438555517247364, 0, 0.01410924253275575, 0, 0, 0, 0.004124413315851447, 0, 0, 0, 0, 0, 0, 0, 0,
        0.008420692572685408, 0.01298562483008977, 0, 0, 0.004771322113688647, 0.003868693213367615, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0111466509120596, 0.01403156038119017, 0, -0.001508363139560228, -0.001696002898317804, 0.01073655131052246, -0.001514370775461763, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.003473824902304878, 5.210601622719169e-05, 0, -0.001284583926189559, 0, 0, 0, 0, 0.005281278132491851, 0.008307103286448686, 0, -0.001074312388499371, 0, 0, 0, 0, 0.007091508177638381, 0.01197754434545787, 0, 0, 0, 0, 0, 0, 0, 0.007579657333834109, -0.001329583502032993, -0.0009160966882366436,
        0, 0, 0, 0.004351497686309685, 0, 0, 0, 0, 0, 0.00438853100104441, 0, 0, 0, 0.004036302054754368, 0, 0.004277058018622209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.009879270876964562, -0.002579135833216216, 0.007311883636580055, 0, 0, 0, 0.01007596182209542, 0, 0.01261851077519138, 0.006354073042640468, 0.01328546805878161, 0, 0, 0, 0, 0, 0, 0, 0.0009215788235519226, 0, 0, 0, 0, 0, 0, 0, 0.0008991588312666908,
        0.008552396202097167, 0.007225908617391515, 0, 0, 0.003763377929911007, 0.00424498008419954, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01111846743833556, 0.009238402010474323, 0, -0.004312436426987752, 0.0132548264915196, 0.01172790209162834, 0, 0.008150580925511117, 0, -0.001374450752039234, 0, 0, -0.001596691424480097, -0.0009640650892770419, 0, -0.00128186138798063, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.001165896053979433, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0.008265623714571407, 0.007819637772626634, 0.004377006635153257, 0.006536620321796502, 0.006586688695302854, 0, 0, 0, 0.01136857756636683, 0.01294382342769633, 0.007351646470676953, 0.007881118861194291, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.002428867510618063, 0, 0, 0, 0, 0.003574974404976373, 0.005880192728772203, 0, -0.001404620136728659, 0, 0.004264667140972739, 0, 0, 0.00620536963367245, 0.01261753142007484, 0.00467785372590396, -0.001117464458550222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004213142734279664, 0, 0, 0, 0, -0.001157151833770036, -0.002542532178745937,
        0, 0, 0, 0, -0.004325539817889044, 0.005153093066172859, -0.00144927555078096, 0, 0.008526352062455396, 0, 0, 0, 0.01011596404958545, 0, 0, 0, 0.01203891073069874, 0, 0.004380094124401699, 0, 0.008071425495684898, -0.002232651556861227, 0, 0, 0.01240812985849265, 0.006339130265059567, 0, 0, 0.009901970730564657, 0, 0.004164753836731723, 0, -0.002634799835571297, 0, 0, 0, -0.002211082165680737, -0.002632659008998902, 0, -0.0009686690231967094, 0.00795877185550741, 0.003876118628331966, 0, 0, 0.004266863290957323, 0, 0.005675896410056496, 0, 0.005124148858167672, 0, 0, -0.001417924562402174, -0.002043175585754407, -0.001626186329788107, 0, -0.001426689205527202, 0.01045814359121886, 0.003903087672672632, 0, 0, 0.005913547800360818, 0.006834395897221254, 0, 0,
        0.01021489573386341,
        -0.004599315000983379, 0.001555255368586686, 0.008725886285005559, 0.008701198173344976, 0.008288226360306706, 0.01123221501768839, 0.01112460089342843, 0.01203625542714812,
        0, 0, 0, 0, 0, 0, 0, 0, 0.006131461498606, 0, 0.006458765408159833, 0, 0.007948164615705619, 0.003196911959275916, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.002629395542014179, 0.005551744873795068, 0.001609726786504423, 0.003253897780381049, 0, 0, 0, 0, 0.00861709832193408, 0.01196020656836642, 0.001154295123080674, 0.0115233406115236, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001988889371782951, 0, 0.001964016619512514, 0,
        0.009438555167268112,
        0, 0, 0, 0, 0.008748577076102151, 0.00708264445594167, 0.01185044918327474, 0.007815786736757351,
        0.008698014991779876, 0.008505999743018669,
        0, 0, -0.001337916775583493, 0, 0, 0, 0, 0, 0, 0.003839061153349305, -0.004509270639005881, 0.009773330940860865, 0, 0, -0.001690082181271167, 0, 0.004541721660026839, 0, 0.007002709245096559, 0, 0, 0, 0, 0, 0.004406016229577577, 0.008522761429125657, 0.008817761645823975, 0.01082813224713589, 0, 0, 0, 0, 0, 0, -0.001614258716982043, 0, 0, 0, 0, 0, 0, 0, -0.003998493408957567, 0.008881289174174791, 0, 0, 0.0032613002610295, 0, 0, 0, -0.001067444942966455, 0.003552521251521712, 0, 0, -0.001753220220089999, 0, 0, 0.003685537801984482, -0.001949434111290908, 0.008830800891578219, 0, 0, 0, 0,
        0.01116884072582837, 0.008844160095545138, 0.007041634310081996, -0.003409736212626188
    ]
    model_ctrs = catboost_model_ctrs_container(
        used_model_ctrs_count = 10,
        compressed_model_ctrs = [
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 16890222057671696980, base_ctr_type = "Counter", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = -0, scale = 15)
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
                16890222057671696980 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 15379737126276794113 : 5, 18446744073709551615 : 0, 14256903225472974739 : 2, 18048946643763804916 : 4, 2051959227349154549 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7024059537692152076 : 6, 18446744073709551615 : 0, 15472181234288693070 : 1, 8864790892067322495 : 0},
                    target_classes_count = 0,
                    counter_denominator = 68,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 9.52883e-44, count = 7), catboost_ctr_mean_history(sum = 8.40779e-45, count = 9), catboost_ctr_mean_history(sum = 5.60519e-45, count = 2)],
                    ctr_total = [68, 7, 6, 9, 4, 2, 5]
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



