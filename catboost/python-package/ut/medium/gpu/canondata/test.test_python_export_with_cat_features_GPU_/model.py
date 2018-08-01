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
    float_feature_count = 5
    cat_feature_count = 11
    binary_feature_count = 83
    tree_count = 20
    float_feature_borders = [
        [36.5, 37.5, 60.5, 61.5, 68.5],
        [107780.5, 204331, 211235.5, 553548.5],
        [13.5],
        [2189.5],
        [38.5]
    ]
    tree_depth = [6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    tree_split_border = [4, 1, 1, 1, 1, 1, 1, 1, 255, 5, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 4, 1, 1, 1, 255, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 255, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
    tree_split_feature_index = [0, 62, 2, 67, 38, 66, 45, 32, 5, 0, 59, 15, 1, 75, 72, 49, 79, 58, 69, 1, 60, 80, 40, 0, 71, 56, 27, 9, 46, 47, 26, 1, 74, 33, 78, 5, 48, 55, 19, 51, 3, 69, 1, 10, 77, 54, 76, 7, 0, 25, 21, 20, 0, 27, 31, 43, 46, 81, 33, 6, 26, 23, 1, 13, 61, 69, 64, 29, 65, 54, 42, 47, 61, 80, 53, 29, 57, 61, 64, 81, 61, 52, 39, 46, 59, 37, 35, 44, 0, 7, 11, 68, 27, 17, 12, 5, 47, 70, 30, 18, 8, 4, 50, 82, 73, 47, 63, 74, 43, 34, 41, 14, 36, 6, 24, 25, 22, 16, 28]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    one_hot_cat_feature_index = [9]
    one_hot_hash_values = [
        [-2114564283]
    ]
    ctr_feature_borders = [
        [0.5],
        [0.02941176667809486, 0.05392156913876534],
        [0.3498816788196564],
        [0.0632147341966629],
        [0.6815651655197144],
        [0.9651535153388977],
        [0.8757874369621277],
        [0.9383436441421509],
        [1.000899910926819],
        [0.8579142093658447],
        [0.5110822319984436],
        [0.2762423753738403],
        [0.2744823694229126],
        [-0.009802941232919693],
        [0.5202279686927795],
        [0.7717800140380859],
        [0.08822647482156754],
        [0.6280156373977661],
        [0.245073527097702],
        [0.2531880140304565, 0.3600040078163147],
        [0.34375, 0.4583333432674408],
        [0.2942708432674408, 0.8697916269302368],
        [0.8500000238418579],
        [0.2352941334247589],
        [0.3899509906768799],
        [0.1897294521331787],
        [0.5431933403015137],
        [-0.009802941232919693],
        [0.03921176493167877],
        [0.1299836784601212],
        [-0.009802941232919693],
        [0.02940882369875908],
        [0.1765923351049423],
        [0.8757874369621277],
        [-0.009802941232919693],
        [-0.009802941232919693],
        [0.6634929180145264],
        [0.3233576416969299, 0.9508548378944397],
        [0.06081081181764603],
        [0.6642736196517944],
        [0.6428571343421936, 0.75, 0.7857142686843872],
        [0.3872548937797546],
        [0.632155179977417],
        [0.4279887676239014],
        [0.3627088367938995],
        [0.04901470616459846],
        [0.3336332738399506],
        [0.07842352986335754],
        [-0.009802941232919693, 0.3529058992862701],
        [0.5712000131607056],
        [0.9532380104064941],
        [0.3963853716850281],
        [0.6323625445365906],
        [0.4662867188453674, 0.5044733881950378],
        [0.7083333134651184],
        [0.1029411777853966, 0.1176470667123795, 0.1421568691730499],
        [0.1274382472038269],
        [0.4151881039142609],
        [0.2634103298187256, 0.3179171085357666],
        [-0.009802941232919693],
        [-0.009802941232919693],
        [0.08822647482156754],
        [0.8409091234207153],
        [0.8325892686843872],
        [-0.01960588246583939],
        [0.8466897010803223],
        [0.6669999957084656],
        [-0.009802941232919693],
        [0.6875],
        [0.4607843160629272],
        [0.9294070601463318],
        [-0.009802941232919693],
        [-0.009802941232919693],
        [0.9554044604301453],
        [0.05514705926179886, 0.7720588445663452],
        [0.6875],
        [1.000899910926819]
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02444444410502911, 0.008571429178118706, 0.006000000052154064, 0, 0.007499999832361937, 0, 0, 0, 0.02419354766607285, 0, 0.02272727154195309, 0, 0, 0, 0.007499999832361937, 0,
        0.007318548392504454, 0, 0.007329545449465513, 0, 0.007318548392504454, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004346903879195452, 0, 0.0154450424015522, 0.02640475332736969, 0.005818636622279882, 0, 0.01244760397821665, 0.02628972381353378, 0, 0, -6.42857194179669e-05, 0, 0, 0, 0, 0.007435713894665241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0.009303716011345387, 0.009453384205698967, 0, 0, 0.007071131840348244, 0, 0, 0, 0, 0.01430819649249315, 0, 0, 0, -0.0004683707666117698, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00721370754763484, 0, 0, 0, 0, 0, 0, 0, 0.00724571431055665, 0.005598218180239201, -6.380357808666304e-05, 0, 0.02087440155446529, 0.01998653635382652, 0.02409933879971504, 0.02516734786331654, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0002306156820850447, 0.006962073035538197, 0.005401887465268373, 0.01401940081268549, 0.006980211474001408, 0.01563218981027603, 0.003643837524577975, 0.01749342121183872, 0.00963042676448822, 0.02392844296991825, 0.01610502414405346, 0.02341570891439915,
        0, -0.0008106189779937267, 0, -0.0005681456532329321, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01619710400700569, 0.01004951354116201, 0.01403097808361053, 0, 0, 0, 0, 0, 0, 0.006985608488321304, -0.0002288860705448315, 0, 0, 0, 0, 0, 0, 0, 0.01709960587322712, 0.01331558357924223, 0, 0, 0, 0, 0, 0, 0, 0.007043925113976002, 0, 0, 0, 0, 0, 0.007139019668102264, 0.02491246350109577, 0.02221493609249592, 0, 0, 0, 0, 0, 0, 0.006761167198419571, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.006659418344497681, 0, 0, 0, 0, 0, 0, 0, 0.02066284045577049, 0.003843937534838915, 0.0248980950564146, 0.01325277704745531, 0, 0, 0.0105514470487833, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0.01197739224880934, 0, 0, 0, -0.0002274451107950881, 0, 0.02353507280349731, 0.02307623811066151, 0, 0, 0, 0, 0, 0, 0, 0, 0.01082733273506165, 0, 0, 0, -0.0002904438006225973, 0, -0.0007751971133984625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0006024401518516243, 0, 0, 0, 0, 0, 0, 0, 0, -0.0003350587212480605, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0008791213040240109, 0.01903260499238968, 0.005012993235141039, 0.01902318373322487, 0.01051970664411783, 0.02149626985192299, 0.01499412301927805, 0.02091122604906559, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01059097610414028, 0, 0, 0, 0, 0, 0, 0, 0.01017615292221308, 0, 0.006756708025932312, 0, 0, 0, 0, 0, 0.02171423844993114, 0, -0.001170339877717197, 0, 0, 0, -0.0006330512114800513, 0, 0, 0, 0, 0, 0, 0, -0.0002719693002291024, 0, 0, -0.0003684838302433491, 0.002135606948286295, 0, 0, 0, 0, 0, 0, 0, 0.006824209354817867, 0, 0, 0, 0, 0, 0, -0.000912127667106688, 0.01876332610845566,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0002699295582715422, 0.002284257905557752, -0.0004666773020289838, 0.01225898787379265, 0, 0, 0, 0, 0, 0, 0.01008061598986387, 0.02230390906333923, 0, 0, 0, 0,
        0, 0.005769095849245787, 0, 0.004319170489907265, 0, 0.01166940852999687, 0, 0.01317906472831964, -0.0002679050667211413, 0.01543235778808594, -0.0008486259030178189, 0.01636309362947941, 0, 0.009370860643684864, 0, 0.01958545669913292, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00654830876737833, 0, -0.0009498699801042676, 0, 0, 0, 0.005729850847274065, -0.0003828521294053644, 0.00334418355487287, -0.001072956249117851, 0.0162800382822752, 0, -0.0004631772171705961, 0, 0.01747974194586277, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004703616257756948, 0.01967307180166245, 0, 0, 0, 0.01335357129573822, 0, 0, 0, 0, 0, 0, 0, 0, 0.0002697087766136974, 0, 0.01016911771148443, 0, 0.007780205924063921, 0, 0.0120719987899065, -0.001097647356800735, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01974833756685257, 0, 0, 0, 0.01735987327992916, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.006285391747951508, 0, 0, 0, 0, 0,
        0, 0.006134070921689272, 0, 0.004509619902819395, 0, -0.001018333015963435, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.009973449632525444, 0, 0, 0, 0, 0, 0, 0, -0.001135406782850623, 0, 0, 0, 0, 0, 0, 0.01905504614114761, 0.007570439018309116, 0.01594502665102482, 0.005366995465010405, 0.01742936670780182, 0.003876778995618224, 0.0151000814512372, 0.009166469797492027, 0, 0.004620596766471863, 0, 0, 0, -0.001660955138504505, 0, 0, 0.01387618482112885, 0.01246349234133959, 0.005615232978016138, 0, 0.01652183756232262, 0.004095642827451229, 0.008783933706581593, 0.004166483879089355, 0, 0, 0, 0, 0, -0.00103561207652092, 0, 0,
        0, 0, 0, 0, 0.006139869801700115, 0, 0.01696298830211163, 0, 0, 0, 0, 0, 0.001038587070070207, 0.001721293316222727, 0.01736384443938732, 0.01587518118321896, 0, 0, 0, 0, 0, 0, 0.01123205292969942, 0, 0, 0, 0, 0, 0, 0, 0.01202062051743269, 0.01534510869532824, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.001007230719551444, 0.01086215488612652, 0.003880853764712811, 0, 0, 0.003281862009316683, 0, 0, 0, 0, -0.001413052785210311, 0, 0, 0, 0.008345676586031914, 0, 0.007983323186635971, 0.01751497574150562, 0.01406861282885075, 0, 0.002870347816497087, 0.01970046199858189, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005381905939429998, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.008920123800635338, -0.001253582187928259, 0, -0.0007160686072893441, 0,
        -0.0004507967969402671, 0, 0, 0, 0, 0, 0, 0, -0.001565663842484355, 0.001724834088236094, 0, 0, -0.001108945580199361, 0, 0, 0, 0, 0.004400026984512806, 0, 0, 0, 0.005357671994715929, 0, 0, 0.001764490851201117, 0.01418506447225809, 0, 0, 0.01052091550081968, 0.01931552402675152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, -0.0004474158049561083, 0, 0, 0, 0, 0, 0, 0, -0.0008075154037214816, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01224045269191265, 0, -0.0009867530316114426, 0, 0, 0, 0, 0, 0.01809972524642944, 0.01828772760927677, 0.001686385367065668, -0.001530608278699219, 0, 0, 0, 0, 0, 0, -0.001246382016688585, 0, 0, 0, 0, 0, 0, 0, 0.01226859632879496, 0.01150708831846714,
        0, 0, 0, 0, 0, 0, 0, 0, 0.01230184640735388, 0.01603750139474869, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007208502385765314, 0, 0, 0.007532537914812565, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01609379425644875, 0.01785399578511715, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.009060865268111229, 0, 0, 0.006220576353371143, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0006497348076663911, 0.004957037046551704, 0.01695586740970612, 0, 0.002619342179968953, 0, 0.01257459726184607, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0.005595019552856684, 0, 0, 0, 0, 0, 0, 0, 0.00558592239394784, -0.001392377889715135, 0, 0, 0, -0.001382482354529202, 0.01369914412498474, 0, 0.006395020522177219, 0, 0, 0, 0, 0, 0.01512627769261599, 0, 0.01585051417350769, 0, 0, 0, 0, 0, 0, 0, -0.0010574737098068, 0, 0, 0, 0, 0, 0, 0, -0.001453210250474513, -0.0009851220529526472, 0, 0, 0, 0, 0.01449866592884064, 0, -0.0005889760213904083, 0, 0, 0, 0, 0, 0.01420286111533642, 0, 0.009212911128997803
    ]
    model_ctrs = catboost_model_ctrs_container(
        used_model_ctrs_count = 77,
        compressed_model_ctrs = [
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791580653471478, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 8405694746487331134, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 5, check_value_equal = 1, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 8466246647289997739, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 5445777084271881951, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 1, check_value_equal = 0, value = 3)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 18024574529690920871, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 4],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493224740170, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 4, 5],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791582260355193, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 768791582260355193, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 4, 5, 8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952491745844305, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 4, 5, 10],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952491745844307, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
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
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493224740165, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 10041049327393282701, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 5, 7],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 5, check_value_equal = 1, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 824839814717248663, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 5, 8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791582259504189, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 6],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493224740164, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 10041049327393282700, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [3, 7],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493224740167, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 10041049327393282703, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
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
                        catboost_bin_feature_index_value(bin_index = 5, check_value_equal = 1, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 8466246647289997740, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 5],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493261641996, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 17677952493261641996, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 10041049327410906820, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 5, 8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 10041049327171048580, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 5, 8, 10],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952491432648590, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 10041049326689763398, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 5, 10],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 10041049327171048582, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 6],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 4)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 15110161923527827288, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 6, 7],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493039013652, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 10041049327172161756, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 6, 8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 10041049327172161759, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 7],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493261641998, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [4, 8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493261641993, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
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
                        catboost_bin_feature_index_value(bin_index = 5, check_value_equal = 1, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 8466246647289997741, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
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
                    catboost_model_ctr(base_hash = 9867321491374199500, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 4, check_value_equal = 0, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 1557060623092924881, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 6, 7],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 5, check_value_equal = 1, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 824839814747742770, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 6, 7, 10],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493294194665, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 6, 8, 10],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 10041049327447366776, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 7],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 10041049327412019993, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 7],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 5, check_value_equal = 1, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 9980497426473087884, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 7],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 13000966989535245560, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493260528848, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 17677952493260528848, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [5, 10],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493260528850, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
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
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 4)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 5783086744600289132, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [6],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 4, check_value_equal = 0, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 7752814180203503130, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [6, 7],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17677952493263343768, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 10041049327413399376, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [6, 7],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 4)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 12663657329316825351, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [6, 10],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 4)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 12663657329316825354, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
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
                        catboost_bin_feature_index_value(bin_index = 5, check_value_equal = 1, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 824839813111232103, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [7],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 0, check_value_equal = 0, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 5445777084271881947, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [7],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 1, check_value_equal = 0, value = 2)
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
                        catboost_bin_feature_index_value(bin_index = 4, check_value_equal = 0, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 1557060623092924883, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [8],
                    binarized_indexes = []
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 768791580653471469, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 8405694746487331109, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [8],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 1, check_value_equal = 0, value = 3)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 18024574529690920892, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = 0, scale = 1),
                    catboost_model_ctr(base_hash = 9522977968701323380, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [8],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 1, check_value_equal = 0, value = 4)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 17781294116708535183, base_ctr_type = "FeatureFreq", target_border_idx = 0, prior_num = 0, prior_denom = 1, shift = 0, scale = 1)
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
                    catboost_model_ctr(base_hash = 768791580653471471, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            ),
            catboost_compressed_model_ctr(
                projection = catboost_projection(
                    transposed_cat_feature_indexes = [10],
                    binarized_indexes = [
                        catboost_bin_feature_index_value(bin_index = 4, check_value_equal = 0, value = 1)
                    ]
                ),
                model_ctrs = [
                    catboost_model_ctr(base_hash = 7752814180203503110, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 1, prior_denom = 1, shift = 0, scale = 1)
                ]
            )
        ],
        ctr_data = catboost_ctr_data(
            learn_ctrs = {
                17677952491745844305 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 944558570445106821 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7371468505993421835 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 2703053467948138254 : 21, 10624812830858719759 : 41, 18446744073709551615 : 0, 18446744073709551615 : 0, 7999694595073717394 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16329932676893214616 : 1, 3613921271646451993 : 27, 4485676444185027480 : 16, 5778698138021950875 : 46, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6057558516954050849 : 28, 18446744073709551615 : 0, 10227770426212885667 : 29, 1958396399320360356 : 57, 16557612739106137509 : 43, 18446744073709551615 : 0, 18446744073709551615 : 0, 5958576118581057832 : 51, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3573106943124584364 : 39, 6082460677131087661 : 44, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14436279871875090225 : 47, 15502868568164779441 : 8, 18446744073709551615 : 0, 13933539874028250804 : 33, 18446744073709551615 : 0, 4550841478791429686 : 42, 18446744073709551615 : 0, 4248102824965761720 : 17, 4731159317230835257 : 7, 544847333557355193 : 9, 2246976902800319163 : 11, 8225897030342439996 : 55, 6525538820087947960 : 58, 9999297712774288062 : 23, 18446744073709551615 : 0, 18446744073709551615 : 0, 4993919621748289985 : 35, 16709688263950738114 : 52, 18446744073709551615 : 0, 18446744073709551615 : 0, 11362490414905272005 : 32, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17854162903566919242 : 19, 10570551877277045835 : 34, 17911332594876536140 : 25, 12364931380134158538 : 0, 1581563305888667470 : 20, 12920031267399879115 : 49, 9830871216565614030 : 15, 16016048497777336270 : 54, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16379713396536878165 : 12, 12535441751037464533 : 40, 18081842965779842135 : 3, 17964140040136016983 : 22, 3691734829891921751 : 38, 11445581760369838038 : 56, 1809243368101590363 : 18, 13147711329612802008 : 31, 18446744073709551615 : 0, 10240577009863185886 : 36, 18446744073709551615 : 0, 18446744073709551615 : 0, 10301477354371972833 : 48, 18446744073709551615 : 0, 18446744073709551615 : 0, 277624169761932388 : 10, 18446744073709551615 : 0, 7952701424106589414 : 30, 12233333307231209319 : 37, 12620312099102323560 : 59, 18446744073709551615 : 0, 8227339161997134826 : 24, 18446744073709551615 : 0, 14379611359654904428 : 4, 6492328680625889389 : 53, 12585231856797791725 : 2, 11524046188588617967 : 50, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12847992161315246453 : 5, 18446744073709551615 : 0, 14550121730558210423 : 6, 18446744073709551615 : 0, 16675132211273682809 : 45, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 3, 1, 0, 0, 1, 3, 4, 0, 6, 0, 5, 0, 4, 0, 1, 0, 1, 0, 1, 0, 4, 0, 1, 1, 7, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 1, 2, 1, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 2, 3, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1, 1, 0, 0, 1, 0, 1]
                ),
                17677952491745844307 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 12386861758883532418 : 9, 18446744073709551615 : 0, 1615194939518637828 : 27, 18446744073709551615 : 0, 6282805614514371974 : 55, 18446744073709551615 : 0, 7984935183757335944 : 45, 18446744073709551615 : 0, 8042104875066952842 : 4, 6247725372209840139 : 2, 5764668879944766602 : 16, 5186539704000666381 : 46, 18446744073709551615 : 0, 18446744073709551615 : 0, 11515863767753292944 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 6510485676727294867 : 5, 1048747059899601044 : 43, 8212615245970258837 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1329552534770423325 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12117686985652909478 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 1033962021405470249 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 14812291057069738284 : 20, 4287306346270768173 : 36, 18446744073709551615 : 0, 18446744073709551615 : 0, 248609998944785840 : 54, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9992426192305263030 : 1, 15723158860768052023 : 25, 16594914033306627510 : 15, 13749899362210214201 : 32, 17887935727143550905 : 42, 18446744073709551615 : 0, 9743905216976139708 : 23, 18446744073709551615 : 0, 18446744073709551615 : 0, 18166796106075650879 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 14067633988441960386 : 53, 10220106254518185923 : 39, 11802919703730877636 : 52, 18446744073709551615 : 0, 18067813707702657862 : 47, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18191698266252687691 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9165362083576827855 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 7596033389440299218 : 29, 18446744073709551615 : 0, 16660079067913029716 : 38, 18446744073709551615 : 0, 526803805060551766 : 13, 12654084922678955223 : 8, 18446744073709551615 : 0, 14356214491921919193 : 10, 1888390545754488410 : 51, 6849666623986928987 : 49, 18446744073709551615 : 0, 18446744073709551615 : 0, 11865561032414258910 : 41, 18446744073709551615 : 0, 10372181779362786528 : 48, 18446744073709551615 : 0, 18446744073709551615 : 0, 5024983930317320419 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11516656418978967656 : 18, 4233045392689094249 : 30, 11573826110288584554 : 24, 17084754933191592809 : 34, 13690800895010267500 : 19, 6027424895546206952 : 0, 3493364731977662444 : 14, 9678542013189384684 : 50, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10042206911948926579 : 11, 18446744073709551615 : 0, 11744336481191890549 : 3, 15800972419013521781 : 33, 18446744073709551615 : 0, 18446744073709551615 : 0, 13918480957223190393 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 950539760915361660 : 37, 3903070525275234300 : 31, 18446744073709551615 : 0, 3963970869784021247 : 44},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 3, 1, 0, 0, 2, 3, 6, 0, 7, 0, 7, 0, 5, 0, 1, 0, 1, 0, 4, 0, 1, 1, 6, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3, 0, 2, 0, 1, 0, 1, 1, 1, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 3, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1, 1, 0, 0, 1, 0, 1]
                ),
                9980497426473087884 :
                catboost_ctr_value_table(
                    index_hash_viewer = {8818114060598530624 : 1, 11977580115339394176 : 17, 12461144179858147074 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 11548157811212961413 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 1644045280240179080 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1214622976113746317 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 16857617732403848144 : 2, 5290950991575773969 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6524082897304633048 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10600410912766660700 : 20, 18446744073709551615 : 0, 16279254845622426718 : 0, 18446744073709551615 : 0, 12614316545066493728 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14875900814865445861 : 19, 5945720010523211622 : 15, 7246483080929928871 : 8, 18446744073709551615 : 0, 3141970693103761833 : 11, 10022048025985239274 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15359692319540265391 : 7, 18446744073709551615 : 0, 11255179931714098353 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4654042008258386997 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4347934941247810554 : 16, 271853569121629179 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 1160253775565359550 : 5, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 2.66247e-44, count = 17), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 11), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 10), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 4, 19, 17, 0, 5, 0, 11, 0, 1, 0, 1, 0, 10, 0, 1, 0, 1, 1, 0, 2, 3, 0, 1, 0, 4, 0, 1, 0, 8, 0, 3, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                ),
                1557060623092924881 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 3581428127016485793 : 10, 18446744073709551615 : 0, 17856817611009672707 : 9, 18446744073709551615 : 0, 14455983217430950149 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13125720576600207402 : 4, 5967870314491345259 : 1, 2436149079269713547 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1236773280081879954 : 5, 16151796118569799858 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18336378346035991543 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 8312525161425951098 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13605281311626526238 : 8, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.94273e-44, count = 37), catboost_ctr_mean_history(sum = 1.54143e-44, count = 16), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 5.60519e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [21, 37, 11, 16, 1, 5, 4, 2, 1, 1, 2]
                ),
                1557060623092924883 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 10934650013725255009 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 5844492600899280932 : 0, 18446744073709551615 : 0, 1034166431492604838 : 7, 18446744073709551615 : 0, 6203552979315789704 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 1113395566489815627 : 2, 13957701839509617452 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18034029854971645104 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 9226604805100152147 : 6, 1601191413561926516 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 13302932820562179799 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5765263465902070143 : 1},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.66247e-44, count = 32), catboost_ctr_mean_history(sum = 4.2039e-45, count = 14), catboost_ctr_mean_history(sum = 1.12104e-44, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 4), catboost_ctr_mean_history(sum = 1.54143e-44, count = 4)],
                    ctr_total = [19, 32, 3, 14, 8, 2, 3, 4, 11, 4, 1]
                ),
                17677952493224740164 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 15869504672553169153 : 14, 3630895197587547650 : 13, 18446744073709551615 : 0, 18069894976246263428 : 12, 6657459529952533892 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12403109157674544908 : 21, 7581495141437254476 : 11, 18446744073709551615 : 0, 544300312823816335 : 26, 8994715059648341648 : 25, 18446744073709551615 : 0, 7582268711114204562 : 7, 9997066275032783314 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 8155965639411439190 : 5, 18446744073709551615 : 0, 17626120688916674776 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5135391889252992221 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11422884619242973793 : 9, 3129976559056829986 : 20, 10518099770818402979 : 10, 11182690403015408099 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 2283527241891053351 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10921182301457540139 : 3, 4851313952246684459 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7647347103951847349 : 0, 5184516154834744246 : 27, 18446744073709551615 : 0, 1764719067482953144 : 23, 6066581188437978489 : 16, 8257839345965546298 : 17, 12150488944147554235 : 24, 16694931389731688508 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 9376384394070575999 : 18},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 5.60519e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 1.4013e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 9), catboost_ctr_mean_history(sum = 1.4013e-45, count = 9), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 0, count = 14), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 4.2039e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [1, 5, 1, 2, 4, 3, 0, 6, 1, 7, 0, 9, 1, 9, 1, 5, 0, 14, 0, 2, 2, 0, 1, 0, 1, 0, 3, 4, 0, 1, 1, 4, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1]
                ),
                17677952493224740165 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 5195954639254248834 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 12596183338487933509 : 3, 11415090325326527685 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 15694110684349775305 : 6, 3105842455684076810 : 15, 18446744073709551615 : 0, 11619308647181131660 : 11, 18446744073709551615 : 0, 7384862814707324430 : 16, 16546783282337640335 : 10, 13877983093189917584 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 9803181056021273939 : 12, 17960003200548727507 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 15929159679822070487 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1885423701024940001 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6164444060048187621 : 4, 1036643838009237222 : 9, 18446744073709551615 : 0, 2089642022879543976 : 2, 3679105889079969577 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1862978297920111856 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 11528263922108981619 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7453461884940337974 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16229983591748392701 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 1.4013e-44, count = 18), catboost_ctr_mean_history(sum = 0, count = 26), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2)],
                    ctr_total = [0, 4, 1, 2, 0, 8, 10, 18, 0, 26, 0, 2, 0, 2, 0, 1, 0, 3, 5, 0, 2, 0, 0, 2, 0, 1, 0, 1, 0, 2, 0, 1, 0, 3, 1, 0, 3, 0, 0, 1, 0, 2]
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
                17677952493224740170 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4074149013537909256 : 21, 18446744073709551615 : 0, 12733361023712308234 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10360532617354119439 : 24, 18446744073709551615 : 0, 12179894358259103633 : 13, 18446744073709551615 : 0, 3294711086045205011 : 22, 12096630803572290451 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3646433500654197144 : 4, 5941490484899010585 : 28, 18446744073709551615 : 0, 9780057282422735643 : 3, 5597533724707970587 : 23, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9010253362714991142 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15143877144483529641 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17675697484024171309 : 7, 7968584429078161966 : 17, 2650227733957515949 : 6, 17247737190584974768 : 33, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10533638073885052473 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 13332404291138955964 : 30, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6940720914563173696 : 20, 4866852255313333953 : 29, 18446744073709551615 : 0, 11359972533795927107 : 15, 13655029518040740548 : 1, 2660929095326822084 : 12, 17493596315564465606 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 8146030859722583625 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11603958955687655886 : 18, 3905715562244114895 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13509850721783377623 : 31, 18446744073709551615 : 0, 15682123462219891929 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15033235432953438954 : 11, 2074292331386068202 : 32, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9366449614381720434 : 27, 18446744073709551615 : 0, 15859569892864313588 : 34, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17157073225186666874 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 862979833014771711 : 8},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 4.2039e-45, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 12), catboost_ctr_mean_history(sum = 2.8026e-45, count = 10), catboost_ctr_mean_history(sum = 1.4013e-45, count = 20), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 3, 1, 0, 3, 2, 3, 12, 2, 10, 1, 20, 0, 4, 0, 1, 1, 1, 0, 1, 0, 3, 2, 0, 1, 0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 2, 0, 2, 1, 0, 0, 1, 2, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
                ),
                824839814717248663 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 3025022005729501057 : 33, 18446744073709551615 : 0, 8835904469718210051 : 6, 18446744073709551615 : 0, 11353600181979058821 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12316901326360662794 : 29, 3468071005581645194 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 16393229341822690446 : 30, 18446744073709551615 : 0, 18446744073709551615 : 0, 12387235196588615953 : 8, 11064525677534818577 : 39, 18446744073709551615 : 0, 7476767856365814676 : 16, 18446744073709551615 : 0, 4812021923037324054 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16937603878988534428 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 10486162790753397151 : 0, 18446744073709551615 : 0, 6713122111851187233 : 22, 15828585773926627489 : 7, 5039701985250246947 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 12851525594842641574 : 18, 385581421986046630 : 27, 18446744073709551615 : 0, 17919339127050412585 : 31, 18446744073709551615 : 0, 4228955971116209707 : 37, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13079205657055564467 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7656138162179150648 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 2745670821956349371 : 17, 12925505587962151228 : 42, 18446744073709551615 : 0, 18446744073709551615 : 0, 6821998837418377023 : 38, 18446744073709551615 : 0, 12500842770274143041 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11097488739517162184 : 36, 2167307935174927945 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6243635950636955597 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 5507157863707138512 : 15, 18446744073709551615 : 0, 13567473440941496530 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8391343387511632983 : 14, 13620367102317475032 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 7542065673651754843 : 35, 18446744073709551615 : 0, 17731545493281640157 : 28, 14940185567482897118 : 21, 569522865899526877 : 34, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8199168039991110499 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11183301231201907303 : 32, 7769745735864677736 : 19, 1284858854014567399 : 43, 18446744073709551615 : 0, 16312377278601447019 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15882954974475014256 : 41, 18446744073709551615 : 0, 5462087876845068786 : 26, 18446744073709551615 : 0, 1512538916227490292 : 23, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 4.2039e-45, count = 3), catboost_ctr_mean_history(sum = 4.2039e-45, count = 25), catboost_ctr_mean_history(sum = 1.4013e-45, count = 8), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 5.60519e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 8), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [3, 3, 3, 25, 1, 8, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 2, 1, 1, 6, 2, 3, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                ),
                7752814180203503110 :
                catboost_ctr_value_table(
                    index_hash_viewer = {15788301966321637888 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16571503766256089902 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 7852726667766477841 : 8, 18446744073709551615 : 0, 5942209973749353715 : 3, 18446744073709551615 : 0, 11840406731846624597 : 7, 18446744073709551615 : 0, 6339188657417862231 : 5, 13477523873801719416 : 2, 16503206593760246744 : 4, 2072654927021551577 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.66247e-44, count = 51), catboost_ctr_mean_history(sum = 1.4013e-45, count = 22), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [19, 51, 1, 22, 0, 2, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
                ),
                768791582260355193 :
                catboost_ctr_value_table(
                    index_hash_viewer = {11293780942970616060 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6729923559432788613 : 17, 18446744073709551615 : 0, 16734750840518899207 : 38, 18446744073709551615 : 0, 1692798034794836105 : 44, 18446744073709551615 : 0, 3846243005460524939 : 8, 16856447464380907276 : 23, 18446744073709551615 : 0, 6456762343887838222 : 51, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16060628675036314771 : 42, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1892904960350010775 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 327522037079443354 : 50, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17236464875041178146 : 24, 18311091721957180195 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15071695099758506151 : 28, 18446744073709551615 : 0, 7036983067300301609 : 36, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8484826171500827823 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2122697142665763508 : 27, 6115109679680906933 : 30, 14507296607631921077 : 34, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 793733305022653884 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10804725596601432258 : 4, 6124088198198845634 : 16, 3384441554721946052 : 37, 16394646013642328389 : 32, 12335826644891077 : 40, 18446744073709551615 : 0, 2622855165072204360 : 18, 8435009104781511368 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 544623413399665740 : 31, 18446744073709551615 : 0, 4170750153088331470 : 45, 18446744073709551615 : 0, 15657694241928118096 : 22, 16505741855243928529 : 3, 16621650995289296466 : 49, 16826597711826545361 : 33, 17060665786235511251 : 26, 18446744073709551615 : 0, 13287977750108197206 : 41, 18446744073709551615 : 0, 18446744073709551615 : 0, 5107783893090530905 : 12, 16264846996890918746 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13895130505070997856 : 46, 18446744073709551615 : 0, 543926509552703458 : 47, 18446744073709551615 : 0, 18446744073709551615 : 0, 12027340621914696805 : 10, 18446744073709551615 : 0, 17089913298521652583 : 13, 14637859960342010088 : 43, 18446744073709551615 : 0, 7075089848460761962 : 29, 14677291468696721003 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12296262190973367535 : 39, 18446744073709551615 : 0, 10074002576804182641 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7325197678747769717 : 21, 18446744073709551615 : 0, 7704286084984261751 : 35, 8324644238789756663 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6159874463507084668 : 14, 7167124983023835004 : 48, 2851864150347175038 : 1, 17703706315619762302 : 15},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 3, 1, 0, 0, 2, 3, 6, 0, 7, 0, 8, 0, 5, 0, 1, 0, 1, 0, 4, 0, 1, 1, 7, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3, 0, 2, 0, 1, 0, 1, 2, 1, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 3, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1]
                ),
                17677952491432648590 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14414048846158871559 : 12, 18446744073709551615 : 0, 14638989218618174473 : 41, 1649460783742239370 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 11769730408401037837 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5779175314227622928 : 38, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1323165403597995158 : 26, 18428889556763151895 : 29, 2164667594568318358 : 27, 18446744073709551615 : 0, 17742152858060733338 : 4, 18446744073709551615 : 0, 8312851984779745692 : 31, 4291722588563259037 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11692867453113909922 : 22, 17412370521597043875 : 36, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13700470340432613801 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 11120589457824038700 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 7468055678118005295 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2318342633131784245 : 13, 5810832953686117174 : 33, 5297741329439372599 : 40, 3732592916305803191 : 37, 17895827896624199225 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 7723671602601903292 : 34, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12727582032159672000 : 23, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9858323221942535364 : 3, 5542904409860031684 : 16, 7800901740475460422 : 7, 18446744073709551615 : 0, 2673645599642895048 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3479611531780631629 : 28, 9935553359816092494 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5318359880926327638 : 32, 18446744073709551615 : 0, 17079190329591924568 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12509137837559755364 : 5, 4604868705542068453 : 8, 18446744073709551615 : 0, 15448722916603551847 : 17, 18084208487000104808 : 39, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9118915792783817837 : 19, 6796838119474563310 : 6, 15768090511475095279 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 3927579309257426674 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3949750004310756343 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 15091338242443513338 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16942373517120146559 : 30},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 10), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 5.60519e-45, count = 5), catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 5.60519e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 9.80909e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 10, 2, 0, 0, 2, 4, 5, 0, 8, 0, 5, 0, 1, 0, 1, 0, 5, 4, 6, 0, 1, 0, 1, 7, 0, 0, 1, 0, 2, 0, 3, 0, 1, 0, 1, 0, 2, 2, 1, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
                ),
                9522977968701323380 :
                catboost_ctr_value_table(
                    index_hash_viewer = {17151879688829397503 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 14282620878612260867 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 12420782654419932198 : 2, 18446744073709551615 : 0, 15473953381485119304 : 6, 9743509310306231593 : 3, 9551523844202795562 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10742856347075653999 : 5},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 6.86636e-44, count = 33), catboost_ctr_mean_history(sum = 9.80909e-45, count = 1), catboost_ctr_mean_history(sum = 7.00649e-45, count = 3)],
                    ctr_total = [49, 33, 7, 1, 5, 3, 3]
                ),
                12663657329316825351 :
                catboost_ctr_value_table(
                    index_hash_viewer = {2175514086045614720 : 10, 4296471548430045057 : 39, 18446744073709551615 : 0, 921104906633544451 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10288723324655951240 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14365051340117978892 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4414214375755796752 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8548807434165013530 : 37, 4032320746383113498 : 41, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17573829074316411807 : 33, 11247976407609143199 : 36, 18446744073709551615 : 0, 12145529984993450018 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16221858000455477670 : 35, 10318966679015796135 : 27, 18446744073709551615 : 0, 16618658391014953385 : 19, 18446744073709551615 : 0, 11830179625682986411 : 23, 18446744073709551615 : 0, 18446744073709551615 : 0, 17691571901642163502 : 26, 11095449725322414383 : 32, 13047909940039982766 : 45, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12268469139096705972 : 44, 18446744073709551615 : 0, 9824044767001248822 : 2, 3682349249048180150 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 7758677264510207802 : 46, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17653058209313622596 : 29, 18446744073709551615 : 0, 18446744073709551615 : 0, 2935010538933377479 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10398195814013006924 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11909408760680197200 : 21, 11395851389676435152 : 5, 7099082591273521106 : 1, 17770801036639374291 : 3, 64660978913791828 : 42, 18446744073709551615 : 0, 6364352690912949078 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10440680706374976730 : 4, 2096284951048403931 : 11, 9903273901998459611 : 13, 7437266201540159195 : 34, 18446744073709551615 : 0, 11513594217002186847 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18016483140608796131 : 43, 18446744073709551615 : 0, 18446744073709551615 : 0, 3896939534752759014 : 17, 3646067082361272167 : 24, 18233366617961394920 : 38, 18446744073709551615 : 0, 18235763632755310698 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12010148773363095534 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16086476788825123186 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 3953091611385902709 : 14, 18446744073709551615 : 0, 7178311726270731895 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11168747272611932410 : 30, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15291520964881068415 : 31},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 7.00649e-45, count = 4), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 8.40779e-45, count = 4), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 2), catboost_ctr_mean_history(sum = 5.60519e-45, count = 9), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 9.80909e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1)],
                    ctr_total = [5, 4, 2, 2, 6, 4, 3, 1, 3, 2, 4, 9, 1, 1, 7, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 5, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1]
                ),
                10041049327413399376 :
                catboost_ctr_value_table(
                    index_hash_viewer = {4042082061862408320 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7884511582485373322 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7030775828033746576 : 5, 18446744073709551615 : 0, 10946924698793439250 : 38, 8168424424900142355 : 29, 18446744073709551615 : 0, 18446744073709551615 : 0, 4630745903071679510 : 42, 4269889639072053398 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8687973919261798939 : 25, 13051844150027759644 : 23, 18446744073709551615 : 0, 10109960028276952990 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9004929399508941090 : 15, 18446744073709551615 : 0, 8177415751357997988 : 41, 18446744073709551615 : 0, 18446744073709551615 : 0, 1797935499069926311 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13808337340964467371 : 0, 18446744073709551615 : 0, 4741992351141480365 : 34, 1509942713249238958 : 32, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5973100642165243826 : 43, 3888256596689853619 : 30, 14375168484547614899 : 33, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11866342041679969211 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9909324918683866687 : 11, 7437848415134251327 : 13, 11209477939895821631 : 31, 18446744073709551615 : 0, 18446744073709551615 : 0, 15627562220085567172 : 39, 18446744073709551615 : 0, 18446744073709551615 : 0, 4371355355420267079 : 37, 15354360356848803016 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11218469266353677264 : 16, 16814167555614897617 : 14, 6312890958797077714 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11681580651965248598 : 26, 16849390855960146647 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16447610503299602651 : 22, 2820731062440340572 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16848913219507634920 : 20, 18446744073709551615 : 0, 1228836124076374122 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 10836836223971477485 : 36, 17321493568029573614 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1795667267659521650 : 28, 7412408870415946355 : 6, 16467757813577946868 : 27, 3170371727453184757 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5307011782729114234 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9770169711645119102 : 12, 13305091271955709694 : 2},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 8.40779e-45, count = 5), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 8.40779e-45, count = 4), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 5.60519e-45, count = 2), catboost_ctr_mean_history(sum = 5.60519e-45, count = 9), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 9.80909e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 7.00649e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [6, 5, 2, 2, 6, 4, 3, 1, 4, 2, 4, 9, 1, 1, 7, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 5, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1]
                ),
                12663657329316825354 :
                catboost_ctr_value_table(
                    index_hash_viewer = {360547735767635392 : 18, 6626358140135277569 : 11, 18446744073709551615 : 0, 12926049852134434819 : 3, 18446744073709551615 : 0, 8137571086802467845 : 10, 7276029099120804678 : 15, 18446744073709551615 : 0, 13998963362761644936 : 2, 18446744073709551615 : 0, 5945645236880733194 : 21, 14811342999173170379 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 9933325300960879950 : 20, 260483072505384143 : 9, 6131436228120730256 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 3459946326760557395 : 16, 7476138733731413844 : 12, 18446744073709551615 : 0, 2678462905660264278 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 125101860875029659 : 5, 17657146886543900124 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10244046410269665891 : 7, 9276072266162253411 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3406474052393002540 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17536305174900898095 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4856198895284494964 : 17, 16850420485877436981 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13881220535435893241 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.68156e-44, count = 5), catboost_ctr_mean_history(sum = 1.4013e-44, count = 10), catboost_ctr_mean_history(sum = 1.26117e-44, count = 5), catboost_ctr_mean_history(sum = 2.24208e-44, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 13), catboost_ctr_mean_history(sum = 2.8026e-45, count = 4), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [12, 5, 10, 10, 9, 5, 16, 2, 1, 13, 2, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                ),
                17677952493261641993 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 11545140384581344194 : 21, 12393187997897154627 : 16, 18247833082231120644 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 16826482028724893063 : 19, 5614500782388391176 : 4, 4113075472728399048 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4456981526622469517 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16207172316530986324 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14024607384117533720 : 1, 15049653060765064665 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16281224030707851869 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13795598398842563042 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18228892429670301478 : 23, 11046793500786150118 : 8, 18446744073709551615 : 0, 4212090381442982761 : 22, 8427151491140895401 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 4515329127061034285 : 10, 5941525859275606574 : 15, 5093478245959796141 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 3192720961219193650 : 11, 14526321418639812659 : 0, 13280186361242095092 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11967148012295355516 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 6672248879072856191 : 20},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 4.2039e-45, count = 14), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 9.80909e-45, count = 2), catboost_ctr_mean_history(sum = 5.60519e-45, count = 12), catboost_ctr_mean_history(sum = 5.60519e-45, count = 21), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [3, 14, 2, 0, 7, 2, 4, 12, 4, 21, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 1, 1, 0, 1, 1, 0]
                ),
                17677952493261641996 :
                catboost_ctr_value_table(
                    index_hash_viewer = {16259707375369223360 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13847085545544291780 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7654620248348398600 : 2, 18446744073709551615 : 0, 9243796653651753418 : 5, 18446744073709551615 : 0, 1681026541770505292 : 22, 1292491219513334285 : 21, 13677090684479491854 : 23, 6494991755595340494 : 15, 7494438315637327440 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18410271455579776277 : 14, 6336919059871405781 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9974519673449003035 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5899717636280359390 : 13, 18446744073709551615 : 0, 15904544917366469984 : 1, 18446744073709551615 : 0, 862592111642406882 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 18161161563956788133 : 11, 18446744073709551615 : 0, 3340544229935902247 : 12, 18446744073709551615 : 0, 14827488318775688873 : 16, 15675535932091499306 : 3, 18446744073709551615 : 0, 15230422751883885548 : 24, 18446744073709551615 : 0, 1662085889209686126 : 27, 18446744073709551615 : 0, 1062699037197581552 : 4, 14072903496117963889 : 17, 18446744073709551615 : 0, 15434641073738489523 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14277121817972567864 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 18160464660109825851 : 9, 16406258951888748923 : 18, 17480885798804750972 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 5.60519e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 11), catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 5.60519e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 9.80909e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [0, 12, 2, 0, 0, 3, 4, 7, 0, 11, 0, 8, 0, 1, 0, 5, 4, 7, 0, 2, 0, 1, 7, 0, 0, 2, 0, 2, 0, 3, 0, 2, 1, 1, 0, 2, 3, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
                ),
                7752814180203503130 :
                catboost_ctr_value_table(
                    index_hash_viewer = {14591795653440117248 : 9, 3812458928802352640 : 19, 16031103881690819777 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 10918373804899154693 : 4, 2002444088171013702 : 13, 10200488935662485831 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16939442126893167761 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12990050366695140501 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16503206593760246744 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13556881510278288029 : 16, 15649470839308619998 : 15, 18446744073709551615 : 0, 14931585970071951136 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 5397540690114647075 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11300006847281354472 : 2, 6619561440864924457 : 1, 3223795087593081450 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8668830525758017779 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8258953332285675196 : 14, 10128637724524112380 : 10, 18446744073709551615 : 0, 11772109559350781439 : 6},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 7.00649e-45, count = 9), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 9), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 2.8026e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 12), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 9.80909e-45, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 5), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [5, 9, 0, 6, 0, 4, 0, 9, 0, 4, 2, 4, 0, 2, 2, 12, 0, 2, 0, 2, 7, 4, 1, 3, 0, 2, 1, 4, 1, 1, 2, 5, 0, 3, 1, 0, 0, 2, 0, 1]
                ),
                17677952493261641998 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 14339393822756684802 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9042729150784344201 : 14, 18446744073709551615 : 0, 1434197551787351435 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 15505904404980462094 : 13, 17132136727440490127 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2690081920877379861 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1532562665111458202 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14397741423195249570 : 34, 18446744073709551615 : 0, 18446744073709551615 : 0, 18052491238949695525 : 32, 18446744073709551615 : 0, 18446744073709551615 : 0, 5961989641064476328 : 15, 777303952308747305 : 27, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15890374780837199661 : 30, 16738422394153010094 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 11699844009042731185 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5970980967522331961 : 35, 1590910265550022970 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11601902557128801344 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14909972007605802568 : 23, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5867491582615089871 : 28, 2743913003832016080 : 25, 18446744073709551615 : 0, 7716892253132515538 : 26, 18446744073709551615 : 0, 8557324777698838228 : 10, 18446744073709551615 : 0, 4383219007951416278 : 11, 5231266621267226711 : 31, 10600672353715374294 : 33, 7399805521932916569 : 36, 18446744073709551615 : 0, 2482461723210813787 : 18, 2164920571584601052 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7725883579590371171 : 37, 16967431379427980772 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 4392210334409271911 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 13356805169196840554 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10871179537331551727 : 29, 18446744073709551615 : 0, 3402816234720019185 : 8, 2724972351271196914 : 38, 8122374639275138803 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11414809869912342394 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 15496913078522606461 : 16, 18446744073709551615 : 0, 17469145413950259711 : 21},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 9), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 5.60519e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 4.2039e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0)],
                    ctr_total = [0, 9, 2, 0, 0, 1, 4, 7, 0, 7, 0, 6, 0, 5, 1, 1, 0, 1, 0, 6, 3, 6, 0, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 4, 0, 1, 0, 1, 0, 1, 0, 1, 3, 0, 0, 1, 0, 2, 0, 2, 0, 1, 0, 1, 5, 0, 0, 1, 0, 2, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
                ),
                17677952493263343768 :
                catboost_ctr_value_table(
                    index_hash_viewer = {4042082061862408320 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7884511582485373322 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7030775828033746576 : 5, 18446744073709551615 : 0, 10946924698793439250 : 38, 8168424424900142355 : 29, 18446744073709551615 : 0, 18446744073709551615 : 0, 4630745903071679510 : 42, 4269889639072053398 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8687973919261798939 : 25, 13051844150027759644 : 23, 18446744073709551615 : 0, 10109960028276952990 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9004929399508941090 : 15, 18446744073709551615 : 0, 8177415751357997988 : 41, 18446744073709551615 : 0, 18446744073709551615 : 0, 1797935499069926311 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13808337340964467371 : 0, 18446744073709551615 : 0, 4741992351141480365 : 34, 1509942713249238958 : 32, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5973100642165243826 : 43, 3888256596689853619 : 30, 14375168484547614899 : 33, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11866342041679969211 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9909324918683866687 : 11, 7437848415134251327 : 13, 11209477939895821631 : 31, 18446744073709551615 : 0, 18446744073709551615 : 0, 15627562220085567172 : 39, 18446744073709551615 : 0, 18446744073709551615 : 0, 4371355355420267079 : 37, 15354360356848803016 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11218469266353677264 : 16, 16814167555614897617 : 14, 6312890958797077714 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11681580651965248598 : 26, 16849390855960146647 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16447610503299602651 : 22, 2820731062440340572 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16848913219507634920 : 20, 18446744073709551615 : 0, 1228836124076374122 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 10836836223971477485 : 36, 17321493568029573614 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1795667267659521650 : 28, 7412408870415946355 : 6, 16467757813577946868 : 27, 3170371727453184757 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5307011782729114234 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9770169711645119102 : 12, 13305091271955709694 : 2},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 2.8026e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 8.40779e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 6, 5, 0, 0, 2, 0, 2, 0, 6, 0, 4, 0, 3, 0, 1, 0, 4, 0, 2, 0, 4, 2, 7, 0, 1, 0, 1, 6, 1, 1, 0, 1, 1, 0, 2, 0, 2, 0, 1, 0, 2, 0, 2, 0, 2, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 2, 0, 2, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1]
                ),
                824839814747742770 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 12453832937674008065 : 22, 2625440842438384130 : 14, 925382775098021635 : 42, 1696070020708462721 : 41, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6575042037185219979 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 12949991684148159118 : 4, 18446744073709551615 : 0, 3496003553139302288 : 57, 1543543338421733905 : 24, 18446744073709551615 : 0, 6197430359401170323 : 51, 4206754966615994516 : 1, 5619871353883761557 : 5, 2616456849048944022 : 39, 18446744073709551615 : 0, 13413086020627200024 : 56, 18446744073709551615 : 0, 6692784864510971674 : 7, 5500310991592013594 : 44, 6135240958181983772 : 48, 16653514951319078940 : 58, 17650701400100428957 : 13, 12540606719575139997 : 36, 18446744073709551615 : 0, 18446744073709551615 : 0, 14125049337031811874 : 46, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8846365495388663591 : 15, 13260559338340471208 : 47, 18446744073709551615 : 0, 15341038993303868330 : 43, 6252983785507735467 : 18, 15343436008097784108 : 17, 11265667436333908013 : 10, 846289427112326446 : 23, 10073193563414949933 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 2357502373779516722 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1060763986728376119 : 16, 10793146182180281144 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8276419647954405820 : 33, 888774319474296252 : 38, 14547039627851880894 : 49, 18446744073709551615 : 0, 5100120151951434304 : 8, 459611347081066048 : 30, 1961687830101506369 : 32, 18446744073709551615 : 0, 18446744073709551615 : 0, 8464576753708115653 : 54, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12540904769170143305 : 26, 13560961002694491977 : 50, 14931775737698275017 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14107238548848644687 : 55, 18446744073709551615 : 0, 3740983538599038929 : 34, 18446744073709551615 : 0, 18446744073709551615 : 0, 6534570401924442708 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16073149413079603033 : 45, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7324720632502234845 : 53, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14681501449658885217 : 37, 11070351772854485858 : 35, 11401048647964262497 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7426639054358269545 : 29, 2880502763403740265 : 31, 13726330766357426795 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 4813144953217298414 : 27, 18446744073709551615 : 0, 9343421339288788720 : 2, 14799244276984636912 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 594992558711420532 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6931717142343722232 : 3, 3738849606713461113 : 52, 4340623699063671545 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 4.2039e-45, count = 5), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 9), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 9.80909e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [3, 5, 2, 2, 1, 3, 3, 1, 1, 1, 1, 1, 1, 9, 1, 1, 7, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 3, 1, 5, 3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
                ),
                824839813111232103 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 10934650013725255009 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 5844492600899280932 : 0, 18446744073709551615 : 0, 1034166431492604838 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1113395566489815627 : 2, 13957701839509617452 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18034029854971645104 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 9226604805100152147 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13302932820562179799 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15316838452862012827 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.12104e-44, count = 36), catboost_ctr_mean_history(sum = 1.96182e-44, count = 11), catboost_ctr_mean_history(sum = 2.8026e-45, count = 10), catboost_ctr_mean_history(sum = 1.54143e-44, count = 4)],
                    ctr_total = [8, 36, 14, 11, 2, 10, 11, 4, 5]
                ),
                8405694746487331109 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 8473802870189803490 : 0, 7071392469244395075 : 3, 18446744073709551615 : 0, 8806438445905145973 : 2, 619730330622847022 : 1, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.14906e-43, count = 12), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6)],
                    ctr_total = [82, 12, 1, 6]
                ),
                10041049327171048580 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 4056036245110216065 : 8, 18446744073709551615 : 0, 11731113499454873091 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18158023435003188105 : 0, 16363643932146075402 : 2, 10270740755974173066 : 34, 15302458263936901644 : 32, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1882915654362385041 : 13, 16626404236663530130 : 4, 18446744073709551615 : 0, 18328533805906494100 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4722970645793390498 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11149880581341705512 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 6481465543296421931 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7392333346994735670 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12005786732634923964 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14006182501561169344 : 23, 18446744073709551615 : 0, 1889280740744869570 : 1, 5862151991393758147 : 18, 18446744073709551615 : 0, 9736988193929341509 : 33, 18446744073709551615 : 0, 18446744073709551615 : 0, 12285650654639908168 : 37, 7351519018472868041 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 834536569803511502 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10303950895436231637 : 29, 8509571392579118934 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13777709788122571739 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 17939632011589457374 : 26, 8772331697096573662 : 27, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3243000596515268201 : 19, 18446744073709551615 : 0, 1347716499416068331 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1782627853034568176 : 30, 18446744073709551615 : 0, 1711381398175610226 : 9, 15223993835718121715 : 36, 3413510967418574196 : 3, 3295808041774749044 : 16, 16926123404961085685 : 25, 18446744073709551615 : 0, 5587655443449874040 : 14, 18446744073709551615 : 0, 3692371346350674170 : 22, 14018989085211469563 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 14079889429720256510 : 31, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.4013e-44, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 9), catboost_ctr_mean_history(sum = 1.12104e-44, count = 6), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 7.00649e-45, count = 11), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 9.80909e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [10, 2, 2, 9, 8, 6, 1, 1, 5, 11, 1, 1, 7, 1, 2, 3, 1, 1, 2, 4, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]
                ),
                10041049327171048582 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2416422934830324870 : 23, 15352238185636868231 : 18, 18446744073709551615 : 0, 13456954088537668361 : 31, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13820618987297210256 : 8, 18446744073709551615 : 0, 15522748556540174226 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 7378718046372763029 : 17, 17696893032571474070 : 13, 18446744073709551615 : 0, 15801608935472274200 : 20, 7681482600623517977 : 11, 4728951836263645337 : 25, 18446744073709551615 : 0, 7742382945132304924 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 16165273834231816095 : 7, 18446744073709551615 : 0, 5393607014866921505 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11820516950415236519 : 0, 10026137447558123816 : 2, 18446744073709551615 : 0, 8964951779348950058 : 29, 18446744073709551615 : 0, 18446744073709551615 : 0, 15294275843101576621 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 10288897752075578544 : 4, 18446744073709551615 : 0, 11991027321318542514 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5494379716576807738 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4812374096753753926 : 15, 5335644672331629895 : 9, 18446744073709551615 : 0, 143959058708470345 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4027022074293069517 : 34, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6841839114768630611 : 27, 1054826862406784084 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13998518329866469600 : 1, 15581331779079161313 : 32, 18446744073709551615 : 0, 3399481709341389923 : 12, 18446744073709551615 : 0, 17910779040521939045 : 24, 5948144170051956582 : 33, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12943774158925111532 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4305215880408835443 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10628078699335212664 : 30, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11602125527001505788 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.54143e-44, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 11), catboost_ctr_mean_history(sum = 1.4013e-44, count = 7), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 1.4013e-44, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 7), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [11, 2, 3, 11, 10, 7, 1, 5, 10, 1, 1, 7, 2, 2, 3, 2, 1, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                ),
                8466246647289997739 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 12337359831519453058 : 0, 18446744073709551615 : 0, 3462861689708330564 : 10, 6193042878898900581 : 7, 9955981968190923718 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 7606262797109987753 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6973539969458659060 : 3, 13860744542689514389 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 16503206593760246744 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 2242442935049193755 : 2, 9129647508280049084 : 6, 8193958724117795869 : 9, 18446744073709551615 : 0, 11772109559350781439 : 5},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6), catboost_ctr_mean_history(sum = 1.4013e-44, count = 38), catboost_ctr_mean_history(sum = 0, count = 20), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 5.60519e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1)],
                    ctr_total = [1, 5, 1, 6, 10, 38, 0, 20, 0, 4, 1, 1, 4, 0, 2, 0, 1, 0, 0, 3, 0, 1, 2, 1]
                ),
                18024574529690920871 :
                catboost_ctr_value_table(
                    index_hash_viewer = {11772109559350781439 : 4, 18446744073709551615 : 0, 12337359831519453058 : 10, 18446744073709551615 : 0, 3462861689708330564 : 11, 6193042878898900581 : 7, 9955981968190923718 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 7606262797109987753 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6973539969458659060 : 3, 13860744542689514389 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 16503206593760246744 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 2242442935049193755 : 2, 9129647508280049084 : 8, 8193958724117795869 : 12, 18446744073709551615 : 0, 14687079002600389023 : 9},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 9.80909e-45, count = 33), catboost_ctr_mean_history(sum = 4.2039e-45, count = 25), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [1, 3, 1, 2, 7, 33, 3, 25, 1, 4, 0, 1, 2, 0, 2, 0, 3, 0, 0, 4, 2, 3, 0, 3, 0, 1]
                ),
                10041049327172161756 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 7591082115406918913 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16609914423024815883 : 79, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2947729946697366032 : 67, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1747100377421337877 : 45, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 688779167165725721 : 34, 18446744073709551615 : 0, 2390908736408689691 : 59, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12485828686297784107 : 76, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18116750275904253490 : 2, 15080985694568606770 : 26, 3441749364927451444 : 39, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8196669531629639738 : 47, 18446744073709551615 : 0, 16350523875373403452 : 65, 1910130166587793469 : 14, 3210283187799748413 : 51, 3612259735830757439 : 36, 2100546127316296510 : 72, 18446744073709551615 : 0, 16174754976619755330 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4712533024756798802 : 66, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1951646835795105624 : 33, 2675836416845132120 : 44, 13268280384053818970 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13208004644029510495 : 30, 7791717225000005216 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8958633917173958501 : 29, 18446744073709551615 : 0, 18446744073709551615 : 0, 944833316659531112 : 7, 6615501587043447913 : 71, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17286180302682250607 : 21, 18446744073709551615 : 0, 17859958192029424753 : 46, 18446744073709551615 : 0, 1115343687562837107 : 38, 18446744073709551615 : 0, 16021250581815544693 : 52, 18446744073709551615 : 0, 18446744073709551615 : 0, 15754561104342592632 : 70, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14489631383475886718 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18332060904098851720 : 11, 18446744073709551615 : 0, 1587446399632264074 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17478325149647224974 : 77, 18446744073709551615 : 0, 733710645180637328 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16419568529928495766 : 61, 18446744073709551615 : 0, 18446744073709551615 : 0, 3244631815353412249 : 15, 12476837359839928474 : 43, 1709840197601582745 : 73, 2110765276180879772 : 62, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7340761787208052128 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13489111233202375331 : 53, 18446744073709551615 : 0, 942497986273570725 : 78, 8887965463165302438 : 49, 18446744073709551615 : 0, 7907592930791199656 : 54, 13313342334448727209 : 18, 18446744073709551615 : 0, 7511272158111358123 : 50, 18446744073709551615 : 0, 16891671241997922733 : 55, 18446744073709551615 : 0, 18446744073709551615 : 0, 15003250764752625840 : 20, 18022869816319413424 : 32, 14335805918303332017 : 68, 16037935487546295987 : 63, 6375973732451541681 : 75, 16420649963778722224 : 80, 1546465267828251574 : 74, 18446744073709551615 : 0, 11854284788259111864 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10381815302203731404 : 4, 3318721001109633485 : 35, 3219274514257604046 : 19, 8814972803518824399 : 17, 16760440280410556112 : 5, 10517102372761788369 : 27, 15825775943968466 : 9, 12408874081425905107 : 16, 9980034949543187408 : 31, 8850196103864073429 : 57, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10150545320446493403 : 58, 10024723128977295580 : 41, 18446744073709551615 : 0, 18446744073709551615 : 0, 9102024323793860319 : 69, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8828822460557096163 : 56, 18446744073709551615 : 0, 18446744073709551615 : 0, 8849718467411561702 : 23, 18446744073709551615 : 0, 10551848036654525672 : 60, 10000182259821531625 : 37, 18446744073709551615 : 0, 4692931370061970411 : 28, 2837641471875404267 : 64, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5537249059016160241 : 48, 18446744073709551615 : 0, 3123341690686710771 : 24, 18446744073709551615 : 0, 15320050618309627125 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5305896519859636476 : 42, 18446744073709551615 : 0, 7008026089102600446 : 3, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 4.2039e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 5.60519e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [3, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                ),
                8466246647289997740 :
                catboost_ctr_value_table(
                    index_hash_viewer = {5321795528652759552 : 3, 1871794946608052991 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 2572990630596346628 : 9, 9755089559480497988 : 18, 714275690842131332 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14488270330580782411 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6052518548450009169 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 12538518194927513684 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9204844949746414424 : 13, 10052892563062224857 : 5, 3493345142105552026 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 7304087665005811933 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9774030212041317154 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15587441985908139302 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10783615582859474474 : 0, 14429922730142217643 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8224442176515017331 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 5550804513927730230 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7807421160518048379 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4473747915336949119 : 8},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 10), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 8.40779e-45, count = 2), catboost_ctr_mean_history(sum = 5.60519e-45, count = 12), catboost_ctr_mean_history(sum = 1.4013e-45, count = 9), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 15), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 10, 2, 0, 6, 2, 4, 12, 1, 9, 0, 4, 0, 2, 3, 15, 1, 1, 0, 2, 3, 6, 0, 3, 0, 1, 0, 2, 0, 3, 0, 3, 0, 1, 1, 1, 1, 1, 0, 1]
                ),
                17781294116708535183 :
                catboost_ctr_value_table(
                    index_hash_viewer = {17151879688829397503 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 14282620878612260867 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 12420782654419932198 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 9743509310306231593 : 2, 9551523844202795562 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10742856347075653999 : 3},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.13505e-43, count = 11), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [81, 11, 1, 6, 1, 1]
                ),
                8466246647289997741 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 3581428127016485793 : 2, 18446744073709551615 : 0, 17856817611009672707 : 3, 18446744073709551615 : 0, 14455983217430950149 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13125720576600207402 : 10, 5967870314491345259 : 6, 9724886183021484844 : 5, 2436149079269713547 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1236773280081879954 : 1, 16151796118569799858 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18336378346035991543 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 8312525161425951098 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 18), catboost_ctr_mean_history(sum = 2.8026e-44, count = 17), catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 19), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 18, 20, 17, 0, 8, 0, 1, 0, 19, 0, 2, 2, 3, 0, 5, 0, 3, 0, 2, 0, 1]
                ),
                10041049327172161759 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5872148808208377347 : 15, 18446744073709551615 : 0, 7574278377451341317 : 23, 18446744073709551615 : 0, 11937526589665073415 : 27, 10335306737124382215 : 59, 15791244932576986633 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18179756495244236684 : 16, 17455566914194210188 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 7081899133666848528 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 10265180648719063443 : 0, 3920542425808128276 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10435691019622369438 : 43, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4028980239118013348 : 25, 2594425063705713189 : 2, 9300392593962552996 : 55, 10675097053756216103 : 5, 3894691722888847399 : 26, 12377226622999180073 : 8, 17073933871812010409 : 50, 12074334133672640299 : 32, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2363072524549189424 : 36, 5394199281612494641 : 24, 4065202093792153394 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 4648258120096471861 : 34, 18446744073709551615 : 0, 12141599168644091063 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9559977705419941565 : 37, 18446744073709551615 : 0, 13306234163714742719 : 4, 11262107274662905535 : 30, 530158360389107904 : 52, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9990571522047476550 : 56, 15333292942936916422 : 14, 11774614965375084744 : 17, 10664877904891632841 : 53, 11680260744968870730 : 58, 13476744534618048714 : 48, 16263945700423788235 : 51, 16192699245564830285 : 21, 18446744073709551615 : 0, 17894828814807794255 : 47, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11883052778684969816 : 29, 16446418064672675289 : 57, 18446744073709551615 : 0, 18446744073709551615 : 0, 12924601121332542940 : 31, 18446744073709551615 : 0, 1400211835565554014 : 42, 12246717677444511711 : 10, 18446744073709551615 : 0, 13948847246687475681 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11392981922992884965 : 49, 6047760552197722086 : 20, 13095111492235848935 : 6, 498925369617004392 : 33, 2977939252735254503 : 44, 17050385346227356138 : 54, 16356049002575341547 : 9, 8911460792948664812 : 1, 2457359615429717740 : 38, 5253845536319192686 : 41, 10073597086696680687 : 18, 18446744073709551615 : 0, 13187096597848308337 : 7, 17414050244986898033 : 11, 669435740520310387 : 39, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4483588064301240696 : 45, 18446744073709551615 : 0, 7403768006548035322 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9033280618676961023 : 46},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 9.80909e-45, count = 2), catboost_ctr_mean_history(sum = 5.60519e-45, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 4), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 5.60519e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [7, 2, 4, 2, 2, 4, 3, 1, 2, 1, 1, 4, 2, 1, 1, 4, 4, 3, 1, 3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                ),
                768791580653471469 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 8473802870189803490 : 0, 7071392469244395075 : 3, 18446744073709551615 : 0, 8806438445905145973 : 2, 619730330622847022 : 1, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.94273e-44, count = 61), catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5)],
                    ctr_total = [21, 61, 0, 12, 0, 1, 1, 5]
                ),
                10041049326689763398 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14414048846158871559 : 12, 18446744073709551615 : 0, 14638989218618174473 : 41, 1649460783742239370 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 11769730408401037837 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5779175314227622928 : 38, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1323165403597995158 : 26, 18428889556763151895 : 29, 2164667594568318358 : 27, 18446744073709551615 : 0, 17742152858060733338 : 4, 18446744073709551615 : 0, 8312851984779745692 : 31, 4291722588563259037 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11692867453113909922 : 22, 17412370521597043875 : 36, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13700470340432613801 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 11120589457824038700 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 7468055678118005295 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2318342633131784245 : 13, 5810832953686117174 : 33, 5297741329439372599 : 40, 3732592916305803191 : 37, 17895827896624199225 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 7723671602601903292 : 34, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12727582032159672000 : 23, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9858323221942535364 : 3, 5542904409860031684 : 16, 7800901740475460422 : 7, 18446744073709551615 : 0, 2673645599642895048 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3479611531780631629 : 28, 9935553359816092494 : 21, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5318359880926327638 : 32, 18446744073709551615 : 0, 17079190329591924568 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12509137837559755364 : 5, 4604868705542068453 : 8, 18446744073709551615 : 0, 15448722916603551847 : 17, 18084208487000104808 : 39, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9118915792783817837 : 19, 6796838119474563310 : 6, 15768090511475095279 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 3927579309257426674 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3949750004310756343 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 15091338242443513338 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16942373517120146559 : 30},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.4013e-44, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 9), catboost_ctr_mean_history(sum = 1.12104e-44, count = 5), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 7.00649e-45, count = 10), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 9.80909e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [10, 2, 2, 9, 8, 5, 1, 1, 5, 10, 1, 1, 7, 1, 2, 3, 1, 1, 2, 3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                ),
                768791580653471471 :
                catboost_ctr_value_table(
                    index_hash_viewer = {2136296385601851904 : 0, 7428730412605434673 : 5, 9959754109938180626 : 2, 14256903225472974739 : 3, 8056048104805248435 : 1, 18446744073709551615 : 0, 12130603730978457510 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10789443546307262781 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.8026e-44, count = 73), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [20, 73, 0, 2, 1, 0, 0, 1, 1, 0, 0, 2, 0, 1]
                ),
                17677952493260528848 :
                catboost_ctr_value_table(
                    index_hash_viewer = {3632340108106778112 : 12, 84580555217079201 : 5, 1856503610704726976 : 8, 12055230997206289283 : 2, 16771526449192646880 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3152779373080459276 : 4, 14225011642249373260 : 9, 18198689053211288334 : 6, 16796278652265879919 : 13, 4201158457639332815 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9710576150271683444 : 1, 6178854915050051732 : 0, 8308165749326275029 : 11, 4776444514104643317 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 28), catboost_ctr_mean_history(sum = 2.94273e-44, count = 17), catboost_ctr_mean_history(sum = 0, count = 11), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 28, 21, 17, 0, 11, 0, 2, 0, 2, 0, 1, 0, 2, 0, 5, 0, 3, 0, 2, 0, 4, 1, 0, 0, 1, 0, 1]
                ),
                17677952493265578087 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18171586759681088672 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14557079040021784102 : 1, 1894223316800506727 : 9, 18446744073709551615 : 0, 11879805695908083497 : 2, 11687820229804647466 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12879152732677505903 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4426716004344559893 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8230941806183355321 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 1064533880431424572 : 5, 17607571949008043997 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.8026e-44, count = 58), catboost_ctr_mean_history(sum = 0, count = 11), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [20, 58, 0, 11, 0, 1, 0, 2, 0, 3, 1, 0, 0, 1, 1, 0, 0, 2, 0, 1]
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
                17677952493260528850 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 5717724512618337697 : 2, 18446744073709551615 : 0, 5133782457465682915 : 12, 11196527390020060580 : 8, 11961955270333222981 : 9, 5761100149665496677 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15262016962202059306 : 3, 18446744073709551615 : 0, 11861182568623336748 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 12026216826389142735 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 3373069665683731858 : 1, 18288092504171651762 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13367377011060337464 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17153616595626919517 : 11, 15741577697228378142 : 6, 17780934287826733279 : 5},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 34), catboost_ctr_mean_history(sum = 2.8026e-44, count = 19), catboost_ctr_mean_history(sum = 0, count = 13), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 34, 20, 19, 0, 13, 0, 2, 0, 2, 0, 1, 0, 3, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
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
                5445777084271881947 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 10934650013725255009 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 5844492600899280932 : 2, 18446744073709551615 : 0, 1034166431492604838 : 3, 18446744073709551615 : 0, 6203552979315789704 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 1113395566489815627 : 0, 13957701839509617452 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9226604805100152147 : 6, 1601191413561926516 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 13302932820562179799 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5765263465902070143 : 1},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 10), catboost_ctr_mean_history(sum = 2.24208e-44, count = 10), catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 4.2039e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 22), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 3)],
                    ctr_total = [0, 10, 16, 10, 0, 12, 3, 7, 0, 22, 1, 1, 0, 8, 0, 6, 2, 0, 0, 3]
                ),
                768791580653471478 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 15379737126276794113 : 5, 18446744073709551615 : 0, 14256903225472974739 : 3, 18048946643763804916 : 6, 2051959227349154549 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7024059537692152076 : 4, 18446744073709551615 : 0, 15472181234288693070 : 1, 8864790892067322495 : 2},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 4.2039e-45, count = 6), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6), catboost_ctr_mean_history(sum = 1.4013e-44, count = 58), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 4)],
                    ctr_total = [3, 6, 1, 6, 10, 58, 1, 5, 5, 0, 2, 0, 0, 4]
                ),
                8405694746487331128 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 13987540656699198946 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18089724839685297862 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10353740403438739754 : 1, 3922001124998993866 : 0, 13686716744772876732 : 4, 18293943161539901837 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 5.1848e-44, count = 42), catboost_ctr_mean_history(sum = 1.82169e-44, count = 2), catboost_ctr_mean_history(sum = 5.60519e-45, count = 3)],
                    ctr_total = [37, 42, 13, 2, 4, 3]
                ),
                10041049327447366776 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 228832412018222341 : 6, 18446744073709551615 : 0, 11579036573410064263 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2142920538933900555 : 20, 18446744073709551615 : 0, 11420714090427158285 : 19, 18446744073709551615 : 0, 17720405802426315535 : 5, 3215834049561110672 : 33, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 346575239343974036 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 13139983920087306647 : 32, 14860408764928037144 : 1, 286844492446271769 : 41, 18446744073709551615 : 0, 18446744073709551615 : 0, 10925792178412610972 : 23, 12726869934920056605 : 27, 11945848411936959644 : 46, 18446744073709551615 : 0, 11343638620497380128 : 42, 9857611124702919969 : 11, 15541558334966787106 : 50, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10990677728635501222 : 45, 4919457811166910375 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 4237122415554814250 : 37, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 339035928827901487 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8200830002684883256 : 0, 6893797804197340345 : 13, 1058988547593232698 : 16, 11714417785040418747 : 14, 18446744073709551615 : 0, 6067291172676902717 : 31, 16636473811085647678 : 26, 18446744073709551615 : 0, 483329372556896832 : 30, 3198032362459766081 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12661894127993305031 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4340360739111205579 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1471101928894068943 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 464994231589622356 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14915048362378503384 : 10, 5278641733246315480 : 12, 1537907742216832473 : 29, 5054839022797264859 : 38, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6888411174261376229 : 34, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16939687026671270763 : 51, 14120581721888279787 : 36, 18080292852670312173 : 25, 7952734526884932333 : 47, 8723830392309106799 : 28, 9875412811804264560 : 21, 15038402360561546607 : 52, 16771855022716002162 : 17, 5933240490959917807 : 18, 7006154001587127924 : 15, 8813616260402002415 : 39, 18446744073709551615 : 0, 5540766610232480247 : 48, 18446744073709551615 : 0, 16586264761736307193 : 44, 18446744073709551615 : 0, 6712598941894663547 : 49, 17585370940655764860 : 3, 9392162505557741693 : 43, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 4.2039e-45, count = 6), catboost_ctr_mean_history(sum = 4.2039e-45, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 5), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 7), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.12104e-44, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 4.2039e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 7.00649e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [3, 6, 3, 2, 3, 5, 2, 1, 1, 2, 2, 7, 1, 1, 8, 3, 1, 2, 1, 2, 1, 1, 1, 3, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                ),
                8405694746487331129 :
                catboost_ctr_value_table(
                    index_hash_viewer = {7537614347373541888 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5903587924673389870 : 4, 18278593470046426063 : 9, 10490918088663114479 : 8, 18446744073709551615 : 0, 407784798908322194 : 5, 5726141494028968211 : 6, 1663272627194921140 : 7, 8118089682304925684 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15431483020081801594 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 1403990565605003389 : 0, 3699047549849816830 : 1, 14914630290137473119 : 2},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.66247e-44, count = 2), catboost_ctr_mean_history(sum = 1.4013e-44, count = 20), catboost_ctr_mean_history(sum = 3.92364e-44, count = 5), catboost_ctr_mean_history(sum = 5.60519e-45, count = 3), catboost_ctr_mean_history(sum = 4.2039e-45, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2)],
                    ctr_total = [19, 2, 10, 20, 28, 5, 4, 3, 3, 4, 1, 2]
                ),
                5783086744600289132 :
                catboost_ctr_value_table(
                    index_hash_viewer = {14931585970071951136 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 5397540690114647075 : 8, 8825784475868822724 : 11, 10918373804899154693 : 4, 18446744073709551615 : 0, 9860698619030651943 : 7, 17528105968102438951 : 15, 10200488935662485831 : 3, 11300006847281354472 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 15718091127471100013 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 1888464406455459152 : 1, 16939442126893167761 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12990050366695140501 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 16503206593760246744 : 13, 18446744073709551615 : 0, 3937733491348552474 : 14, 18446744073709551615 : 0, 8258953332285675196 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11772109559350781439 : 5},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.96182e-44, count = 6), catboost_ctr_mean_history(sum = 1.4013e-44, count = 10), catboost_ctr_mean_history(sum = 1.54143e-44, count = 5), catboost_ctr_mean_history(sum = 2.24208e-44, count = 2), catboost_ctr_mean_history(sum = 1.82169e-44, count = 2), catboost_ctr_mean_history(sum = 7.00649e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [14, 6, 10, 10, 11, 5, 16, 2, 13, 2, 5, 3, 1, 1, 1, 1]
                ),
                5445777084271881951 :
                catboost_ctr_value_table(
                    index_hash_viewer = {11772109559350781439 : 4, 18446744073709551615 : 0, 12337359831519453058 : 8, 18446744073709551615 : 0, 3462861689708330564 : 10, 18446744073709551615 : 0, 9955981968190923718 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 7606262797109987753 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6973539969458659060 : 2, 13860744542689514389 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 16503206593760246744 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 2242442935049193755 : 3, 18446744073709551615 : 0, 8193958724117795869 : 9, 10924139913308365886 : 6, 14687079002600389023 : 1},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 1.12104e-44, count = 18), catboost_ctr_mean_history(sum = 2.8026e-45, count = 40), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2)],
                    ctr_total = [1, 3, 1, 4, 8, 18, 2, 40, 0, 2, 5, 0, 2, 0, 1, 3, 2, 3, 0, 1, 0, 3, 0, 2]
                ),
                8405694746487331131 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 14452488454682494753 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1388452262538353895 : 8, 8940247467966214344 : 2, 4415016594903340137 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 41084306841859596 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8678739366408346384 : 1, 18446744073709551615 : 0, 4544226147037566482 : 11, 14256903225472974739 : 5, 16748601451484174196 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5913522704362245435 : 3, 1466902651052050075 : 7, 2942073219785550491 : 12, 15383677753867481021 : 6, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.24208e-44, count = 6), catboost_ctr_mean_history(sum = 1.4013e-44, count = 11), catboost_ctr_mean_history(sum = 1.54143e-44, count = 6), catboost_ctr_mean_history(sum = 2.24208e-44, count = 2), catboost_ctr_mean_history(sum = 1.82169e-44, count = 5), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1)],
                    ctr_total = [16, 6, 10, 11, 11, 6, 16, 2, 13, 5, 3, 1, 1]
                ),
                18024574529690920892 :
                catboost_ctr_value_table(
                    index_hash_viewer = {17151879688829397503 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 14282620878612260867 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 12420782654419932198 : 2, 18446744073709551615 : 0, 15473953381485119304 : 6, 9743509310306231593 : 3, 9551523844202795562 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10742856347075653999 : 5},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 2.10195e-44, count = 34), catboost_ctr_mean_history(sum = 8.40779e-45, count = 27), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2)],
                    ctr_total = [15, 34, 6, 27, 0, 7, 0, 1, 0, 5, 0, 3, 1, 2]
                ),
                15110161923527827288 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4774468950989647619 : 24, 16020701755717470212 : 20, 4621308622046045445 : 6, 2919179052803081475 : 44, 10471866573136752518 : 32, 8576582476037552648 : 49, 7882246132385538057 : 8, 437657922758861322 : 1, 9192553231179393160 : 46, 6522474812938725258 : 19, 15226786739838940812 : 37, 18446744073709551615 : 0, 8940247374797094543 : 10, 18446744073709551615 : 0, 10642376944040058513 : 29, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6568726319216336023 : 36, 17376709210067783448 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 12775563440030710555 : 38, 11060530414346023708 : 52, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15845090011728125473 : 14, 18446744073709551615 : 0, 17547219580971089443 : 23, 18446744073709551615 : 0, 3463723719475269925 : 27, 1861503866934578725 : 53, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9705953625054433194 : 15, 8981764044004406698 : 28, 18446744073709551615 : 0, 18446744073709551615 : 0, 17054840337186596654 : 34, 18446744073709551615 : 0, 18446744073709551615 : 0, 1791377778529259953 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15858939668766397237 : 42, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2358208922112407481 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 1961888149432565948 : 39, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9453866756073833665 : 17, 3335160722738945090 : 12, 12567366267225461315 : 2, 826589723772749506 : 50, 2201294183566412613 : 5, 13867632926408595525 : 26, 3903423752809376583 : 7, 8600131001622206919 : 45, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12336013728068937550 : 35, 18446744073709551615 : 0, 14038143297311901520 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 14621199323616219987 : 33, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10206141410907137496 : 41, 18446744073709551615 : 0, 18446744073709551615 : 0, 15171514601954139611 : 22, 18446744073709551615 : 0, 4832431293524939229 : 4, 10503099563908856030 : 47, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6859490072747112932 : 13, 18446744073709551615 : 0, 3300812095185281254 : 16, 2191075034701829351 : 48, 5002941664428245224 : 31, 18446744073709551615 : 0, 18446744073709551615 : 0, 7718896375375026795 : 21, 18446744073709551615 : 0, 9421025944617990765 : 43, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3409249908495166326 : 25, 7972615194482871799 : 51, 18446744073709551615 : 0, 18446744073709551615 : 0, 4450798251142739450 : 30, 18446744073709551615 : 0, 18446744073709551615 : 0, 3772914807254708221 : 9, 18446744073709551615 : 0, 5475044376497672191 : 11},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 5.60519e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [1, 4, 2, 0, 2, 3, 0, 2, 0, 2, 0, 4, 1, 2, 0, 2, 0, 1, 0, 1, 1, 4, 0, 1, 0, 1, 0, 1, 2, 2, 4, 0, 1, 3, 0, 1, 0, 3, 1, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 5, 0, 1, 1, 1, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 2, 0, 1, 1, 0, 1, 0, 1, 0, 0, 5, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
                ),
                8405694746487331134 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 15379737126276794113 : 5, 18446744073709551615 : 0, 14256903225472974739 : 3, 18048946643763804916 : 6, 2051959227349154549 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7024059537692152076 : 4, 18446744073709551615 : 0, 15472181234288693070 : 1, 8864790892067322495 : 2},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.26117e-44, count = 7), catboost_ctr_mean_history(sum = 9.52883e-44, count = 6), catboost_ctr_mean_history(sum = 7.00649e-45, count = 2)],
                    ctr_total = [9, 7, 68, 6, 5, 2, 4]
                ),
                17677952493039013652 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 7591082115406918913 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16609914423024815883 : 79, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2947729946697366032 : 67, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1747100377421337877 : 45, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 688779167165725721 : 34, 18446744073709551615 : 0, 2390908736408689691 : 59, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12485828686297784107 : 76, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18116750275904253490 : 2, 15080985694568606770 : 26, 3441749364927451444 : 39, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8196669531629639738 : 47, 18446744073709551615 : 0, 16350523875373403452 : 65, 1910130166587793469 : 14, 3210283187799748413 : 51, 3612259735830757439 : 36, 2100546127316296510 : 72, 18446744073709551615 : 0, 16174754976619755330 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4712533024756798802 : 66, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1951646835795105624 : 33, 2675836416845132120 : 44, 13268280384053818970 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13208004644029510495 : 30, 7791717225000005216 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8958633917173958501 : 29, 18446744073709551615 : 0, 18446744073709551615 : 0, 944833316659531112 : 7, 6615501587043447913 : 71, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17286180302682250607 : 21, 18446744073709551615 : 0, 17859958192029424753 : 46, 18446744073709551615 : 0, 1115343687562837107 : 38, 18446744073709551615 : 0, 16021250581815544693 : 52, 18446744073709551615 : 0, 18446744073709551615 : 0, 15754561104342592632 : 70, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14489631383475886718 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18332060904098851720 : 11, 18446744073709551615 : 0, 1587446399632264074 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17478325149647224974 : 77, 18446744073709551615 : 0, 733710645180637328 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16419568529928495766 : 61, 18446744073709551615 : 0, 18446744073709551615 : 0, 3244631815353412249 : 15, 12476837359839928474 : 43, 1709840197601582745 : 73, 2110765276180879772 : 62, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7340761787208052128 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13489111233202375331 : 53, 18446744073709551615 : 0, 942497986273570725 : 78, 8887965463165302438 : 49, 18446744073709551615 : 0, 7907592930791199656 : 54, 13313342334448727209 : 18, 18446744073709551615 : 0, 7511272158111358123 : 50, 18446744073709551615 : 0, 16891671241997922733 : 55, 18446744073709551615 : 0, 18446744073709551615 : 0, 15003250764752625840 : 20, 18022869816319413424 : 32, 14335805918303332017 : 68, 16037935487546295987 : 63, 6375973732451541681 : 75, 16420649963778722224 : 80, 1546465267828251574 : 74, 18446744073709551615 : 0, 11854284788259111864 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10381815302203731404 : 4, 3318721001109633485 : 35, 3219274514257604046 : 19, 8814972803518824399 : 17, 16760440280410556112 : 5, 10517102372761788369 : 27, 15825775943968466 : 9, 12408874081425905107 : 16, 9980034949543187408 : 31, 8850196103864073429 : 57, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10150545320446493403 : 58, 10024723128977295580 : 41, 18446744073709551615 : 0, 18446744073709551615 : 0, 9102024323793860319 : 69, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8828822460557096163 : 56, 18446744073709551615 : 0, 18446744073709551615 : 0, 8849718467411561702 : 23, 18446744073709551615 : 0, 10551848036654525672 : 60, 10000182259821531625 : 37, 18446744073709551615 : 0, 4692931370061970411 : 28, 2837641471875404267 : 64, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5537249059016160241 : 48, 18446744073709551615 : 0, 3123341690686710771 : 24, 18446744073709551615 : 0, 15320050618309627125 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5305896519859636476 : 42, 18446744073709551615 : 0, 7008026089102600446 : 3, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 3, 2, 0, 0, 1, 0, 1, 0, 1, 0, 3, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 1, 1, 0, 1, 0, 1, 2, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 4, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 3, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 2, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1]
                ),
                17677952491747546147 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17787954881284471813 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7454420046185256717 : 20, 16256335682944813838 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1636731659193698578 : 48, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5922800847845598742 : 22, 14182197490569975831 : 27, 7624930417088562712 : 52, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10422205982269444643 : 44, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3411314423057176877 : 30, 18446744073709551615 : 0, 18446744073709551615 : 0, 4522605207985801776 : 29, 18446744073709551615 : 0, 13192676729576349746 : 62, 16466569643076362291 : 8, 18300934243650069811 : 58, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 4431368220400894274 : 60, 18446744073709551615 : 0, 18446744073709551615 : 0, 14233673023285815109 : 50, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2899749022061236299 : 53, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8023290181753164880 : 65, 9933882341717515345 : 66, 3233597379123467602 : 47, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8402263143377857370 : 37, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17651556054977644126 : 21, 15680080812126751838 : 55, 17708725746287261024 : 28, 18446744073709551615 : 0, 1780070264439091554 : 19, 15773274901763725923 : 0, 16328374789029446500 : 51, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16177106547947603049 : 13, 18446744073709551615 : 0, 17879236117190567019 : 3, 3489127981302646635 : 41, 14241655703424067948 : 56, 15943785272667031918 : 43, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9771448801094703501 : 67, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11530748061647284369 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 5994047302704556692 : 57, 18446744073709551615 : 0, 18446744073709551615 : 0, 10117199296271121559 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 9999128863307626394 : 5, 18446744073709551615 : 0, 11701258432550590364 : 6, 7854656800704835228 : 39, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 118997543255608737 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 10779812027622989220 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 6111396989577705639 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16127325828303939500 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 5576091289432675759 : 49, 14224606228188042159 : 59, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14966077412008197812 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 10025163577623610551 : 32, 1755789550731085240 : 64, 7501413217152384697 : 14, 16355005890516862393 : 46, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14797650915799523780 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13730933025438975688 : 35, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 724243645116964305 : 42, 18446744073709551615 : 0, 11702735195037717203 : 31, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16072394239031333591 : 45, 18446744073709551615 : 0, 11159883566315996889 : 34, 11603752796664724186 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16142728109259286750 : 10, 18446744073709551615 : 0, 17844857678502250720 : 12, 18446744073709551615 : 0, 9628264367976338914 : 16, 15813441649188061154 : 61, 18446744073709551615 : 0, 18446744073709551615 : 0, 2145056323740669926 : 40, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9516068082126538479 : 54, 18446744073709551615 : 0, 18446744073709551615 : 0, 10037970161273910770 : 38, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17560274819071548920 : 23, 11038948726666369272 : 25, 18446744073709551615 : 0, 8596718462362217979 : 63, 18446744073709551615 : 0, 10298848031605181949 : 33, 16215728555360712189 : 36, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 4), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 6), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 2, 1, 0, 0, 1, 3, 4, 0, 4, 0, 3, 0, 4, 0, 3, 0, 1, 0, 1, 0, 1, 0, 4, 0, 1, 1, 6, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
                ),
                9867321491374199500 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 3581428127016485793 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14455983217430950149 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13125720576600207402 : 5, 5967870314491345259 : 6, 9724886183021484844 : 7, 2436149079269713547 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1236773280081879954 : 1, 16151796118569799858 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8312525161425951098 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13605281311626526238 : 9, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 18), catboost_ctr_mean_history(sum = 1.96182e-44, count = 9), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 0, count = 19), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 1.12104e-44, count = 11), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3)],
                    ctr_total = [0, 18, 14, 9, 0, 7, 0, 19, 0, 6, 0, 2, 8, 11, 0, 2, 0, 2, 0, 3]
                ),
                10041049327410906820 :
                catboost_ctr_value_table(
                    index_hash_viewer = {16259707375369223360 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13847085545544291780 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7654620248348398600 : 2, 18446744073709551615 : 0, 9243796653651753418 : 5, 18446744073709551615 : 0, 1681026541770505292 : 22, 1292491219513334285 : 21, 13677090684479491854 : 23, 6494991755595340494 : 15, 7494438315637327440 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18410271455579776277 : 14, 6336919059871405781 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9974519673449003035 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5899717636280359390 : 13, 18446744073709551615 : 0, 15904544917366469984 : 1, 18446744073709551615 : 0, 862592111642406882 : 25, 18446744073709551615 : 0, 18446744073709551615 : 0, 18161161563956788133 : 11, 18446744073709551615 : 0, 3340544229935902247 : 12, 18446744073709551615 : 0, 14827488318775688873 : 16, 15675535932091499306 : 3, 18446744073709551615 : 0, 15230422751883885548 : 24, 18446744073709551615 : 0, 1662085889209686126 : 27, 18446744073709551615 : 0, 1062699037197581552 : 4, 14072903496117963889 : 17, 18446744073709551615 : 0, 15434641073738489523 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14277121817972567864 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 18160464660109825851 : 9, 16406258951888748923 : 18, 17480885798804750972 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.68156e-44, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 11), catboost_ctr_mean_history(sum = 1.54143e-44, count = 8), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 1.54143e-44, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 7), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 5.60519e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [12, 2, 3, 11, 11, 8, 1, 5, 11, 2, 1, 7, 2, 2, 3, 2, 2, 2, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1]
                ),
                10041049327412019993 :
                catboost_ctr_value_table(
                    index_hash_viewer = {12653400145582130496 : 1, 13035191669214674561 : 5, 7188816836410779808 : 4, 9634357275635952003 : 6, 3254436053539040417 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16061267211184267017 : 11, 18446744073709551615 : 0, 18446744073709551615 : 0, 18140574223860629292 : 14, 707921246595766797 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 9130812135695277968 : 2, 4844161989476173969 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13514752404240993397 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 3490899219630952952 : 10, 18446744073709551615 : 0, 1155235699154202746 : 9, 15274270191700277019 : 7, 18446744073709551615 : 0, 12264198141704391741 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 1.68156e-44, count = 36), catboost_ctr_mean_history(sum = 8.40779e-45, count = 21), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 7.00649e-45, count = 4), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1)],
                    ctr_total = [12, 36, 6, 21, 1, 2, 2, 2, 1, 5, 5, 4, 2, 1, 1]
                ),
                9867321491374199502 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 10934650013725255009 : 10, 18446744073709551615 : 0, 18446744073709551615 : 0, 5844492600899280932 : 2, 18446744073709551615 : 0, 1034166431492604838 : 1, 18446744073709551615 : 0, 6203552979315789704 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 1113395566489815627 : 0, 13957701839509617452 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18034029854971645104 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 9226604805100152147 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13302932820562179799 : 3, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 15316838452862012827 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5765263465902070143 : 7},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 10), catboost_ctr_mean_history(sum = 1.54143e-44, count = 7), catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 0, count = 12), catboost_ctr_mean_history(sum = 0, count = 10), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 8), catboost_ctr_mean_history(sum = 1.12104e-44, count = 10), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 10, 11, 7, 0, 12, 0, 12, 0, 10, 1, 1, 0, 8, 8, 10, 2, 2, 0, 6, 0, 1]
                ),
                17677952493294194665 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 14590129323275859969 : 21, 4761737228040236034 : 13, 3061679160699873539 : 41, 4729919983694621444 : 51, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8806247999156649096 : 39, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5682453444175299852 : 36, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 285702114596126864 : 30, 5632299938741154192 : 57, 18446744073709551615 : 0, 8333726745003022227 : 49, 15952973246181437460 : 40, 6343051352217846420 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 15549382406229051928 : 54, 18446744073709551615 : 0, 18446744073709551615 : 0, 16446935983613358747 : 19, 343067263211379228 : 58, 1340253711992729245 : 12, 7024200922256596382 : 56, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16261345722633663778 : 44, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14166509076554175142 : 37, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17479732393699636012 : 16, 18446744073709551615 : 0, 2982585812714178350 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3980241388377606578 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 10355191035340545717 : 4, 18446744073709551615 : 0, 3197060372330228023 : 15, 17395486763323672120 : 23, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 3025070705076148156 : 5, 10412716033556257724 : 31, 21656200241330621 : 38, 18446744073709551615 : 0, 7236416537553286208 : 8, 4097984215703358273 : 7, 2905510342784400193 : 43, 3540440309374370371 : 46, 9945806070767526596 : 34, 10600873139309967557 : 52, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 14677201154771995209 : 25, 15697257388296343881 : 48, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10665758689532857807 : 45, 9024898650572774224 : 42, 6436482984909430481 : 50, 3658183136700122066 : 17, 18446744073709551615 : 0, 8670866787526294612 : 10, 7478392914607336532 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18209445798681454937 : 0, 16902413600193912026 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 8198345533372667743 : 18, 18446744073709551615 : 0, 16817797835260737121 : 35, 13206648158456337762 : 33, 9272675415381189347 : 55, 18446744073709551615 : 0, 11952238979044267493 : 47, 18446744073709551615 : 0, 16311554771983004263 : 29, 18446744073709551615 : 0, 9562935439960121449 : 28, 18446744073709551615 : 0, 15862627151959278699 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 6949441338819150318 : 26, 18446744073709551615 : 0, 11479717724890640624 : 2, 12336975088890661616 : 9, 16935540662586488816 : 27, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11512437900041031286 : 53, 18446744073709551615 : 0, 9068013527945574136 : 3, 6476920084665523449 : 11, 1146182889791425528 : 32, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 5.60519e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 6), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 8.40779e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 3, 4, 0, 0, 2, 0, 2, 0, 1, 0, 6, 0, 2, 0, 3, 0, 1, 0, 1, 0, 2, 0, 1, 2, 7, 0, 1, 0, 1, 6, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 2, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3, 0, 1, 2, 2, 0, 1, 1, 0, 0, 1, 0, 2, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                ),
                10041049327393282700 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 15869504672553169153 : 14, 3630895197587547650 : 13, 18446744073709551615 : 0, 18069894976246263428 : 12, 6657459529952533892 : 15, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 12403109157674544908 : 21, 7581495141437254476 : 11, 18446744073709551615 : 0, 544300312823816335 : 26, 8994715059648341648 : 25, 18446744073709551615 : 0, 7582268711114204562 : 7, 9997066275032783314 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 8155965639411439190 : 5, 18446744073709551615 : 0, 17626120688916674776 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5135391889252992221 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11422884619242973793 : 9, 3129976559056829986 : 20, 10518099770818402979 : 10, 11182690403015408099 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 2283527241891053351 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10921182301457540139 : 3, 4851313952246684459 : 22, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7647347103951847349 : 0, 5184516154834744246 : 27, 18446744073709551615 : 0, 1764719067482953144 : 23, 6066581188437978489 : 16, 8257839345965546298 : 17, 12150488944147554235 : 24, 16694931389731688508 : 6, 18446744073709551615 : 0, 18446744073709551615 : 0, 9376384394070575999 : 18},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 8.40779e-45, count = 3), catboost_ctr_mean_history(sum = 9.80909e-45, count = 6), catboost_ctr_mean_history(sum = 1.12104e-44, count = 9), catboost_ctr_mean_history(sum = 1.4013e-44, count = 6), catboost_ctr_mean_history(sum = 1.96182e-44, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 7), catboost_ctr_mean_history(sum = 1.4013e-45, count = 5), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [6, 3, 7, 6, 8, 9, 10, 6, 14, 2, 2, 1, 1, 7, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1]
                ),
                10041049327393282701 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 18446744073709551615 : 0, 5195954639254248834 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 12596183338487933509 : 3, 11415090325326527685 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 15694110684349775305 : 6, 3105842455684076810 : 15, 18446744073709551615 : 0, 11619308647181131660 : 11, 18446744073709551615 : 0, 7384862814707324430 : 16, 16546783282337640335 : 10, 13877983093189917584 : 20, 18446744073709551615 : 0, 18446744073709551615 : 0, 9803181056021273939 : 12, 17960003200548727507 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 15929159679822070487 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1885423701024940001 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6164444060048187621 : 4, 1036643838009237222 : 9, 18446744073709551615 : 0, 2089642022879543976 : 2, 3679105889079969577 : 17, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1862978297920111856 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 11528263922108981619 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7453461884940337974 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 16229983591748392701 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 5.60519e-45, count = 3), catboost_ctr_mean_history(sum = 1.12104e-44, count = 28), catboost_ctr_mean_history(sum = 3.64338e-44, count = 2), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 5), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1)],
                    ctr_total = [4, 3, 8, 28, 26, 2, 2, 1, 3, 5, 2, 2, 1, 1, 2, 1, 3, 1, 3, 1, 2]
                ),
                10041049327393282703 :
                catboost_ctr_value_table(
                    index_hash_viewer = {7515733889724454912 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 2160905354121516547 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13659069800549444297 : 3, 7791826943727985930 : 2, 7884511582485373322 : 7, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18022007786552474063 : 19, 18446744073709551615 : 0, 6068383991325515601 : 25, 7524725216182310545 : 24, 17609669744399151123 : 18, 18446744073709551615 : 0, 18446744073709551615 : 0, 11681580651965248598 : 15, 576145588900686679 : 22, 13155646805788779928 : 0, 18446744073709551615 : 0, 5849831644443487770 : 5, 3372332782322797723 : 17, 18446744073709551615 : 0, 9865453060805390877 : 8, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9800431588293194596 : 10, 9048109927352371876 : 11, 16801589031893337254 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2099530300070748010 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 4741992351141480365 : 21, 17321493568029573614 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2151914027663660914 : 6, 9012245698387122739 : 20, 3718664820244579636 : 23, 2925864759981622644 : 1, 15505365976869715893 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 0,
                    counter_denominator = 101,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 5.60519e-45, count = 3), catboost_ctr_mean_history(sum = 1.82169e-44, count = 25), catboost_ctr_mean_history(sum = 2.38221e-44, count = 1), catboost_ctr_mean_history(sum = 1.26117e-44, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 2), catboost_ctr_mean_history(sum = 4.2039e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 1)],
                    ctr_total = [4, 3, 13, 25, 17, 1, 9, 2, 1, 4, 1, 1, 2, 2, 3, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1]
                ),
                768791582259504189 :
                catboost_ctr_value_table(
                    index_hash_viewer = {18446744073709551615 : 0, 4639344346382560065 : 0, 6768655180658783362 : 24, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17601732372345076103 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 9253120901934657613 : 11, 18446744073709551615 : 0, 12494120118534626831 : 23, 18446744073709551615 : 0, 18446744073709551615 : 0, 15104639456961940114 : 1, 10170507820794899987 : 26, 18446744073709551615 : 0, 18446744073709551615 : 0, 17626484575418309142 : 27, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 10540782073424112667 : 3, 5606650437257072540 : 18, 18446744073709551615 : 0, 14838774965469232670 : 17, 18446744073709551615 : 0, 16546754159773737760 : 20, 8171065581604191777 : 28, 8376012298141440672 : 10, 17449294303896545953 : 13, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 2686709533857156199 : 15, 8500597432574416232 : 21, 4462546031259207335 : 25, 12885436920358718506 : 2, 6984702425902276202 : 12, 17008555610512316647 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 8962398883312995119 : 7, 10515720428538797616 : 19, 18446744073709551615 : 0, 11572918221740308402 : 29, 3982985296232888499 : 6, 646524893007459764 : 22, 582150902654165941 : 9, 5031364380791762038 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 7009060838202480955 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 5478643861907335871 : 8},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 1.4013e-45, count = 2), catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 1.4013e-44, count = 15), catboost_ctr_mean_history(sum = 0, count = 19), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 7.00649e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 4, 1, 2, 0, 7, 10, 15, 0, 19, 0, 1, 0, 2, 0, 2, 0, 1, 0, 2, 5, 0, 2, 0, 0, 1, 0, 1, 0, 4, 0, 3, 0, 1, 0, 1, 0, 3, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1]
                ),
                13000966989535245560 :
                catboost_ctr_value_table(
                    index_hash_viewer = {8818114060598530624 : 3, 11977580115339394176 : 10, 12461144179858147074 : 19, 18446744073709551615 : 0, 18446744073709551615 : 0, 11548157811212961413 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1644045280240179080 : 9, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 1214622976113746317 : 15, 9385139042667852302 : 11, 18446744073709551615 : 0, 16857617732403848144 : 8, 5290950991575773969 : 5, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 6524082897304633048 : 13, 13549211095007995929 : 1, 18446744073709551615 : 0, 18446744073709551615 : 0, 10600410912766660700 : 20, 18446744073709551615 : 0, 16279254845622426718 : 4, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 13987500608421715490 : 16, 18446744073709551615 : 0, 18446744073709551615 : 0, 14875900814865445861 : 7, 18446744073709551615 : 0, 7246483080929928871 : 18, 18446744073709551615 : 0, 3141970693103761833 : 2, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 11255179931714098353 : 14, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 17345413579475959033 : 6, 4347934941247810554 : 17, 271853569121629179 : 12, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0, 18446744073709551615 : 0},
                    target_classes_count = 2,
                    counter_denominator = 0,
                    ctr_mean_history = [catboost_ctr_mean_history(sum = 0, count = 7), catboost_ctr_mean_history(sum = 2.24208e-44, count = 10), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 4.2039e-45, count = 7), catboost_ctr_mean_history(sum = 0, count = 5), catboost_ctr_mean_history(sum = 0, count = 21), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 1.4013e-45, count = 0), catboost_ctr_mean_history(sum = 2.8026e-45, count = 0), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 4), catboost_ctr_mean_history(sum = 0, count = 3), catboost_ctr_mean_history(sum = 0, count = 2), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1), catboost_ctr_mean_history(sum = 0, count = 1)],
                    ctr_total = [0, 7, 16, 10, 0, 4, 3, 7, 0, 5, 0, 21, 0, 1, 0, 2, 0, 2, 0, 2, 0, 1, 1, 0, 2, 0, 0, 2, 0, 3, 0, 4, 0, 3, 0, 2, 0, 1, 0, 1, 0, 1]
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



