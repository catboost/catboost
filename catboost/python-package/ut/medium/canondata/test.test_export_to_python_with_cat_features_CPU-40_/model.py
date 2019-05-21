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
    binary_feature_count = 20
    tree_count = 40
    float_feature_borders = [
        [17.5, 23.5, 25, 29.5, 34.5, 36.5, 45.5, 46.5, 50, 54.5, 56, 57.5, 68.5],
        [38811, 79129.5, 126119, 128053.5, 129787.5, 167944, 185890.5, 197032.5, 203488.5, 204331, 215061, 216825, 243690.5, 246689, 288417.5, 303732.5, 318173.5, 337225.5, 401155.5, 553548.5],
        [5.5, 6.5, 10.5, 11.5, 12.5, 13.5, 14.5],
        [1087, 3280, 5842, 7493, 11356, 17537.5],
        [808.5, 1738, 1862, 1881.5, 1944.5, 2189.5, 2396],
        [17, 22, 35.5, 36.5, 38.5, 46.5, 49, 70]
    ]
    tree_depth = [3, 6, 4, 0, 6, 6, 4, 6, 6, 6, 2, 0, 6, 6, 6, 2, 5, 3, 4, 4, 5, 6, 6, 6, 6, 6, 2, 6, 3, 6, 6, 4, 6, 6, 6, 1, 6, 3, 2, 2]
    tree_split_border = [3, 5, 1, 6, 3, 6, 4, 3, 6, 2, 3, 1, 5, 6, 4, 1, 7, 4, 1, 9, 6, 6, 16, 19, 3, 4, 255, 5, 6, 5, 6, 7, 13, 3, 7, 3, 2, 7, 1, 1, 2, 8, 9, 6, 10, 2, 1, 7, 4, 3, 6, 6, 5, 4, 7, 1, 4, 11, 3, 4, 6, 5, 12, 1, 5, 4, 5, 2, 6, 4, 2, 6, 5, 3, 5, 5, 1, 11, 5, 5, 13, 5, 2, 6, 15, 5, 7, 17, 1, 1, 3, 7, 1, 4, 3, 255, 6, 3, 10, 4, 4, 4, 4, 3, 12, 2, 4, 1, 1, 3, 3, 6, 5, 1, 5, 4, 2, 8, 1, 4, 4, 2, 3, 2, 1, 7, 2, 7, 1, 5, 8, 18, 2, 4, 1, 2, 2, 2, 2, 1, 4, 5, 1, 1, 5, 7, 2, 20, 2, 4, 1, 1, 3, 4, 14, 1, 2, 3, 1, 13, 3, 6, 6, 2, 4, 5, 1, 5, 1, 8, 2, 3, 4, 4, 2, 3, 6, 1, 1]
    tree_split_feature_index = [4, 3, 3, 15, 4, 16, 3, 0, 3, 0, 16, 3, 2, 2, 14, 3, 2, 2, 14, 0, 1, 3, 1, 1, 4, 3, 6, 1, 4, 16, 14, 5, 0, 3, 1, 14, 1, 0, 2, 1, 14, 5, 1, 16, 1, 16, 16, 14, 3, 16, 2, 5, 16, 2, 5, 0, 4, 0, 5, 5, 4, 1, 0, 5, 15, 0, 13, 13, 16, 2, 17, 11, 10, 12, 13, 5, 14, 1, 10, 16, 0, 18, 3, 16, 1, 12, 5, 1, 15, 17, 10, 5, 3, 1, 15, 6, 4, 1, 0, 11, 16, 13, 2, 18, 1, 17, 11, 19, 13, 10, 2, 14, 2, 3, 1, 12, 5, 5, 14, 13, 3, 4, 11, 10, 8, 4, 18, 11, 3, 4, 0, 1, 12, 15, 9, 7, 2, 8, 9, 14, 7, 14, 10, 8, 0, 5, 19, 1, 15, 2, 18, 4, 11, 18, 1, 7, 4, 2, 3, 1, 19, 2, 0, 9, 14, 12, 4, 11, 3, 1, 12, 13, 13, 10, 11, 7, 3, 12, 11]
    tree_split_xor_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cat_features_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    one_hot_cat_feature_index = [9]
    one_hot_hash_values = [
        [-2114564283]
    ]
    ctr_feature_borders = [
        [1.99999905, 2.99999905, 5.99999905, 10.999999],
        [2.99999905, 5.99999905],
        [4.99999905, 10.999999],
        [1.99999905, 6.99999905, 9.99999905, 12.999999, 13.999999],
        [7.99999905, 8.99999905, 9.99999905, 10.999999, 11.999999, 12.999999, 13.999999],
        [8.99999905, 10.999999, 11.999999, 12.999999, 13.999999],
        [3.99999905, 6.99999905, 9.99999905, 12.999999, 13.999999],
        [2.99999905, 6.99999905, 7.99999905, 8.99999905, 9.99999905, 11.999999, 12.999999],
        [3.99999905, 4.99999905, 9.99999905, 10.999999, 11.999999, 12.999999],
        [7.99999905, 8.99999905, 9.99999905, 10.999999, 11.999999, 14.999999],
        [0.999998987, 9.99999905],
        [7.99999905, 8.99999905, 10.999999, 11.999999, 12.999999],
        [9.99999905, 10.999999, 13.999999]
    ]

    ## Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.0247826081417177, 0.007499999832361937, 0, 0, 0.004285714189921106, 0, 0, 0,
        0.007314130275453562, 0, 0, 0, 0.01462826055090712, 0.02507701808726935, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01478395709301162, 0, 0.004157142767097268, 0, 0.01860609784453417, 0.02659683736528569, 0, 0.007443749834876508, -2.571428456476758e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0.01638063876501454, 0.02380575259985709, 0.02411385550700692, 0, -0.0002288360502803083, 0, 0, 0, 0.004249299300415698, 0.01425210528800581, 0.02217085516721529, 0, 0, 0, 0.003648277052247476,
        0.02102152929009783,
        0.01371499773370333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00701371938954183, 0.005091488980583503, 0, 0, 0.00714328770250907, -0.0001850235434319913, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0179267105817776, 0, 0.02383749444039185, 0, -0.0004783483635810837, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.008749025933918865, -0.001111040774507343, 0.02272214498463282, 0.01086513779757344, -0.0001850235434319913, 0, -0.0003214433826480735, 0, 0, -0.0006105921764434946, 0, 0.007088628388631012, 0, 0, 0, 0,
        0.02084793112377555, 0.01407126823359207, 0.01966020724472699, 0.01409681958975294, 0, -0.0001836358668872683, 0, 0, 0, 0, 0.01299817614135207, 0.00993668716961521, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01779780838358081, 0.006597357416953818, 0, 0, 0, 0, -0.0006986816978977739, 0, 0.0052785958345046, 0, 0, 0, 0, 0, 0, 0, 0.007035463676904604, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0.01661034404668668, -0.0003399953468206781, 0.01625750478583812, 0, 0.01797847193471006, -0.0005549459141294521, 0.02333879949646473, 0, 0, 0, 0, 0, -0.0003119698043871852, 0, -0.0005994850376434002, 0,
        0.01710319211204323, 0.01560623241442419, 0, 0.02133122041910412, -0.001431117437408744, 0, 0, 0.01007604317454983, 0, 0, 0, 0.006369891624392348, 0, 0, 0, 0, 0, -0.0005562380441962263, 0, 0, 0, 0, 0, 0, 0, -0.0001780965036534573, 0, 0, 0, 0, 0, 0, 0.01105378717727718, 0.01806144256086388, 0, 0.02239584671512843, 0.002028875080027128, 0, 0, 0.01457495343093066, 0, 0, 0, 0, 0, 0, 0, 0, -0.0005359447956479413, 0, 0, 0, -0.0002839839112831344, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.006370013732816687, 0, 0, 0, 0.006406816811931343, 0, 0.01296054400301605, 0, 0.01083511588224894, 0, 0.006617147198109119, 0, 0.006606900931323229, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.006156436209887427, 0, 0, 0, 0, 0, 0, 0, 0, 0.006297835927400227, 0.006170145625969743, 0, 0, 0, 0.01244002977857584, 0.006335779592764959, 0.006156436209887427, 0.008142066865121457, 0.02239817059379595, 0, -0.0009890936438156461, 0.002521499756956008, 0.01489713896418543,
        0.009832332826629916, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.008729545586726201, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.001139838056355739, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0007214391363149076, 0, 0, 0, 0, 0, 0.007333271002267101, -0.0004365751334007293, -0.0003935825717301806, 0, 0.02032248901789579, 0, 0, 0, 0, 0, 0.01347084661762059, 0, 0, 0, 0.02182026701476901, 0,
        0.015289232841898, 0.02171247811263226, -0.001057503775003371, 0,
        0.01704638172022497,
        0.01305950117782678, 0.01111915488769635, 0, 0, -0.002293725291648447, 0, 0, 0, 0, 0.02035119977873903, 0, 0, 0, 0, 0, 0, 0.003805244496588547, 0.006018874996646361, 0.004098803055199486, 0, 0, 0, 0, 0, 0, 0.01649729716378107, 0, 0.005566323364480105, 0, 0, 0, 0, 0, 0, 0, 0, 0.006206163317760357, -0.001279192076735749, 0, 0, 0, 0, 0, 0, 0, 0.009050381002757778, 0, 0, 0, 0, 0, 0, 0, -0.0005105472844170011, -0.002597475182407176, 0, 0, 0, 0, 0, 0, 0.01130889559333052, 0, 0.005793273781542023,
        0.005385844187061546, 0.01941857520421757, 0, 0, 0, 0.00391823246447859, 0, 0, 0, 0.00534590030631105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005576299771857777, 0.01488036762149975, 0, 0.004384239072856996, 0, 0.01306543514107295, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.001544201336428375, 0, 0, 0, 0,
        0.005595052173153455, 0, 0, 0, 0.008966904110719518, 0, 0, 0, 0.005263422232486664, 0.005308924163405311, 0, 0, 0.0133478485888289, 0.01847785166942322, 0, 0, 0, 0, 0, 0, 0.005964256921444628, 0.009085975084231758, 0.006122527213291228, 0, 0, 0, 0, 0, 0.01230053809264916, 0.01795319008541956, 0.005583256692704017, 0.01372922288830219, 0, 0, 0, 0, 0, 0.01075849323712693, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.003307686521783279, 0.008676768565895478, 0.005868149622064753, -0.002321417476468854, 0, 0, 0, 0, 0, 0, 0, 0,
        0.006076608260217913, 0.008303440276433507, 0.0186665148366681, 0.01788919644323674,
        0.005638790756317237, 0, 0.01053981848122245, 0.001174995178066526, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01079998525243702, 0.009083587923269477, 0.01517333348219349, 0.004457572170928376, 0.005418920312467382, 0, 0.01636955021909233, 0.01591964322038451, 0, 0, 0, 0, 0.004975451352658202, 0, 0.01672418719922199, 0.0141832115937704,
        0.008515031091142948, 0, 0.009303092396085842, 0.0047580494780383, 0.01701626313341213, 0.01077939137817519, 0.01863349954684334, 0.005838629496368484,
        0.007410123953533556, 0.007447668844215554, 0, 0, 0.0148389650134614, 0.01632008461400023, 0.01535059716083284, 0.01496666322497187, 0, 0, 0, 0, 0.003436538289565867, 0, 0, 0,
        0.008043113672150272, 0.008863707740067561, -0.002779022065048918, 0, 0.01732719939352943, 0.01271846396455622, -0.001123277465242882, -0.0008779333428867616, 0.007194827895389249, 0.003053037892089363, 0, 0, 0.01490701217814606, 0.007699480208911019, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01174844865324022, 0, 0, 0, 0, 0, 0, 0.005660415419993548, 0.005719809451415818, 0, 0, 0, 0, 0, 0, 0.008131477981608825, 0.01665437608369359, -0.003755584937943314, 0.01184528354522408, 0.006757527864993045, 0.01458662848390498, 0.005688005695601196, 0.004822249820304314,
        0.009815808673365818, 0.005676910881489057, -0.002738488257391418, 0, 0, 0.003313805893891721, 0, 0, 0.009089366289678309, 0, 0.0001796486119598288, 0, -0.002370129589633149, 0, -0.0007087426720989659, 0, 0, 0.01284591705776267, 0, 0, 0, 0, 0, 0, 0.01053496034403189, 0.0148620324569916, 0.004625966587129457, 0.009470787336145311, -0.00180764905559765, 0, 0, 0, 0.005131415708224563, 0, 0, 0, 0, 0, 0, 0, 0.01092637790924198, 0, 0, 0, 0, 0, 0, 0, 0.007443032734062464, 0.00712549538854737, 0, 0, 0, 0, 0, 0, 0.007034771092401596, 0.01520851913719603, 0, 0.007360243458835096, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005634334050829555, 0, 0, 0, 0, 0, 0.01079300364911339, 0, 0, 0, 0.004952878936095024, 0, 0.01459874398790975, 0, 0.01595506938215916, 0, 0.004845939091957197, 0, 0.01270086341761359, 0, 0.003931555084570077, 0, 0.00694108433749622, -0.0009137888556818041, 0.005388520416609694, 0, 0.0007946823283007418, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.002719519028352444, 0, 0.01191854258034144, -0.001797885672811125, 0, 0, -0.002160494852479134, 0, 0, 0, 0, 0, 0, 0, -0.000982699293184129, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004833953455068581, 0.004915732344904602, 0.00797764484274654, 0, 0, 0, 0, 0, 0.005823507255115109, -0.001442450386737685, 0.006275339073262435, -0.001243862606799216, 0.009563432795315222, -0.001365678217910908, 0.009199228665853591, 0.004538165681593834, 0.007172300303840214, 0.005183752214194749, 0.004678667441646551, 0, 0, 0, 0, 0, 0.01254064606833146, 0.009963326955777696, 0.01395611207786985, 0.009780386934928461, 0.008414684947833826, 0.006845885008982301, 0.01270831689901905, 0.01158163220830316,
        0.007364784423076871, 0.01229575103845298, 0, 0, 0.00487886435314188, 0.0008576632787756104, 0, 0.004643577436618525, 0.008876155091382641, 0, 0.004227663678820355, 0.01497298201829729, 0, 0, 0.004188794659172639, 0, 0, 0, 0, 0, 0, 0.001065028002776247, 0, 0.0104178514935778, 0, 0, 0, 0, 0, 0, 0.004423913360484941, 0.01339987897340654, 0, -0.002460695710498288, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.002052365058855397, 0, 0.003179125538773643, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0.0113583269221106, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.008514448871033897, 0.01066937685305804, 0, 0, 0.01247903655077084, 0.01536732225891574, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0026974233704738, 0.005485297600937419, 0, 0.004601740157481047, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.008311401645786852, 0, 0, 0.005143941075118817, 0.005114048133815854, 0, -0.001033289211425023, 0, 0, 0, 0, 0, -0.0009055941799171835, 0, 0,
        0.01303648345009104, -0.002443377338881651, 0.002000723872004502, 0,
        0, 0, 0, 0, 0.007321411147033564, 0, 0, 0.007492063624388566, 0, 0, 0, 0, 0, 0, 0, 0, -0.002842414713563691, 0, -0.002202898597590788, 0, 0.01041480605527644, 0, 0.004366199575551555, 0.01139408344649925, 0, 0, 0, 0, 0, 0, -0.000913807652424253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01436942243294844, 0, 0, 0, 0, 0, 0, 0, 0,
        0.013285459882518, 0.0002489398279044267, -0.0009069540951842601, 0, 0.009168503674025124, -0.002276569060860518, 0.002776420864249444, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004367968071269606, 0, 0, 0, 0.003856715009152303, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00489666200044972, 0, 0, 0, 0, 0, 0.01029456547776284, 0.004081606367579858, -0.00287997528339103, 0, 0, 0, 0, 0, 0, 0, -0.004553751745548398, 0, -0.002921846584048689, 0, 0, 0, 0.006616433956674148, 0.003643670377349715, 0, 0, 0, 0, 0, 0, 0, 0, 0.005712281305514105, 0.01002485288479119, 0.01166084242980905, 0, -0.001369058226766301, 0, 0.01320693694582392, 0.01195519866731817,
        0, 0, 0, 0.004335208311467322, 0, 0, 0, 0.004455946721295716, 0, 0, 0, 0.004204245176551936, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.008085984153419795, 0, 0, 0, 0.00306731281043155, -0.004440116438556197, -0.001643904706121034, 0, 0.001229985941820646, 0, 0, 0, 0.00813183409926984, 0, 0, 0, 0.008382439182623533, 0, 0, 0, 0, 0.005731596331521457, 0.007217311045406128, 0, 0.01050203198835751, 0, 0, 0, 0, 0, 0, 0, 0.01270109012915595,
        0, 0.0111605746636502, -0.002187686624623275, 0.008878239825373562, 0, 0, -0.001568517879268077, 0, 0, 0.01267289245518098, -0.001921649851595754, 0.009159736831328685, 0, 0.006777636474702989, 0, 0.00117474620131004,
        0, 0, 0.004207647558568664, 0, 0, 0, 0.004353829096933539, 0, 0, 0, 0.008413702234228644, 0, 0.005015119248678753, 0, -0.001149339852572417, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004258027534417462, 0, 0, 0, -0.002794429896009371, 0, 0, 0, 0.004065670695797399, 0, 0, 0, 0.004674504181030696, 0, 0, 0, 0.01258485587377482, 0.005314977161098466, 0.004863951150657066, 0, 0.009142150335797192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.006848178382705651, 0,
        0.00132906321228425, 0.003585939967434997, 0.0005525726436208371, 0, 0.01207579344699353, 0.007906812813897379, 0.01111318848313069, 0.007645503608603501, -0.001939172683825251, 0, 0, 0, 0, 0.004078311990577124, 0, 0, 0.004164008396459506, -0.002547086118273767, -0.002234381740651598, 0, 0.008321976067768182, 0.009403027783327422, 0.005403897913233834, 0.006584963814366325, 0, 0, 0, 0, -0.002752513448506136, 0, 0, 0.00511694171898756, 0, 0, 0, 0, -0.003105511899597997, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.000653104309328621, -0.001381932406504768, 0, 0, 0.00463944540045659, -0.002332204115196976, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, -0.002524729161098779, -0.002217623877971279, 0, 0, -0.00157936400901029, -0.001672295202155335, 0, 0, 0.01199422176400864, 0.004478420958287578, 0.01024381537107674, 0.001812672100614711, -0.00130389467994245, 0, 0, 0, 0.003285111297959721, 0.007635377449163584, 0.004905107666980937, 0.003767700310616876, -0.004384335075018946, -0.00221676207846946, 0.004756948811043871, 0, 0.003177580894647305, 0.003312365925210134, 0, 0, 0, 0, 0, 0, 0.000553345102357763, 0.003406127142227255, 0, 0, 0.003403784112235348, 0, 0, 0, 0.01097063802169385, 0.008403738441606729, 0.005067334541844657, 0.006975564817912975, 0, 0, 0, 0, 0.007122623639051802, 0.005540443518205193, 0.008038286171598802, 0.006177868330812275, 0.003754454885519627, 0.005078564656952948, 0, 0,
        0.004062495555623578, 0.01253542506150402,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.008941442316593617, 0.003900289289372745, 0.009580620640785232, 0, 0, 0, 0, 0, 0.005846500963111323, 0, 0.008926495999137645, 0, 0, 0, 0, 0, 0.00674014583264209, -0.003416623039683938, 0, 0, -0.002778181140862613, 0, 0, 0, 0.004784163239636235, 0.002425412608577978, 0, 0, -0.002233081422516011, 0, 0, 0, 0.006677186485906239, 0, 0.01014376126473274, 0, -0.002285797293349569, 0, 0.004433805261305226, 0, 0, 0.005010006706891014, 0.01065484170042299, 0.00316446034118898, 0, 0, 0, 0,
        0.009464453439156419, 0.004137385492927821, 0, 0, 0.0114184154035552, 0.002518939207022206, 0.01123312130218802, 0,
        0.0005743024091949682, 0.01021802232022954, 0, -0.001489841063654289,
        0.002229518697513467, 0.01052276888166527, -0.002622151159890901, 0.009925859942424833
    ]
    model_ctrs = catboost_model_ctrs_container(
        used_model_ctrs_count = 13,
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
                    catboost_model_ctr(base_hash = 14216163332699387103, base_ctr_type = "Borders", target_border_idx = 0, prior_num = 0.5, prior_denom = 1, shift = -0, scale = 15),
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



