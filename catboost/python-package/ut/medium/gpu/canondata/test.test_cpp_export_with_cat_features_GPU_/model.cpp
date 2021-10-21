#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>

typedef unsigned long long int TCatboostCPPExportModelCtrBaseHash;

enum class ECatboostCPPExportModelCtrType {
    Borders,
    Buckets,
    BinarizedTargetMeanValue,
    FloatTargetMeanValue,
    Counter,
    FeatureFreq,
    CtrTypesCount
};

struct TCatboostCPPExportModelCtr {
    TCatboostCPPExportModelCtrBaseHash BaseHash;
    ECatboostCPPExportModelCtrType BaseCtrType;
    int TargetBorderIdx = 0;
    float PriorNum = 0.0f;
    float PriorDenom = 1.0f;
    float Shift = 0.0f;
    float Scale = 1.0f;

    inline float Calc(float countInClass, float totalCount) const {
        float ctr = (countInClass + PriorNum) / (totalCount + PriorDenom);
        return (ctr + Shift) * Scale;
    }
};

struct TCatboostCPPExportFloatSplit {
    int FloatFeature = 0;
    float Split = 0.f;
};

struct TCatboostCPPExportOneHotSplit {
    int CatFeatureIdx = 0;
    int Value = 0;
};

struct TCatboostCPPExportBinFeatureIndexValue {
    unsigned int BinIndex = 0;
    bool CheckValueEqual = 0;
    unsigned char Value = 0;
};

struct TCatboostCPPExportCtrMeanHistory {
    float Sum;
    int Count;
};

struct TCatboostCPPExportCtrValueTable {
    std::unordered_map<TCatboostCPPExportModelCtrBaseHash, unsigned int> IndexHashViewer;
    int TargetClassesCount;
    int CounterDenominator;
    std::vector<TCatboostCPPExportCtrMeanHistory> CtrMeanHistory;
    std::vector<int> CtrTotal;
    const unsigned int* ResolveHashIndex(TCatboostCPPExportModelCtrBaseHash hash) const {
        auto search = IndexHashViewer.find(hash);
        if (search == IndexHashViewer.end()) {
            return NULL;
        }
        return &search->second;
    }
};

struct TCatboostCPPExportCtrData {
    std::unordered_map<TCatboostCPPExportModelCtrBaseHash, TCatboostCPPExportCtrValueTable> LearnCtrs;
};

struct TCatboostCPPExportCompressedModelCtr {
    struct TCatboostCPPExportProjection {
        std::vector<int> transposedCatFeatureIndexes;
        std::vector<TCatboostCPPExportBinFeatureIndexValue> binarizedIndexes;
    } Projection;
    std::vector<TCatboostCPPExportModelCtr> ModelCtrs;
};

struct TCatboostCPPExportModelCtrs {
    unsigned int UsedModelCtrsCount;
    std::vector<TCatboostCPPExportCompressedModelCtr> CompressedModelCtrs;
    TCatboostCPPExportCtrData CtrData;
};

/* Model data */
static const struct CatboostModel {
    CatboostModel() {};
    unsigned int FloatFeatureCount = 6;
    unsigned int CatFeatureCount = 11;
    unsigned int BinaryFeatureCount = 13;
    unsigned int TreeCount = 20;
    std::vector<std::vector<float>> FloatFeatureBorders = {
        {30.5, 33.5, 36.5, 38.5, 41.5, 45.5, 52.5, 53.5, 54.5, 56, 58.5, 60.5},
        {38811, 51773, 73183.5, 115363.5, 116831.5, 117562, 119180.5, 160753.5, 164533.5, 188654.5, 204331, 215992, 222939, 303732.5, 318173.5, 325462, 337225.5},
        {8, 9.5, 10.5, 12.5, 13.5, 14.5, 15.5},
        {3280},
        {808.5, 1881.5, 2189.5, 2396},
        {34, 42, 46.5}
    };
    std::vector<unsigned int> TreeDepth = {6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
    std::vector<unsigned int> TreeSplits = {0, 60, 43, 1, 36, 44, 53, 0, 33, 40, 35, 60, 4, 41, 6, 22, 57, 36, 33, 10, 29, 13, 44, 43, 36, 15, 59, 18, 16, 65, 36, 43, 33, 38, 44, 2, 7, 30, 11, 42, 66, 57, 50, 51, 33, 67, 45, 46, 66, 44, 13, 17, 23, 25, 28, 60, 64, 36, 49, 47, 53, 19, 58, 14, 31, 47, 32, 57, 29, 44, 24, 36, 8, 57, 48, 20, 32, 53, 5, 46, 62, 61, 26, 36, 54, 30, 44, 13, 37, 33, 68, 27, 52, 21, 1, 13, 65, 36, 54, 12, 63, 69, 27, 52, 25, 5, 18, 54, 30, 53, 3, 55, 56, 65, 54, 39, 34, 9, 36};
    std::vector<unsigned char> TreeSplitIdxs = {1, 7, 3, 2, 1, 255, 1, 1, 5, 4, 7, 7, 5, 1, 7, 11, 4, 1, 5, 11, 1, 2, 255, 3, 1, 4, 6, 7, 5, 2, 1, 3, 5, 2, 255, 3, 8, 2, 12, 2, 3, 4, 6, 7, 5, 4, 1, 2, 3, 255, 2, 6, 12, 14, 17, 7, 1, 1, 5, 3, 1, 8, 5, 3, 3, 3, 4, 4, 1, 255, 13, 1, 9, 4, 4, 9, 4, 1, 6, 2, 2, 1, 15, 1, 1, 2, 255, 2, 1, 5, 5, 16, 8, 10, 2, 2, 2, 1, 1, 1, 3, 1, 16, 8, 14, 6, 7, 1, 2, 1, 4, 2, 3, 2, 1, 3, 6, 10, 1};
    std::vector<unsigned short> TreeSplitFeatureIndex = {0, 9, 5, 0, 3, 6, 8, 0, 2, 4, 2, 9, 0, 5, 0, 1, 9, 3, 2, 0, 2, 1, 6, 5, 3, 1, 9, 1, 1, 11, 3, 5, 2, 4, 6, 0, 0, 2, 0, 5, 11, 9, 7, 7, 2, 11, 7, 7, 11, 6, 1, 1, 1, 1, 1, 9, 11, 3, 7, 7, 8, 1, 9, 1, 2, 7, 2, 9, 2, 6, 1, 3, 0, 9, 7, 1, 2, 8, 0, 7, 10, 10, 1, 3, 9, 2, 6, 1, 4, 2, 11, 1, 7, 1, 0, 1, 11, 3, 9, 1, 10, 12, 1, 7, 1, 0, 1, 9, 2, 8, 0, 9, 9, 11, 9, 4, 2, 0, 3};
    std::vector<unsigned char> TreeSplitXorMask = {0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<unsigned int> CatFeaturesIndex = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<unsigned int> OneHotCatFeatureIndex = {9};
    std::vector<std::vector<int>> OneHotHashValues = {
        {-1291328762}
    };
    std::vector<std::vector<float>> CtrFeatureBorders = {
        {0.151785716f, 0.255357146f, 0.514285684f, 0.566071391f, 0.669642866f, 0.721428573f, 0.773214281f, 0.876785696f},
        {0.0539215691f},
        {0.4375f, 0.484375f, 0.53125f, 0.578125f, 0.625f, 0.671875f, 0.765625f},
        {0.744791687f, 0.802083373f, 0.859375f},
        {0.671875f, 0.7421875f, 0.7890625f, 0.8125f, 0.90625f},
        {0.0147058833f}
    };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[1248] = {
        0.02693549171090126, 0.016834681853652, 0.05611560866236687, 0.016834681853652, 0, 0, 0.02693549171090126, 0, 0, -0.06045181304216385, 0, 0.0332612507045269, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.003443456720560789, 0.016834681853652, 0.05471271649003029, 0.02693549171090126, 0, 0, 0.02693549171090126, 0, 0, -0.01258350163698196, 0, 0.0529089979827404, 0, -0.1555932760238647, 0, -0.01785496808588505, 0, -0.06045181304216385, 0, 0, 0, 0, 0, 0, 0, -0.1209036260843277, 0, -0.06045181304216385, 0, -0.06045181304216385, 0, 0,
        0.02506768889725208, 0.04182743281126022, -0.1176357716321945, 0.01666522584855556, 0, 0, -0.09188125282526016, -0.09032792598009109, 0, 0, 0, 0, 0, 0, 0, -0.0484265573322773, 0, 0, 0, 0, 0, 0, 0, 0.01780721545219421, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, -0.04758866876363754, 0, -0.03544744104146957, 0, 0, 0, 0, 0, 0, 0, -0.1149678826332092, 0, 0.01333585102111101, 0, 0, 0, -0.03657766059041023, 0, 0.01643096096813679, 0, 0, 0, 0, 0, 0, 0, -0.02113769575953484, 0, 0, 0.02602875605225563, 0, 0, 0.02206515520811081, 0.03829669579863548, -0.04668805003166199, -0.05368863418698311, 0, 0, 0, 0.01145753264427185, 0, 0, 0.01651922054588795, 0.04024598374962807, 0.01152023393660784, 0.02480306103825569, 0.01651922054588795, 0.01297604106366634, 0.01651922054588795, 0.03571422770619392, -0.09577208757400513, 0.01297604106366634, 0, 0, 0, 0, 0, 0, -0.03933489695191383, 0.04043979942798615,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02076880261301994, 0, 0, 0, 0.02408863231539726, 0, 0, 0, 0.02555848658084869, 0, -0.02688526175916195, 0, 0, 0, 0, 0, 0.00641361391171813, 0, 0, 0, 0, 0, 0, 0, -0.04082908853888512, 0, 0, 0, 0, 0, 0, 0, 0.03621402382850647, 0, 0, 0, 0, 0, 0, 0, 0.008785637095570564, -0.09384063631296158, -0.09785124659538269, -0.04452911764383316, 0.02491718344390392, -0.06961620599031448, 0.03094919770956039, 0,
        0.03752057999372482, -0.05430274456739426, 0, 0, 0, 0, 0, 0, 0.0262700542807579, 0, 0, 0, 0.008810088038444519, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.04832702875137329, 0, 0, 0, 0, 0, 0, 0, -0.04981405660510063, 0, 0, 0, 0, 0, 0, 0, 0.004693563561886549, -0.05462824925780296, -0.09412149339914322, -0.03054996952414513, 0, 0, 0, 0, 0.02383723855018616, -0.01504899561405182, -0.04108761623501778, 0,
        0.01952279172837734, 0.02030212618410587, 0, 0, 0.007748174481093884, 0.007748174481093884, 0, 0, 0.01852091774344444, -0.04596410319209099, 0, 0, 0, 0, 0, 0, -0.04689731448888779, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02128397673368454, 0.005550147034227848, -0.05912360548973083, -0.04936866834759712, 0.008823232725262642, -0.0435105599462986, 0, -0.02818886935710907, 0, -0.02517659589648247, -0.03791209682822227, 0, -0.00411313446238637, -0.05778161063790321, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0223609022796154, 0, 0, 0, -0.04493410885334015, 0, 0, 0,
        0.01927667669951916, -0.0122012784704566, 0, -0.02562474086880684, 0.003748495364561677, -0.01899564079940319, 0, 0.01599221862852573, 0, 0, 0, 0, 0, 0, 0, -0.03410714119672775, 0.01907448656857014, -0.01370074972510338, 0, 0, 0.005142920184880495, -0.04302866384387016, 0, 0.01375737600028515, 0, 0, 0, 0, 0, 0, 0, 0, 0.01111484132707119, 0.01384436152875423, 0, 0.004849039949476719, -0.01796090602874756, -0.03753593936562538, 0, 0.01600027456879616, 0, 0, 0, 0.01565166935324669, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.03091626241803169, 0, -0.03244799003005028, 0, 0, 0, 0, 0, 0, 0, 0,
        -0.02707350999116898, -0.04232979565858841, 0, 0, 0, 0, 0, 0, 0, 0.01855760999023914, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.03876397758722305, 0.000777918379753828, 0, 0.009220310486853123, 0, 0, -0.005969627760350704, 0.01148187182843685, 0.02395823784172535, -0.05185458809137344, 0, 0, 0, 0, -0.0622195191681385, 0.00130667828489095, -0.06120626628398895, 0.005886656697839499, 0, 0.01420364063233137, 0, 0, 0.02068913541734219, 0.02124432474374771, 0, -0.04393156617879868, 0, 0, 0, 0, 0, 0.01290311757475138,
        0, 0, 0, 0, 0, -0.02440501749515533, 0, 0, 0, -0.0007252611685544252, 0, -0.02240169607102871, 0, 0.01725051738321781, 0, 0.01617801934480667, 0, 0, 0, 0, 0, 0, 0, 0, -0.01754793338477612, 0.01511536259204149, 0, 0.01193950697779655, -0.05382330715656281, -0.02424784377217293, 0, -0.01213663816452026, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.006040013395249844, 0, 0.00644347257912159, -0.04524829611182213, 0.02768208086490631, -0.06565642356872559, 0.005230056121945381,
        0, 0, 0, 0, 0, 0, 0, 0.001154616009443998, -0.06717750430107117, -0.06672454625368118, 0, 0.04497268423438072, -0.006638107355684042, 0, 0, 0.003822522237896919, 0, 0, 0, 0, 0, 0, 0, 0, -0.02082127705216408, 0, 0, 0, -0.0276322066783905, 0, 0, 0, 0.01272711250931025, 0, 0, 0, 0.005188304465264082, 0, 0, 0, 0.001318523776717484, -0.03758146613836288, 0, 0.01048499438911676, 0.01310580596327782, 0.01231118384748697, 0, 0.009643109515309334, 0, 0, 0, 0, 0, 0, 0, 0, -0.05912145227193832, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, -0.02858741767704487, 0, 0, 0, 0, 0, 0, 0, 0.006622334476560354, 0, 0, 0, 0, -0.03294631466269493, 0, 0, 0.0284188911318779, -0.03659430891275406, 0, 0, 0.006740574724972248, -0.03251014277338982, 0.01072904001921415, 0, 0.008157648146152496, 0.0009698899812065065, 0, 0.003501784987747669, 0.008005658164620399, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.009820797480642796, 0, 0, 0, 0, 0, 0, 0, -0.02964661829173565, -0.01921207457780838, 0, -0.02380258217453957, -0.04232074320316315, -0.04032497853040695, 0, -0.03387097269296646, 0.003377727931365371, -0.01973405107855797, 0.002676635049283504, 0.01378667168319225, 0.008677917532622814,
        0, 0, 0, 0, 0, 0.004691216163337231, 0, 0, 0, 0, 0, 0.01352364104241133, -0.02999754808843136, 0.002496740547940135, -0.005570050328969955, 0.0005558785633184016, 0, 0.01710779592394829, 0, 0, 0, 0.004156660754233599, 0, 0, -0.03040000796318054, -0.008088712580502033, -0.01772724092006683, -0.04144030809402466, 0, 0.001927934004925191, -0.03533845394849777, 0.01732594333589077, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002472052583470941, 0.001330485451035202, -0.0001908160047605634, 0.003871968248859048, 0, 0.01122252084314823, 0, 0, 0.000794741150457412, 0.0005826323176734149, 0, 0, -0.06253682821989059, 0.03894615173339844, 0.01833097636699677, -0.03140610083937645, -0.01340639498084784, 0.007858364842832088, -0.00284273037686944, -0.002590719843283296,
        -0.028050497174263, 0, 0, 0, -0.02767914161086082, 0, 0, 0, 0.005121743772178888, 0, 0, 0, 0.004688972141593695, 0, 0.01964802481234074, 0, -0.03957171365618706, 0, -0.02893281169235706, 0, 0.003387592500075698, 0, -0.01717274263501167, 0, 0.02549606375396252, -0.02240447513759136, 0.01656786166131496, -0.01739497855305672, -0.002052380703389645, 0, 0.002711264882236719, 0, 0, 0, 0, 0, -0.03990764915943146, -0.0196488481014967, 0, 0, -0.01406504306942225, 0, 0, -0.01200511120259762, 0.003275195602327585, 0, 0.007544561289250851, 0, 0.01691423915326595, 0, 0, -0.01635716482996941, 0.0105605386197567, 0, 0, 0, -0.02062411792576313, -0.02120476588606834, 0, 0, -0.005713172256946564, 0, 0.02052039094269276, 0,
        0, 0.01754138991236687, -0.05452883988618851, 0, -0.0003603198856581002, -0.001603313605301082, -0.02728622779250145, 0.006405466236174107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.003439212683588266, 0, -0.01852335035800934, 0, -0.007570572663098574, 0, 0, 0, 0, 0.002630927134305239, -0.0006290119490586221, 0.00801873579621315, -0.0008414569892920554, 0, -0.03651335835456848, 0, 0, 0, 0.006405187305063009, 0, 0.01136835757642984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004545720759779215, 0, 0, 0, 0, 0, 0, 0, 0.004275229293853045, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0003328477323520929, 0, 0, 0, -0.02664374560117722, 0, 0, 0, 0.001278525218367577, 0, 0, 0, 0.001657701097428799, 0, 0.007889611646533012, 0, 0.006845497526228428, -0.02054899930953979, 0, 0, -0.005184169858694077, -0.03668992593884468, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.01024019438773394, 0, 0, 0, 0.004991895984858274, 0, 0, 0, 0.004828719887882471, 0, 0, 0, -0.008514708839356899, 0,
        0.002307872055098414, 0, -0.0006217224290594459, 0, 0, 0, 0, 0, 0.009394885040819645, 0, 0, 0, 0, 0, 0, 0, 0.001345049124211073, 0, 0.002288502175360918, 0, 0.001117915846407413, 0, -0.02337954007089138, 0, -0.002236610045656562, 0, 0, 0, 0, 0, 0, 0, -0.01625911705195904, -0.01485063042491674, 0.01610347256064415, -0.02836156636476517, 0, 0, 0, 0, 0.01195000391453505, 0, 0, 0.006867077201604843, 0, 0, 0, 0, -0.02086837030947208, -0.01258245017379522, 0.00422438932582736, 0, 0.01376893557608128, 0.01538644824177027, 0.003643847536295652, 0, -0.0316268689930439, -0.01857415772974491, 0.01336295250803232, 0, 0.004795695189386606, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, -0.0004854931903537363, 0, 0, 0, 0, 0, 0, 0, 0, 0.008892692625522614, 0, 0, 0, 0, 0, 0, -0.02332793362438679, 0.008391190320253372, 0, -0.008358562365174294, 0, -0.03078049793839455, 0, -0.01807161793112755, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001571401255205274, 0, 0, 0, 0,
        0, 0.001043740310706198, 0, 0, 0, 0.01273976545780897, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.04575853422284126, 0, 0, 0, 0.008656300604343414, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.003877149196341634, -0.0004702096630353481, 0, 0, 0, -0.01197487488389015, 0, 0, 0, 0, -0.0308377742767334, 0.004493621177971363, 0, 0, 0, 0.005071056541055441, 0, 0.00173116778023541, 0, 0, 0, 0.008548818528652191, 0, 0, 0, -0.03210081160068512, 0, 0.01986251212656498, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0.002748517086729407, 0, -0.007554010953754187, 0, -0.01805269531905651, 0, -0.005167258903384209, 0.008071610704064369, 0, 0, 0.01438718289136887, 0, 0, 0, 0, 0, 0, 0, -0.01214142329990864, 0, -0.01827161200344563, 0, 0, 0, 0.01362149510532618, 0, -0.003576485207304358, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.000345046108122915, 0, 0.001016982714645565, 0, -0.0004262780130375177, 0, -0.0003270286251790822, 0, -0.01140759605914354, 0, -0.02047092840075493, 0, 0.0159019660204649, 0, -0.007305529899895191,
        0.00744778523221612, 0, 0.002416926436126232, -0.004626938607543707, 0, 0, -0.01917820237576962, 0, 0, 0, 0.01229006238281727, -0.01500250585377216, 0, 0, -0.00808330625295639, 0, 0, 0, 0.02298767119646072, 0.004097537603229284, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.02458551712334156, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0148780345916748, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    double Scale = 1;
    double Bias = 0.7821782231;
    struct TCatboostCPPExportModelCtrs modelCtrs = {
        .UsedModelCtrsCount = 6,
        .CompressedModelCtrs = {
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791580653471478ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 8405694746487331134ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {7},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791580653471474ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791580653471469ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 768791580653471469ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {10},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 8405694746487331111ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            }
        },
        .CtrData = {
            .LearnCtrs = {
                {
                    768791580653471469ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8473802870189803490ull, 0}, {7071392469244395075ull, 2}, {18446744073709551615ull, 0}, {8806438445905145973ull, 3}, {619730330622847022ull, 1}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 2.94273e-44, .Count = 61}, {.Sum = 0, .Count = 12}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {21, 61, 0, 12, 1, 5, 0, 1}
                    }
                },
                {
                    768791580653471474ull,
                    {
                        .IndexHashViewer = {{3607388709394294015ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18356215166324018775ull, 2}, {18365206492781874408ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14559146096844143499ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11416626865500250542ull, 3}, {5549384008678792175ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 22}, {.Sum = 2.8026e-45, .Count = 3}, {.Sum = 0, .Count = 14}, {.Sum = 2.66247e-44, .Count = 17}, {.Sum = 0, .Count = 22}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {0, 22, 2, 3, 0, 14, 19, 17, 0, 22, 1, 1}
                    }
                },
                {
                    768791580653471478ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {15379737126276794113ull, 5}, {18446744073709551615ull, 0}, {14256903225472974739ull, 3}, {18048946643763804916ull, 1}, {2051959227349154549ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7024059537692152076ull, 6}, {18446744073709551615ull, 0}, {15472181234288693070ull, 2}, {8864790892067322495ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 1.4013e-44, .Count = 58}, {.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 6}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 4.2039e-45, .Count = 6}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 7.00649e-45, .Count = 0}},
                        .CtrTotal = {10, 58, 0, 4, 1, 6, 1, 5, 3, 6, 2, 0, 5, 0}
                    }
                },
                {
                    8405694746487331111ull,
                    {
                        .IndexHashViewer = {{2136296385601851904ull, 0}, {7428730412605434673ull, 2}, {9959754109938180626ull, 6}, {14256903225472974739ull, 1}, {8056048104805248435ull, 3}, {18446744073709551615ull, 0}, {12130603730978457510ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10789443546307262781ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 1.30321e-43, .Count = 1}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {93, 1, 2, 2, 1, 1, 1}
                    }
                },
                {
                    8405694746487331134ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {15379737126276794113ull, 5}, {18446744073709551615ull, 0}, {14256903225472974739ull, 3}, {18048946643763804916ull, 1}, {2051959227349154549ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7024059537692152076ull, 6}, {18446744073709551615ull, 0}, {15472181234288693070ull, 2}, {8864790892067322495ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 9.52883e-44, .Count = 4}, {.Sum = 9.80909e-45, .Count = 6}, {.Sum = 1.26117e-44, .Count = 2}},
                        .CtrTotal = {68, 4, 7, 6, 9, 2, 5}
                    }
                }
            }
        }
    };
} CatboostModelStatic;

static std::unordered_map<std::string, int> CatFeatureHashes = {
    {"Female", -2114564283},
    {"Protective-serv", -2075156126},
    {"Assoc-voc", -2029370604},
    {"Married-civ-spouse", -2019910086},
    {"Federal-gov", -1993066135},
    {"Transport-moving", -1903253868},
    {"Farming-fishing", -1888947309},
    {"Prof-school", -1742589394},
    {"Self-emp-inc", -1732053524},
    {"?", -1576664757},
    {"Handlers-cleaners", -1555793520},
    {"0", -1438285038},
    {"Philippines", -1437257447},
    {"Male", -1291328762},
    {"11th", -1209300766},
    {"Unmarried", -1158645841},
    {"Local-gov", -1105932163},
    {"Divorced", -993514283},
    {"Some-college", -870577664},
    {"Asian-Pac-Islander", -787966085},
    {"Sales", -760428919},
    {"Self-emp-not-inc", -661998850},
    {"Widowed", -651660490},
    {"Masters", -453513993},
    {"State-gov", -447941100},
    {"Doctorate", -434936054},
    {"White", -218697806},
    {"Own-child", -189887997},
    {"Amer-Indian-Eskimo", -86031875},
    {"Exec-managerial", -26537793},
    {"Husband", 60472414},
    {"Italy", 117615621},
    {"Not-in-family", 143014663},
    {"n", 239748506},
    {"Married-spouse-absent", 261588508},
    {"Prof-specialty", 369959660},
    {"Assoc-acdm", 475479755},
    {"Adm-clerical", 495735304},
    {"Bachelors", 556725573},
    {"HS-grad", 580496350},
    {"Craft-repair", 709691013},
    {"Other-relative", 739168919},
    {"Other-service", 786213683},
    {"9th", 840896980},
    {"Separated", 887350706},
    {"10th", 888723975},
    {"Mexico", 972041323},
    {"Hong", 995245846},
    {"1", 1121341681},
    {"Tech-support", 1150039955},
    {"Black", 1161225950},
    {"Canada", 1510821218},
    {"Wife", 1708186408},
    {"United-States", 1736516096},
    {"Never-married", 1959200218},
    {"Machine-op-inspct", 2039859473},
    {"7th-8th", 2066982375},
    {"Private", 2084267031},
};

static inline TCatboostCPPExportModelCtrBaseHash CalcHash(TCatboostCPPExportModelCtrBaseHash a, TCatboostCPPExportModelCtrBaseHash b) {
    const static constexpr TCatboostCPPExportModelCtrBaseHash MAGIC_MULT = 0x4906ba494954cb65ull;
    return MAGIC_MULT * (a + MAGIC_MULT * b);
}

static inline TCatboostCPPExportModelCtrBaseHash CalcHash(
    const std::vector<unsigned char>& binarizedFeatures,
    const std::vector<int>& hashedCatFeatures,
    const std::vector<int>& transposedCatFeatureIndexes,
    const std::vector<TCatboostCPPExportBinFeatureIndexValue>& binarizedFeatureIndexes) {
    TCatboostCPPExportModelCtrBaseHash result = 0;
    for (const int featureIdx : transposedCatFeatureIndexes) {
        auto valPtr = &hashedCatFeatures[featureIdx];
        result = CalcHash(result, (TCatboostCPPExportModelCtrBaseHash)valPtr[0]);
    }
    for (const auto& binFeatureIndex : binarizedFeatureIndexes) {
        const unsigned char* binFPtr = &binarizedFeatures[binFeatureIndex.BinIndex];
        if (!binFeatureIndex.CheckValueEqual) {
            result = CalcHash(result, (TCatboostCPPExportModelCtrBaseHash)(binFPtr[0] >= binFeatureIndex.Value));
        } else {
            result = CalcHash(result, (TCatboostCPPExportModelCtrBaseHash)(binFPtr[0] == binFeatureIndex.Value));
        }
    }
    return result;
}

static void CalcCtrs(const TCatboostCPPExportModelCtrs& modelCtrs,
                     const std::vector<unsigned char>& binarizedFeatures,
                     const std::vector<int>& hashedCatFeatures,
                     std::vector<float>& result) {
    TCatboostCPPExportModelCtrBaseHash ctrHash;
    size_t resultIdx = 0;

    for (size_t i = 0; i < modelCtrs.CompressedModelCtrs.size(); ++i) {
        auto& proj = modelCtrs.CompressedModelCtrs[i].Projection;
        ctrHash = CalcHash(binarizedFeatures, hashedCatFeatures,
                           proj.transposedCatFeatureIndexes, proj.binarizedIndexes);
        for (size_t j = 0; j < modelCtrs.CompressedModelCtrs[i].ModelCtrs.size(); ++j) {
            auto& ctr = modelCtrs.CompressedModelCtrs[i].ModelCtrs[j];
            auto& learnCtr = modelCtrs.CtrData.LearnCtrs.at(ctr.BaseHash);
            const ECatboostCPPExportModelCtrType ctrType = ctr.BaseCtrType;
            const unsigned int* bucketPtr = learnCtr.ResolveHashIndex(ctrHash);
            if (bucketPtr == NULL) {
                result[resultIdx] = ctr.Calc(0.f, 0.f);
            } else {
                unsigned int bucket = *bucketPtr;
                if (ctrType == ECatboostCPPExportModelCtrType::BinarizedTargetMeanValue || ctrType == ECatboostCPPExportModelCtrType::FloatTargetMeanValue) {
                    const TCatboostCPPExportCtrMeanHistory& ctrMeanHistory = learnCtr.CtrMeanHistory[bucket];
                    result[resultIdx] = ctr.Calc(ctrMeanHistory.Sum, ctrMeanHistory.Count);
                } else if (ctrType == ECatboostCPPExportModelCtrType::Counter || ctrType == ECatboostCPPExportModelCtrType::FeatureFreq) {
                    const std::vector<int>& ctrTotal = learnCtr.CtrTotal;
                    const int denominator = learnCtr.CounterDenominator;
                    result[resultIdx] = ctr.Calc(ctrTotal[bucket], denominator);
                } else if (ctrType == ECatboostCPPExportModelCtrType::Buckets) {
                    auto ctrIntArray = learnCtr.CtrTotal;
                    const int targetClassesCount = learnCtr.TargetClassesCount;
                    int goodCount = 0;
                    int totalCount = 0;
                    int* ctrHistory = ctrIntArray.data() + bucket * targetClassesCount;
                    goodCount = ctrHistory[ctr.TargetBorderIdx];
                    for (int classId = 0; classId < targetClassesCount; ++classId) {
                        totalCount += ctrHistory[classId];
                    }
                    result[resultIdx] = ctr.Calc(goodCount, totalCount);
                } else {
                    auto ctrIntArray = learnCtr.CtrTotal;
                    const int targetClassesCount = learnCtr.TargetClassesCount;

                    if (targetClassesCount > 2) {
                        int goodCount = 0;
                        int totalCount = 0;
                        int* ctrHistory = ctrIntArray.data() + bucket * targetClassesCount;
                        for (int classId = 0; classId < ctr.TargetBorderIdx + 1; ++classId) {
                            totalCount += ctrHistory[classId];
                        }
                        for (int classId = ctr.TargetBorderIdx + 1; classId < targetClassesCount; ++classId) {
                            goodCount += ctrHistory[classId];
                        }
                        totalCount += goodCount;
                        result[resultIdx] = ctr.Calc(goodCount, totalCount);
                    } else {
                        const int* ctrHistory = &ctrIntArray[bucket * 2];
                        result[resultIdx] = ctr.Calc(ctrHistory[1], ctrHistory[0] + ctrHistory[1]);
                    }
                }
            }
            resultIdx += 1;
        }
    }
}

static int GetHash(const std::string& catFeature, const std::unordered_map<std::string, int>& catFeatureHashes) {
    const auto keyValue = catFeatureHashes.find(catFeature);
    if (keyValue != catFeatureHashes.end()) {
        return keyValue->second;
    } else {
        return 0x7fFFffFF;
    }
}

/* Model applicator */
double ApplyCatboostModel(
    const std::vector<float>& floatFeatures,
    const std::vector<std::string>& catFeatures) {
    const struct CatboostModel& model = CatboostModelStatic;

    assert(floatFeatures.size() == model.FloatFeatureCount);
    assert(catFeatures.size() == model.CatFeatureCount);

    /* Binarize features */
    std::vector<unsigned char> binaryFeatures(model.BinaryFeatureCount, 0);
    unsigned int binFeatureIndex = 0;
    {
        /* Binarize float features */
        for (size_t i = 0; i < model.FloatFeatureBorders.size(); ++i) {
            if (!model.FloatFeatureBorders[i].empty()) {
                for (const float border : model.FloatFeatureBorders[i]) {
                    binaryFeatures[binFeatureIndex] += (unsigned char) (floatFeatures[i] > border);
                }
                ++binFeatureIndex;
            }
        }
    }

    std::vector<int> transposedHash(model.CatFeatureCount);
    for (size_t i = 0; i < model.CatFeatureCount; ++i) {
        transposedHash[i] = GetHash(catFeatures[i], CatFeatureHashes);
    }

    if (model.OneHotCatFeatureIndex.size() > 0) {
        /* Binarize one hot cat features */
        std::unordered_map<int, int> catFeaturePackedIndexes;
        for (unsigned int i = 0; i < model.CatFeatureCount; ++i) {
            catFeaturePackedIndexes[model.CatFeaturesIndex[i]] = i;
        };
        for (unsigned int i = 0; i < model.OneHotCatFeatureIndex.size(); ++i) {
            const auto catIdx = catFeaturePackedIndexes.at(model.OneHotCatFeatureIndex[i]);
            const auto hash = transposedHash[catIdx];
            if (!model.OneHotHashValues[i].empty()) {
                for (unsigned int borderIdx = 0; borderIdx < model.OneHotHashValues[i].size(); ++borderIdx) {
                    binaryFeatures[binFeatureIndex] |=
                        (unsigned char) (hash == model.OneHotHashValues[i][borderIdx]) * (borderIdx + 1);
                }
                ++binFeatureIndex;
            }
        }
    }

    if (model.modelCtrs.UsedModelCtrsCount > 0) {
        /* Binarize CTR cat features */
        std::vector<float> ctrs(model.modelCtrs.UsedModelCtrsCount);
        CalcCtrs(model.modelCtrs, binaryFeatures, transposedHash, ctrs);

        for (size_t i = 0; i < model.CtrFeatureBorders.size(); ++i) {
            for (const float border : model.CtrFeatureBorders[i]) {
                binaryFeatures[binFeatureIndex] += (unsigned char)(ctrs[i] > border);
            }
            ++binFeatureIndex;
        }
    }

    /* Extract and sum values from trees */
    double result = 0.0;
    const unsigned int* treeSplitsPtr = model.TreeSplits.data();
    const double* leafValuesPtr = model.LeafValues;
    size_t treePtr = 0;
    for (unsigned int treeId = 0; treeId < model.TreeCount; ++treeId) {
        const unsigned int currentTreeDepth = model.TreeDepth[treeId];
        unsigned int index = 0;
        for (unsigned int depth = 0; depth < currentTreeDepth; ++depth) {
            const unsigned char borderVal = model.TreeSplitIdxs[treePtr + depth];
            const unsigned int featureIndex = model.TreeSplitFeatureIndex[treePtr + depth];
            const unsigned char xorMask = model.TreeSplitXorMask[treePtr + depth];
            index |= ((binaryFeatures[featureIndex] ^ xorMask) >= borderVal) << depth;
        }
        result += leafValuesPtr[index];
        treeSplitsPtr += currentTreeDepth;
        leafValuesPtr += (1 << currentTreeDepth);
        treePtr += currentTreeDepth;
    }
    return model.Scale * result + model.Bias;
}
