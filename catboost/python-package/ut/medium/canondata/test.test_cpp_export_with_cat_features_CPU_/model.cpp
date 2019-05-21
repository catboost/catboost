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
    unsigned int BinaryFeatureCount = 16;
    unsigned int TreeCount = 20;
    std::vector<std::vector<float>> FloatFeatureBorders = {
        {17.5, 23.5, 25, 29.5, 45.5, 50, 56, 57.5, 68.5},
        {38811, 79129.5, 129787.5, 167944, 185890.5, 203488.5, 204331, 215061, 288417.5, 303732.5, 401155.5},
        {5.5, 11.5, 12.5, 13.5, 14.5},
        {1087, 3280, 5842, 7493, 11356, 17537.5},
        {1862, 1881.5, 2189.5},
        {17, 35.5, 36.5, 38.5, 46.5, 49, 70}
    };
    std::vector<unsigned int> TreeDepth = {3, 6, 4, 0, 6, 6, 4, 6, 6, 6, 2, 0, 6, 6, 6, 2, 5, 3, 4, 4};
    std::vector<unsigned int> TreeSplits = {31, 29, 25, 54, 31, 59, 28, 2, 30, 1, 57, 25, 22, 23, 50, 25, 24, 21, 47, 5, 12, 30, 18, 19, 31, 28, 41, 11, 33, 58, 51, 39, 8, 27, 13, 49, 10, 4, 20, 9, 48, 40, 14, 59, 15, 56, 55, 52, 28, 57, 23, 38, 58, 21, 39, 0, 32, 6, 35, 36, 33, 11, 7, 34, 53, 3, 46, 45, 59, 21, 60, 43, 42, 44, 46, 37, 47, 16, 42, 58, 8, 61, 26, 59, 17};
    std::vector<unsigned char> TreeSplitIdxs = {1, 5, 1, 2, 1, 5, 4, 3, 6, 2, 3, 1, 3, 4, 4, 1, 5, 2, 1, 6, 4, 6, 10, 11, 1, 4, 255, 3, 3, 4, 5, 6, 9, 3, 5, 3, 2, 5, 1, 1, 2, 7, 6, 5, 7, 2, 1, 6, 4, 3, 4, 5, 4, 2, 6, 1, 2, 7, 2, 3, 3, 3, 8, 1, 1, 4, 2, 1, 5, 2, 1, 1, 1, 1, 2, 4, 1, 8, 1, 4, 9, 1, 2, 5, 9};
    std::vector<unsigned short> TreeSplitFeatureIndex = {4, 3, 3, 12, 4, 13, 3, 0, 3, 0, 13, 3, 2, 2, 11, 3, 2, 2, 11, 0, 1, 3, 1, 1, 4, 3, 6, 1, 4, 13, 11, 5, 0, 3, 1, 11, 1, 0, 2, 1, 11, 5, 1, 13, 1, 13, 13, 11, 3, 13, 2, 5, 13, 2, 5, 0, 4, 0, 5, 5, 4, 1, 0, 5, 12, 0, 10, 10, 13, 2, 14, 8, 7, 9, 10, 5, 11, 1, 7, 13, 0, 15, 3, 13, 1};
    std::vector<unsigned char> TreeSplitXorMask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<unsigned int> CatFeaturesIndex = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<unsigned int> OneHotCatFeatureIndex = {9};
    std::vector<std::vector<int>> OneHotHashValues = {
        {-2114564283}
    };
    std::vector<std::vector<float>> CtrFeatureBorders = {
        {13.999999f},
        {12.999999f},
        {11.999999f},
        {6.99999905f, 13.999999f},
        {2.99999905f, 6.99999905f, 7.99999905f, 8.99999905f, 11.999999f, 12.999999f},
        {11.999999f, 12.999999f},
        {7.99999905f, 8.99999905f, 9.99999905f, 11.999999f, 14.999999f},
        {9.99999905f},
        {12.999999f}
    };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[698] = {
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
        0.008043113672150272, 0.008863707740067561, -0.002779022065048918, 0, 0.01732719939352943, 0.01271846396455622, -0.001123277465242882, -0.0008779333428867616, 0.007194827895389249, 0.003053037892089363, 0, 0, 0.01490701217814606, 0.007699480208911019, 0, 0
    };
    struct TCatboostCPPExportModelCtrs modelCtrs = {
        .UsedModelCtrsCount = 9,
        .CompressedModelCtrs = {
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 14216163332699387101ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387101ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387101ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 16890222057671696978ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Counter, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {7},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 14216163332699387103ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387103ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387103ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = -0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 14216163332699387072ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387072ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = -0, .Scale = 15}
                }
            }
        },
        .CtrData = {
            .LearnCtrs = {
                {
                    14216163332699387072ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8473802870189803490ull, 2}, {7071392469244395075ull, 1}, {18446744073709551615ull, 0}, {8806438445905145973ull, 3}, {619730330622847022ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 12}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 2.94273e-44, .Count = 61}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 12, 1, 5, 21, 61, 0, 1}
                    }
                },
                {
                    14216163332699387101ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13987540656699198946ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18089724839685297862ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10353740403438739754ull, 2}, {3922001124998993866ull, 0}, {13686716744772876732ull, 1}, {18293943161539901837ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 37}, {.Sum = 0, .Count = 4}, {.Sum = 3.08286e-44, .Count = 20}, {.Sum = 0, .Count = 13}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 3}},
                        .CtrTotal = {0, 37, 0, 4, 22, 20, 0, 13, 0, 2, 0, 3}
                    }
                },
                {
                    14216163332699387103ull,
                    {
                        .IndexHashViewer = {{3607388709394294015ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18356215166324018775ull, 0}, {18365206492781874408ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14559146096844143499ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11416626865500250542ull, 3}, {5549384008678792175ull, 2}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 14}, {.Sum = 0, .Count = 22}, {.Sum = 0, .Count = 22}, {.Sum = 2.66247e-44, .Count = 17}, {.Sum = 2.8026e-45, .Count = 3}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {0, 14, 0, 22, 0, 22, 19, 17, 2, 3, 1, 1}
                    }
                },
                {
                    16890222057671696978ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13987540656699198946ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18089724839685297862ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10353740403438739754ull, 2}, {3922001124998993866ull, 0}, {13686716744772876732ull, 1}, {18293943161539901837ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 42,
                        .CtrMeanHistory = {{.Sum = 5.1848e-44, .Count = 4}, {.Sum = 5.88545e-44, .Count = 13}, {.Sum = 2.8026e-45, .Count = 3}},
                        .CtrTotal = {37, 4, 42, 13, 2, 3}
                    }
                }
            }
        }
    };
} CatboostModelStatic;

static std::unordered_map<std::string, int> CatFeatureHashes = {
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
            for (const float border : model.FloatFeatureBorders[i]) {
                binaryFeatures[binFeatureIndex] += (unsigned char)(floatFeatures[i] > border);
            }
            ++binFeatureIndex;
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
            for (unsigned int borderIdx = 0; borderIdx < model.OneHotHashValues[i].size(); ++borderIdx) {
                binaryFeatures[binFeatureIndex] |= (unsigned char)(hash == model.OneHotHashValues[i][borderIdx]) * (borderIdx + 1);
            }
            ++binFeatureIndex;
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
    return result;
}
