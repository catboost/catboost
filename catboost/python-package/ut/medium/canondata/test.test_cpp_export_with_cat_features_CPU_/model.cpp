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
    unsigned int BinaryFeatureCount = 20;
    unsigned int TreeCount = 20;
    std::vector<std::vector<float>> FloatFeatureBorders = {
        {18.5, 34.5, 68.5},
        {200721, 215061, 231641.5, 281044.5, 337225.5, 553548.5},
        {6.5, 9.5, 12.5, 13.5, 14.5},
        {1087, 3280, 5842, 11356, 17537.5},
        {808.5, 1862, 1944.5, 2396},
        {36.5, 42, 70}
    };
    std::vector<unsigned int> TreeDepth = {3, 6, 6, 6, 3, 3, 4, 0, 0, 0, 4, 2, 1, 5, 5, 6, 0, 3, 4, 5};
    std::vector<unsigned int> TreeSplits = {17, 20, 32, 34, 0, 5, 47, 33, 36, 45, 4, 19, 21, 14, 23, 41, 7, 48, 13, 6, 57, 50, 28, 15, 22, 2, 44, 17, 39, 42, 12, 18, 59, 30, 35, 1, 37, 55, 56, 25, 53, 26, 29, 11, 52, 22, 31, 46, 8, 10, 58, 9, 47, 18, 3, 48, 51, 38, 27, 57, 49, 24, 16, 43, 54, 40};
    std::vector<unsigned char> TreeSplitIdxs = {4, 2, 6, 1, 1, 3, 5, 7, 2, 3, 2, 1, 3, 1, 1, 2, 5, 1, 5, 4, 1, 3, 2, 2, 4, 3, 2, 4, 3, 3, 4, 5, 1, 4, 1, 2, 1, 1, 2, 3, 2, 255, 3, 3, 1, 4, 5, 4, 6, 2, 2, 1, 5, 5, 1, 1, 1, 2, 1, 1, 2, 2, 3, 1, 1, 1};
    std::vector<unsigned short> TreeSplitFeatureIndex = {3, 4, 7, 8, 0, 1, 12, 7, 9, 12, 1, 4, 4, 3, 5, 11, 1, 13, 2, 1, 18, 13, 7, 3, 4, 0, 12, 3, 10, 11, 2, 3, 19, 7, 9, 0, 10, 17, 17, 5, 15, 6, 7, 2, 15, 4, 7, 12, 1, 2, 18, 2, 12, 3, 1, 13, 14, 10, 7, 18, 13, 5, 3, 12, 16, 11};
    std::vector<unsigned char> TreeSplitXorMask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<unsigned int> CatFeaturesIndex = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<unsigned int> OneHotCatFeatureIndex = {9};
    std::vector<std::vector<int>> OneHotHashValues = {
        {-2114564283}
    };
    std::vector<std::vector<float>> CtrFeatureBorders = {
        {0.999998987f, 1.99999905f, 3.99999905f, 6.99999905f, 8.99999905f, 10.999999f, 12.999999f},
        {2.99999905f},
        {6.99999905f, 8.99999905f},
        {0.999998987f, 7.99999905f, 12.999999f},
        {6.99999905f, 8.99999905f, 12.999999f},
        {6.99999905f, 7.99999905f, 9.99999905f, 11.999999f, 13.999999f},
        {0.999998987f, 12.999999f, 13.999999f},
        {6.99999905f},
        {2.99999905f, 4.99999905f},
        {10.999999f},
        {11.999999f, 14.999999f},
        {0.999998987f, 1.99999905f},
        {12.999999f}
    };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[442] = {
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
        0.005796650788699248, 0.002692987287766075, 0, 0, 0, 0.005493297219148043, -0.001813419916642705, 0, 0.002869765075874072, 0, 0, 0, 0.008596719468130386, 0, 0, 0, 0, 0, 0, 0, 0.01723908915402649, 0.006092324227009916, 0, 0, 0, 0, 0, 0, 0.01549819505598307, 0.006372982395348424, -0.003046820126714731, -0.001364968436561024
    };
    struct TCatboostCPPExportModelCtrs modelCtrs = {
        .UsedModelCtrsCount = 13,
        .CompressedModelCtrs = {
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 14216163332699387099ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387099ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = -0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {4},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 16890222057671696979ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Counter, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15}
                }
            },
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
                    .transposedCatFeatureIndexes = {5},
                    .binarizedIndexes = {
                        {.BinIndex = 1, .CheckValueEqual = 0, .Value = 1}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 692698791827290762ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = -0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 14216163332699387072ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387072ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387072ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = -0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {10},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 14216163332699387074ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387074ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = -0, .Scale = 15}
                }
            }
        },
        .CtrData = {
            .LearnCtrs = {
                {
                    692698791827290762ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {3581428127016485793ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14455983217430950149ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13125720576600207402ull, 8}, {5967870314491345259ull, 6}, {9724886183021484844ull, 1}, {2436149079269713547ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1236773280081879954ull, 2}, {16151796118569799858ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8312525161425951098ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13605281311626526238ull, 9}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 18}, {.Sum = 0, .Count = 2}, {.Sum = 1.68156e-44, .Count = 8}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 6}, {.Sum = 1.4013e-44, .Count = 12}, {.Sum = 0, .Count = 19}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 3}},
                        .CtrTotal = {0, 18, 0, 2, 12, 8, 0, 7, 0, 2, 0, 6, 10, 12, 0, 19, 0, 2, 0, 3}
                    }
                },
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
                    14216163332699387074ull,
                    {
                        .IndexHashViewer = {{2136296385601851904ull, 0}, {7428730412605434673ull, 1}, {9959754109938180626ull, 3}, {14256903225472974739ull, 5}, {8056048104805248435ull, 2}, {18446744073709551615ull, 0}, {12130603730978457510ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10789443546307262781ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 2.8026e-44, .Count = 73}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {20, 73, 0, 2, 0, 2, 1, 0, 0, 1, 0, 1, 1, 0}
                    }
                },
                {
                    14216163332699387099ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {15379737126276794113ull, 5}, {18446744073709551615ull, 0}, {14256903225472974739ull, 2}, {18048946643763804916ull, 4}, {2051959227349154549ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7024059537692152076ull, 6}, {18446744073709551615ull, 0}, {15472181234288693070ull, 1}, {8864790892067322495ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 1.4013e-44, .Count = 58}, {.Sum = 1.4013e-45, .Count = 6}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 4.2039e-45, .Count = 6}, {.Sum = 0, .Count = 4}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 7.00649e-45, .Count = 0}},
                        .CtrTotal = {10, 58, 1, 6, 1, 5, 3, 6, 0, 4, 2, 0, 5, 0}
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
                    16890222057671696978ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13987540656699198946ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18089724839685297862ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10353740403438739754ull, 2}, {3922001124998993866ull, 0}, {13686716744772876732ull, 1}, {18293943161539901837ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 42,
                        .CtrMeanHistory = {{.Sum = 5.1848e-44, .Count = 4}, {.Sum = 5.88545e-44, .Count = 13}, {.Sum = 2.8026e-45, .Count = 3}},
                        .CtrTotal = {37, 4, 42, 13, 2, 3}
                    }
                },
                {
                    16890222057671696979ull,
                    {
                        .IndexHashViewer = {{7537614347373541888ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5903587924673389870ull, 1}, {18278593470046426063ull, 6}, {10490918088663114479ull, 8}, {18446744073709551615ull, 0}, {407784798908322194ull, 10}, {5726141494028968211ull, 3}, {1663272627194921140ull, 0}, {8118089682304925684ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15431483020081801594ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1403990565605003389ull, 2}, {3699047549849816830ull, 11}, {14914630290137473119ull, 7}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 28,
                        .CtrMeanHistory = {{.Sum = 4.2039e-45, .Count = 28}, {.Sum = 2.66247e-44, .Count = 4}, {.Sum = 2.8026e-44, .Count = 2}, {.Sum = 5.60519e-45, .Count = 10}, {.Sum = 4.2039e-45, .Count = 1}, {.Sum = 7.00649e-45, .Count = 2}},
                        .CtrTotal = {3, 28, 19, 4, 20, 2, 4, 10, 3, 1, 5, 2}
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
