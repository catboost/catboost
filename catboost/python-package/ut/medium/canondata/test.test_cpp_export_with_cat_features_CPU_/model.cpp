#include <string>

#ifdef GOOOGLE_CITY_HASH // Required revision https://github.com/google/cityhash/tree/00b9287e8c1255b5922ef90e304d5287361b2c2a or earlier
    #include "city.h"
#else
    #include <util/digest/city.h>
#endif

#include <cstdio>
#include <vector>
#include <unordered_map>
#include <cassert>

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
    unsigned int BinaryFeatureCount = 23;
    unsigned int TreeCount = 20;
    std::vector<std::vector<float>> FloatFeatureBorders = {
        {45.5},
        {121010, 168783.5, 198094.5, 200721, 209752.5, 332801, 350449},
        {4.5, 8, 9.5, 13.5},
        {1087, 3280, 7493, 11356, 17537.5},
        {1738, 1881.5, 2189.5},
        {17, 31.5, 35.5, 36.5, 49}
    };
    std::vector<unsigned int> TreeDepth = {3, 0, 1, 2, 2, 0, 6, 2, 5, 4, 6, 2, 0, 3, 2, 4, 0, 5, 6, 4};
    std::vector<unsigned int> TreeSplits = {18, 22, 19, 28, 51, 5, 8, 11, 25, 17, 21, 29, 42, 4, 31, 30, 20, 46, 40, 33, 16, 15, 38, 41, 22, 9, 10, 17, 48, 49, 8, 48, 37, 45, 33, 39, 0, 24, 23, 1, 34, 2, 32, 36, 43, 35, 7, 12, 3, 26, 27, 47, 6, 13, 50, 44, 14};
    std::vector<unsigned char> TreeSplitIdxs = {2, 3, 3, 1, 1, 5, 1, 4, 255, 1, 2, 1, 1, 4, 3, 2, 1, 1, 4, 2, 5, 4, 2, 1, 3, 2, 3, 1, 3, 1, 1, 3, 1, 2, 2, 3, 1, 5, 4, 1, 1, 2, 1, 1, 1, 1, 7, 1, 3, 1, 1, 2, 6, 2, 2, 1, 3};
    std::vector<unsigned short> TreeSplitFeatureIndex = {4, 5, 4, 9, 22, 1, 2, 2, 6, 4, 5, 10, 17, 1, 10, 10, 5, 20, 15, 11, 3, 3, 15, 16, 5, 2, 2, 4, 20, 21, 2, 20, 15, 19, 11, 15, 0, 5, 5, 1, 12, 1, 11, 14, 18, 13, 1, 3, 1, 7, 8, 20, 1, 3, 21, 19, 3};
    std::vector<unsigned char> TreeSplitXorMask = {0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<unsigned int> CatFeaturesIndex = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<unsigned int> OneHotCatFeatureIndex = {9};
    std::vector<std::vector<int>> OneHotHashValues = {
        {-1291328762}
    };
    std::vector<std::vector<float>> CtrFeatureBorders = {
        {1.999999f},
        {8.999999f},
        {12.999999f},
        {4.999999f, 5.999999f, 9.999999f},
        {6.999999f, 9.999999f},
        {5.999999f},
        {10.999999f},
        {11.999999f},
        {5.999999f, 9.999999f, 10.999999f, 13.999999f},
        {1.999999f},
        {9.999999f},
        {0.99999899f},
        {2.999999f, 3.999999f},
        {2.999999f, 9.999999f, 12.999999f},
        {5.999999f, 8.999999f},
        {6.999999f}
    };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[346] = {
        0.02538461481722502, 0, 0.02181818133050745, 0.00599999986588955, 0, 0, 0, 0,
        0.02215084809364413,
        0.02172708936280263, -0.0003297677133102539,
        0.02152793739154444, 0.01778877082128263, 0.01850400474626319, 0.02067978983786624,
        0.01567963725707506, 0.0220112634798453, 0, 0.005536589382590209,
        0.01974820773748437,
        0, 0, 0, 0, 0.00668422720013354, 0.006656183451483404, 0, -0.00121032712489911, 0.01630675349288336, 0.01040940023674122, 0, 0, 0.01853636369398814, 0.02130145929132601, 0.004606578054190896, -0.0006521362759071478, 0, 0, 0, 0, 0, -0.001305236211565434, 0, 0, 0, 0, 0, 0, 0, 0.009936563966338846, 0, 0, 0, 0, 0, 0, 0, 0.00668422720013354, 0, 0, 0.01739248942175277, 0.01489443623183825, 0, 0, 0.01961891635677588, 0.02056829642804693, 0, 0.006781180916326713, 0, 0, 0, 0, 0, -0.001674910548736256, 0, 0, 0, 0, 0, 0, 0, 0.01015745703339937, 0, 0,
        0.004397862157920608, 0, 0.0173169712303545, 0.01995834279965041,
        0, 0.006573278111265952, 0, 0.01056177845088946, 0, -0.000838967326307618, 0, -0.002210365106577664, 0, 0, 0.01252797537745427, 0.0236805208742473, 0, 0.008373865623709641, 0, 0.01177294973233899, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.001190292731481934,
        0.006172078091740506, 0, 0.006076093315588495, -0.001181365536195357, 0.009704954610537919, 0, 0.02094713146553188, 0, 0.01213569811646915, 0, 0.006267968516740492, -0.001668938697132198, 0.02077356302989403, 0, 0.01404276168665576, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01194789205007902, 0, 0, 0, 0, 0, 0, 0, 0.006188668017761182, 0, 0, 0, 0, 0, 0, 0, 0.006030522616740166, 0.005918989833108543, 0, 0.00956997428119615, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01731715780704228, 0.0200484977359294, 0, 0.02034553074602549, 0, 0.003998183659885355, 0, 0, 0.009791197088019349, 0.01081763758489084, 0, 0.004753520117275608, 0, 0, 0, 0.002113073546331682,
        0.02069773494014405, 0, 0.01995277383978061, 0.008226800289584998,
        0.01642145822000003,
        0, 0.01187160900013913, 0, 0.01916886602566633, 0, -0.0032139235512184, 0.003200190713244949, 0.01618175634200034,
        0.01857021431146683, 0.011534044748459, 0.007379608573513032, 0.003764586258426566,
        0, 0, 0.005451847189699661, 0.005352498027267782, 0.01219957601974083, 0.01240237014666204, 0.005981022937292964, 0.00993196343922742, 0, 0, 0.005587135362170781, 0.01389804493124376, 0, 0, 0.01764949412991807, 0.01233841871780235,
        0.01471105694954564,
        0, 0.002625700658516242, 0, 0.009724031707401531, -0.003376025787578505, 0.006818563531257438, 0, 0.005202021368305096, 0, 0, 0, 0.009424305756923211, 0, 0, 0.01111812876323236, 0.01885070155215754, 0, 0.005795377395515639, 0, 0.005044936656386489, 0, 0.008755366118293646, 0, 0, 0, 0, 0, 0.00509932973029755, 0, 0, 0, 0.01410926625216536,
        -0.001874681421018266, 0, 0, 0, 0, 0, 0.004810577060605464, 0, 0.009180949868100795, -0.001543919024798, 0, 0, 0.01823204901206538, 0.005108850007640364, 0.0166589057282016, 0, -0.002717896811328024, 0, -0.00206289377896663, -0.001865102449209581, 0, 0, 0, 0, 0, 0, -0.002108374658957168, 0, 0.009355673941159435, -0.003960394503023596, 0.007698807248080881, -0.002161145929500197, 0, 0, 0.005017273182540766, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01553680271461355, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.008283936590973836, 0,
        0, 0, 0.005173900140458494, 0, 0.01430852627013582, 0, 0.0143925650164066, -0.003323831052914757, 0, 0, 0, 0, 0, 0, 0, -0.00412862012170337
    };
    struct TCatboostCPPExportModelCtrs modelCtrs = {
        .UsedModelCtrsCount = 16,
        .CompressedModelCtrs = {
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 14216163332699387099ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387099ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = -0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {4},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 14216163332699387100ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387100ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387100ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = -0, .Scale = 15}
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
                        {.BinIndex = 3, .CheckValueEqual = 0, .Value = 4}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 12923321341810884916ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5},
                    .binarizedIndexes = {
                        {.BinIndex = 4, .CheckValueEqual = 0, .Value = 1},
                        {.BinIndex = 5, .CheckValueEqual = 0, .Value = 2},
                        {.BinIndex = 6, .CheckValueEqual = 1, .Value = 1}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 8875426491869161292ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Counter, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {6},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 14216163332699387102ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387102ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = -0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {7},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 16890222057671696976ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Counter, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 14216163332699387072ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 16890222057671696975ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Counter, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15}
                }
            }
        },
        .CtrData = {
            .LearnCtrs = {
                {
                    8875426491869161292ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12388642741946772418ull, 2}, {15554416449052025026ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10159611108399828747ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17281877160932429072ull, 7}, {4069608767902639184ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10879563727165159958ull, 4}, {18446744073709551615ull, 0}, {3893644761655887960ull, 1}, {18446744073709551615ull, 0}, {3836326012375860634ull, 17}, {1838769409751938715ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5822126489489576543ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8894837686808871138ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8800705802312104489ull, 10}, {7657545707537307113ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15610660761574625263ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10553223523899041848ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13625934721218336443ull, 5}, {8407093386812891388ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17609291800955974271ull, 8}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 31,
                        .CtrMeanHistory = {{.Sum = 1.26117e-44, .Count = 15}, {.Sum = 2.8026e-45, .Count = 31}, {.Sum = 7.00649e-45, .Count = 4}, {.Sum = 4.2039e-45, .Count = 1}, {.Sum = 1.26117e-44, .Count = 1}, {.Sum = 5.60519e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 2.8026e-45, .Count = 8}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {9, 15, 2, 31, 5, 4, 3, 1, 9, 1, 4, 2, 1, 2, 2, 8, 1, 1}
                    }
                },
                {
                    12923321341810884916ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {3581428127016485793ull, 3}, {1236773280081879954ull, 2}, {16151796118569799858ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13125720576600207402ull, 5}, {5967870314491345259ull, 4}, {9724886183021484844ull, 1}, {18446744073709551615ull, 0}, {13605281311626526238ull, 6}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 37}, {.Sum = 0, .Count = 4}, {.Sum = 2.66247e-44, .Count = 20}, {.Sum = 0, .Count = 13}, {.Sum = 4.2039e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 3}},
                        .CtrTotal = {0, 37, 0, 4, 19, 20, 0, 13, 3, 0, 0, 2, 0, 3}
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
                    14216163332699387100ull,
                    {
                        .IndexHashViewer = {{7537614347373541888ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5903587924673389870ull, 1}, {18278593470046426063ull, 6}, {10490918088663114479ull, 8}, {18446744073709551615ull, 0}, {407784798908322194ull, 10}, {5726141494028968211ull, 3}, {1663272627194921140ull, 0}, {8118089682304925684ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15431483020081801594ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1403990565605003389ull, 2}, {3699047549849816830ull, 11}, {14914630290137473119ull, 7}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 3}, {.Sum = 5.60519e-45, .Count = 24}, {.Sum = 4.2039e-45, .Count = 16}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 5.60519e-45, .Count = 16}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 4}, {.Sum = 9.80909e-45, .Count = 3}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 5}, {.Sum = 2.8026e-45, .Count = 0}},
                        .CtrTotal = {0, 3, 4, 24, 3, 16, 1, 3, 4, 16, 1, 1, 0, 4, 7, 3, 0, 3, 0, 1, 0, 5, 2, 0}
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
                    14216163332699387102ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {14452488454682494753ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1388452262538353895ull, 5}, {8940247467966214344ull, 9}, {4415016594903340137ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {41084306841859596ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8678739366408346384ull, 4}, {18446744073709551615ull, 0}, {4544226147037566482ull, 12}, {14256903225472974739ull, 6}, {16748601451484174196ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5913522704362245435ull, 0}, {1466902651052050075ull, 3}, {2942073219785550491ull, 8}, {15383677753867481021ull, 2}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 11}, {.Sum = 2.8026e-45, .Count = 9}, {.Sum = 2.8026e-45, .Count = 14}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 6}, {.Sum = 9.80909e-45, .Count = 6}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 8.40779e-45, .Count = 10}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 8}, {.Sum = 0, .Count = 3}, {.Sum = 1.4013e-45, .Count = 4}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {0, 11, 2, 9, 2, 14, 0, 2, 0, 6, 7, 6, 1, 5, 6, 10, 0, 1, 2, 8, 0, 3, 1, 4, 1, 0}
                    }
                },
                {
                    16890222057671696975ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8473802870189803490ull, 2}, {7071392469244395075ull, 1}, {18446744073709551615ull, 0}, {8806438445905145973ull, 3}, {619730330622847022ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 82,
                        .CtrMeanHistory = {{.Sum = 1.68156e-44, .Count = 6}, {.Sum = 1.14906e-43, .Count = 1}},
                        .CtrTotal = {12, 6, 82, 1}
                    }
                },
                {
                    16890222057671696976ull,
                    {
                        .IndexHashViewer = {{3607388709394294015ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18356215166324018775ull, 0}, {18365206492781874408ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14559146096844143499ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11416626865500250542ull, 3}, {5549384008678792175ull, 2}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 36,
                        .CtrMeanHistory = {{.Sum = 1.96182e-44, .Count = 22}, {.Sum = 3.08286e-44, .Count = 36}, {.Sum = 7.00649e-45, .Count = 2}},
                        .CtrTotal = {14, 22, 22, 36, 5, 2}
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
        transposedHash[i] = CityHash64(catFeatures[i].c_str(), catFeatures[i].size()) & 0xffffffff;
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
