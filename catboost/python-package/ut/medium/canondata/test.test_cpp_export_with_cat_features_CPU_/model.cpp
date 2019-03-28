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
        {19.5, 45.5, 57.5},
        {167392.5, 208500.5, 218145.5},
        {10.5, 13.5},
        {1087, 3280, 7493, 11356},
        {1738, 1881.5, 2189.5},
        {44.5, 49}
    };
    std::vector<unsigned int> TreeDepth = {4, 4, 4, 3, 3, 4, 3, 1, 6, 0, 0, 3, 1, 2, 4, 0, 6, 0, 0, 3};
    std::vector<unsigned int> TreeSplits = {8, 14, 11, 20, 42, 15, 18, 13, 12, 1, 19, 28, 7, 12, 35, 13, 21, 39, 3, 43, 25, 36, 30, 34, 9, 7, 41, 22, 0, 10, 27, 17, 37, 5, 26, 40, 33, 6, 9, 24, 16, 38, 21, 11, 2, 4, 16, 31, 23, 29, 32};
    std::vector<unsigned char> TreeSplitIdxs = {1, 3, 4, 4, 4, 1, 2, 2, 1, 2, 3, 2, 2, 1, 2, 2, 1, 1, 1, 5, 1, 3, 1, 1, 2, 2, 3, 2, 1, 3, 1, 1, 1, 3, 2, 2, 3, 1, 2, 1, 2, 2, 1, 4, 3, 2, 2, 1, 3, 3, 2};
    std::vector<unsigned short> TreeSplitFeatureIndex = {3, 4, 3, 6, 15, 5, 6, 4, 4, 0, 6, 10, 2, 4, 13, 4, 7, 15, 1, 15, 9, 13, 11, 13, 3, 2, 15, 7, 0, 3, 10, 6, 14, 1, 9, 15, 12, 2, 3, 8, 5, 14, 7, 3, 0, 1, 5, 12, 7, 10, 12};
    std::vector<unsigned char> TreeSplitXorMask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<unsigned int> CatFeaturesIndex = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<unsigned int> OneHotCatFeatureIndex = {};
    std::vector<std::vector<int>> OneHotHashValues = {
    };
    std::vector<std::vector<float>> CtrFeatureBorders = {
        {1.99999905f, 3.99999905f, 8.99999905f, 10.999999f},
        {3.99999905f, 5.99999905f, 12.999999f},
        {7.99999905f},
        {10.999999f, 12.999999f},
        {5.99999905f, 8.99999905f, 9.99999905f},
        {9.99999905f},
        {5.99999905f, 10.999999f, 12.999999f},
        {2.99999905f, 8.99999905f, 10.999999f},
        {4.99999905f, 7.99999905f},
        {2.99999905f, 3.99999905f, 6.99999905f, 7.99999905f, 13.999999f}
    };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[261] = {
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
        0, 0, 0.01553349164616098, 0.0153003443298752, 0.003182323338779773, 0, 0.01151251234732036, 0.01261814884776815
    };
    struct TCatboostCPPExportModelCtrs modelCtrs = {
        .UsedModelCtrsCount = 10,
        .CompressedModelCtrs = {
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 14216163332699387099ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387099ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387099ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = -0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 4230580741181273963ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 4230580741181273963ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = -0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 14216163332699387101ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 16890222057671696978ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Counter, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15}
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
                    4230580741181273963ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {1799168355831033313ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11936664559898134054ull, 6}, {14666845749088704071ull, 10}, {18429784838380727208ull, 7}, {17027374437435318793ull, 13}, {2862173265672040777ull, 0}, {16080065667299791243ull, 5}, {14677655266354382828ull, 12}, {12391839889973628461ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4082592020331177586ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {315676340386518075ull, 3}, {18446744073709551615ull, 0}, {10716245805238997245ull, 2}, {9313835404293588830ull, 1}, {17603450378469852574ull, 11}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 8}, {.Sum = 0, .Count = 4}, {.Sum = 1.4013e-44, .Count = 46}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 2.8026e-45, .Count = 6}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 4}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 7.00649e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 8, 0, 4, 10, 46, 0, 1, 1, 3, 2, 6, 0, 2, 1, 4, 0, 2, 0, 2, 2, 0, 5, 0, 1, 0, 0, 1}
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
