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
    unsigned int BinaryFeatureCount = 19;
    unsigned int TreeCount = 20;
    std::vector<std::vector<float>> FloatFeatureBorders = {
        {27.5, 30.5, 35.5, 37.5, 41.5, 46.5, 50},
        {449128.5},
        {6.5, 10.5, 13.5, 14.5},
        {1087, 3280, 5842},
        {2189.5},
        {36.5, 46.5, 49}
    };
    std::vector<unsigned int> TreeDepth = {4, 6, 6, 4, 6, 6, 1, 4, 2, 2, 6, 3, 3, 6, 1, 4, 6, 6, 2, 6};
    std::vector<unsigned int> TreeSplits = {36, 9, 2, 13, 13, 15, 46, 37, 0, 55, 31, 5, 11, 28, 35, 52, 32, 10, 6, 12, 32, 43, 1, 30, 12, 24, 36, 42, 16, 4, 17, 14, 37, 37, 47, 51, 54, 18, 45, 41, 32, 37, 49, 27, 53, 22, 23, 36, 21, 14, 36, 19, 34, 37, 21, 8, 25, 40, 7, 13, 31, 17, 29, 25, 36, 24, 9, 39, 20, 44, 13, 54, 33, 50, 26, 3, 36, 48, 37, 9, 27, 38, 50, 14};
    std::vector<unsigned char> TreeSplitIdxs = {2, 2, 3, 2, 2, 1, 1, 3, 1, 1, 2, 6, 4, 1, 1, 2, 3, 3, 7, 1, 3, 1, 2, 1, 1, 2, 2, 5, 1, 5, 2, 3, 3, 3, 2, 1, 4, 3, 2, 4, 3, 3, 4, 5, 3, 1, 1, 2, 2, 3, 2, 1, 5, 3, 2, 1, 3, 3, 1, 2, 2, 2, 2, 3, 2, 2, 2, 2, 1, 1, 2, 4, 4, 5, 4, 4, 2, 3, 3, 2, 5, 1, 5, 3};
    std::vector<unsigned short> TreeSplitFeatureIndex = {12, 2, 0, 3, 3, 4, 16, 12, 0, 18, 11, 0, 2, 10, 12, 17, 11, 2, 0, 3, 11, 14, 0, 11, 3, 9, 12, 13, 5, 0, 5, 3, 12, 12, 16, 17, 17, 5, 15, 13, 11, 12, 16, 9, 17, 8, 9, 12, 7, 3, 12, 6, 11, 12, 7, 2, 9, 13, 1, 3, 11, 5, 10, 9, 12, 9, 2, 13, 7, 15, 3, 17, 11, 16, 9, 0, 12, 16, 12, 2, 9, 13, 16, 3};
    std::vector<unsigned char> TreeSplitXorMask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<unsigned int> CatFeaturesIndex = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<unsigned int> OneHotCatFeatureIndex = {};
    std::vector<std::vector<int>> OneHotHashValues = {
    };
    std::vector<std::vector<float>> CtrFeatureBorders = {
        {5.99999905f},
        {4.99999905f, 9.99999905f},
        {3.99999905f},
        {4.99999905f, 5.99999905f, 7.99999905f, 9.99999905f, 11.999999f},
        {11.999999f, 13.999999f},
        {7.99999905f, 10.999999f, 11.999999f, 12.999999f, 13.999999f},
        {4.99999905f, 12.999999f, 13.999999f},
        {2.99999905f, 6.99999905f, 8.99999905f, 11.999999f, 13.999999f},
        {6.99999905f},
        {7.99999905f, 11.999999f},
        {2.99999905f, 6.99999905f, 8.99999905f, 9.99999905f, 11.999999f},
        {0.999998987f, 1.99999905f, 2.99999905f, 11.999999f},
        {11.999999f}
    };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[672] = {
        0.005858654557193415, 0.004574257211962552, 0.005309405692456533, -0.004837340859872783, 0.005227722527957201, -0.003970818179870656, 0.004752475025415637, -0.01509900968300529, 0, 0, 0, -0.005866336542375311, 0, -0.009386138467800498, 0, -0.01173267308475062,
        0.003976368137671248, 0, 0, 0, 0.005169690515611968, 0, 0, 0, 0, 0, 0, 0, 0.001599356361663726, 0, 0, 0, 0.003984664483974143, 0, 0, 0, 0.005419846583099941, 0, 0, 0, -0.005830056486737188, 0, 0, 0, -0.007081696653487106, -0.01160948002011432, -0.005836555406691942, 0, 0.00159384274818326, 0, 0, 0, 0.00424109318094769, 0, 0, 0, 0, 0, 0, 0, 0.00261543976583078, 0, 0, 0, 0.00364650619645865, 0, 0, 0, 0.004637322716415355, 0, 0, 0, 0.003331221464643159, 0, 0, 0, -0.005018113607206865, -0.01155668299241305, -0.005753093972283941, 0,
        0, 0.001553806523438635, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.006622825197287259, 0.002495840994278184, 0, 0, 0, 0, 0, 0, 0, 0.001564019987817316, 0, 0, 0, 0, 0, 0, 0, 0.003109971329713985, 0, 0.003575501950221085, 0, 0, 0, 0, 0, 0.004354578631253992, 0, 0.004370292348974498, 0, 0, 0, 0, -0.0026997907651662, -0.001610809265053736, -0.00993352243076028, -0.009125699597417505, -0.009127941613620957, 0, 0, 0.00178454171129082, 0, 0.005675716584589659, 0, 0.003113901903582833, 0, 0, 0, 0,
        -0.001314115296816921, 0.005708350173674256, -0.01180611476895657, 0.001524711540623615, -0.002958000055485705, 0.00481546672024603, -0.009000767731207173, -0.00172470942308717, -0.009099374168133032, 0.001510626026285627, -0.005679585306482896, 0, -0.00900122635984436, -0.005623223630249949, 0, 0,
        0, 0, 0, 0, 0.001813243565869722, 0, 0, 0, 0, 0, 0, 0.003062290598790211, 0, 0, 0.001797766693368555, 0.002993807100222541, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.003263932616075313, 0, 0, 0, -0.002972336995564962, 0, -0.01029971256600132, 0, 0.001680431842611455, 0, 0.003301336564206218, 0.00535961715623982, 0, 0.001548292873630225, -0.008304023904761563, 0.004176342852088737, 0, 0, 0, 0, -0.005566858794480265, 0, -0.00562037567221386, 0, 0, 0, 0, 0, 0, 0, -0.01120266373937536, -0.003265402498099214,
        0.004336006214046193, 0.003393750108749793, 0.003996272329876063, 0, 0.004914886469317892, 0.001107146419915755, 0.003603696549093107, 0, 0.002907901054408412, 0.001785097724208549, 0, 0, 0.004527672869002853, -0.006106913324113612, 0.002336135045485468, 0, 0, 0, 0, 0, 0.002894380111316895, -0.01127053972653346, 0.001429492443630908, 0, 0, 0, 0, 0, 0.001472446974456976, -0.01249689281762213, 0.001506206437436821, 0, 0, 0, 0, 0, 0, -0.005533331452313915, 0, 0, 0, -0.005556558935777402, 0, 0, 0, -0.008862460636739157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.005465635786778092, 0, 0,
        0.005401593349285408, -0.007679690743606212,
        0.002792833662311172, -0.005653030436289475, 0.001390644181827061, -0.002984577198678855, 0.004039395310966793, 0, 0.004994939597927501, -0.006423468943758759, 0, 0, 0, 0, 0.003177100238787053, 0.003558554724185827, 0.004469520757264649, -0.009752822164446447,
        -0.002530485590771002, -0.01351910011420432, 0.003056214440059436, 0.003642448591003023,
        -0.006806829483478119, 0, 0.001740228435259057, 0.004938772870228158,
        0, 0, 0, 0, 0, 0, 0, 0, 0.001344546835970351, 0, 0, 0, 0, 0, 0, 0, 0.001345450442209422, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002640785659657609, 0.001644790254337651, 0, 0, 0, 0, 0, -0.005540602847473389, 0, 0, 0.001320831756837683, 0, 0, 0, 0.002616849431631456, 0, 0.001321456397331494, -0.01345872863079496, 0.001276525454501771, 0, 0.004080299079716508, 0, 0.00132822049341367, 0, 0.00134424102993566, -0.002870234694144457, 0, 0, 0, 0, 0.002101335483617512, 0.003645236127499295, 0.002965164785320852, -0.003668831988076188, 0.003287938875338565, 0, 0.004506626286708471, 0,
        0.003829448620325868, -0.0124685045693824, 0.004761423256267189, -0.002047594288467481, 0, -0.008213760651179987, 0, -0.01062445730121561,
        0, -0.01537372074208929, 0, -0.003003652287897843, 0.003357337104641654, 0, 0.004641369575984959, -0.002932082615810923,
        0.001281458672385922, 0, 0.001263638277770864, 0.00207128943693085, 0.001280561843193493, -0.008635111207937257, 0.001247737895270103, -0.00517103903698099, 0, 0, 0, 0, 0, -0.008211882455498608, 0.002573384760301832, -0.00119125409789617, 0.001257644582840557, 0, 0.001224444277046747, 0, 0, -0.005141713814492103, 0.002425547542760039, 0.001833336781286055, 0, 0, 0.002949969522237644, 0, 0.003179858669780385, -0.007957783117623799, 0.004303498042835681, -0.0007408703439873704, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001233276706769122, 0, 0, 0, 0, 0, 0, 0, 0.002359845587681255, 0,
        0.001126054595708068, -0.01362049945919299,
        -0.002689479429769963, 0.002532735168210232, -0.01186902368808777, 0.001184834016074296, 0, 0, 0, 0, -0.0003240632956534191, 0.001210606466921111, -0.009915567500146378, 0.00182972673626195, 0, 0.004173443410035452, 0, 0.002346731025387528,
        0, 0, 0, 0, 0, 0, 0, -0.007607222267036852, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002137158710794722, 0, -0.0004522725232643838, 0, 0, 0.001283615541638656, -0.009297833775687239, 0, 0, 0, 0, 0, 0, 0, -0.00507836644005561, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00114302143700068, -0.008088708521384576, 0, 0, 0, -0.004627404767562463, 0.001244406809941814, 0.002998899380353667, 0.001210938938648711, 0, 0, 0, 0.001243392862804709, -0.005551248224299402, 0.001900612378848822, 0, 0.004093589805625858, -0.0002845360182458464, 0.001952082332821792, 0, 0.003868877190494988, -0.008827014979948967,
        -0.005340921748907342, 0, 0.001780887621812904, 0, 0.001949544524710163, 0, 0, 0, 0.0005507943055779489, -0.008211569236557449, 0.003892441609908703, 0, 0.002616940524350051, 0, 0.001181945591890142, 0, 0, 0, 0, 0, 0.002941320583878851, 0, 0.001171354852605955, 0, 0, 0, 0, 0, 0.003720614228820196, 0, 0.003149112873081884, 0, 0, 0, 0.003022645710248891, 0, 0.001228876089799257, 0, 0.001157118099702172, 0, -0.00615928166050322, -0.004915184387581696, -0.01011145735862785, -0.004592699232581473, 0.001196018145427428, 0, 0.001900506018475205, -0.005102780302058693, 0, 0, 0, 0, 0.001929665051791603, 0, 0.001823626527294318, 0, 0, 0, 0, -0.004853482289971215, 0.003251135865941117, 0, 0.001759460647993311, 0,
        0.003790951867866544, 0.0001140763322472993, 0.004013600111137387, -0.006674506826732431,
        0.001880658569538963, 0, 0, 0, 0, 0, 0.001231083799353105, 0, 0.002210214707231873, 0.002985005046580272, 0.002280368798334714, -0.002828036013346791, 0.002369727748410294, 0, 0.001734396400361892, 0, 0, 0.001689083774685355, 0.00117606081028898, -0.002387409477489091, 0, 0, 0, 0, 0.002950446927528725, -0.0009208952588625675, 0.001757064939059413, -0.01186699589454388, 0.003587311570132973, 0, 0.003335170805833696, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.004767022373528468, 0, 0, 0, 0, 0, -0.004828261705417214, 0, -0.009663092788652133, 0, 0, 0, 0
    };
    double Scale = 1;
    double Bias = 0.7821782231;
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
                    {.BaseHash = 14216163332699387099ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 16890222057671696980ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Counter, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15}
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
                    {.BaseHash = 16890222057671696975ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Counter, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15}
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
                    16890222057671696980ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {15379737126276794113ull, 5}, {18446744073709551615ull, 0}, {14256903225472974739ull, 2}, {18048946643763804916ull, 4}, {2051959227349154549ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7024059537692152076ull, 6}, {18446744073709551615ull, 0}, {15472181234288693070ull, 1}, {8864790892067322495ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 68,
                        .CtrMeanHistory = {{.Sum = 9.52883e-44, .Count = 7}, {.Sum = 8.40779e-45, .Count = 9}, {.Sum = 5.60519e-45, .Count = 2}},
                        .CtrTotal = {68, 7, 6, 9, 4, 2, 5}
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
