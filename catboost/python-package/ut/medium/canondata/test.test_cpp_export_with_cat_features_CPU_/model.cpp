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
        {20.5, 25, 29.5, 34.5, 35.5, 38.5, 45.5, 54.5, 57.5},
        {185890.5, 197408.5, 203488.5, 218145.5, 243690.5, 303732.5, 325462, 421435.5},
        {4.5, 5.5, 6.5, 8, 11.5, 13.5, 15.5},
        {1087, 3280, 5842, 7493, 17537.5},
        {1622.5, 1881.5, 1944.5, 2189.5, 2396},
        {11.5, 19, 27, 35.5, 36.5, 38.5, 42, 44.5, 46.5, 70}
    };
    std::vector<unsigned int> TreeDepth = {2, 6, 6, 6, 6, 6, 3, 6, 6, 6, 6, 6, 6, 2, 5, 6, 6, 0, 2, 1};
    std::vector<unsigned int> TreeSplits = {25, 32, 26, 5, 14, 27, 42, 34, 22, 25, 18, 11, 64, 36, 28, 40, 61, 12, 20, 30, 6, 38, 15, 21, 17, 32, 9, 34, 55, 19, 40, 28, 25, 22, 31, 56, 58, 32, 41, 35, 23, 37, 4, 60, 33, 1, 30, 41, 57, 7, 10, 19, 30, 2, 0, 59, 24, 18, 30, 64, 44, 3, 47, 51, 16, 25, 48, 46, 8, 13, 53, 63, 24, 21, 65, 50, 46, 52, 64, 39, 45, 49, 43, 29, 19, 54, 28, 8, 66, 3, 52, 62, 63};
    std::vector<unsigned char> TreeSplitIdxs = {2, 4, 3, 6, 6, 4, 9, 1, 6, 2, 2, 3, 10, 3, 5, 7, 7, 4, 4, 2, 7, 5, 7, 5, 1, 4, 1, 1, 1, 3, 7, 5, 2, 6, 3, 2, 4, 4, 8, 2, 7, 4, 5, 6, 5, 2, 2, 8, 3, 8, 2, 3, 2, 3, 1, 5, 1, 2, 2, 10, 255, 4, 1, 1, 8, 2, 2, 2, 9, 5, 3, 9, 1, 5, 1, 4, 2, 2, 10, 6, 1, 3, 10, 1, 3, 4, 5, 9, 1, 4, 2, 8, 9};
    std::vector<unsigned short> TreeSplitFeatureIndex = {3, 4, 3, 0, 1, 3, 5, 5, 2, 3, 2, 1, 10, 5, 3, 5, 10, 1, 2, 4, 0, 5, 1, 2, 2, 4, 1, 5, 10, 2, 5, 3, 3, 2, 4, 10, 10, 4, 5, 5, 2, 5, 0, 10, 4, 0, 4, 5, 10, 0, 1, 2, 4, 0, 0, 10, 3, 2, 4, 10, 6, 0, 8, 9, 1, 3, 8, 7, 0, 1, 9, 10, 3, 2, 11, 8, 7, 9, 10, 5, 7, 8, 5, 4, 2, 9, 3, 0, 12, 0, 9, 10, 10};
    std::vector<unsigned char> TreeSplitXorMask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<unsigned int> CatFeaturesIndex = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<unsigned int> OneHotCatFeatureIndex = {9};
    std::vector<std::vector<int>> OneHotHashValues = {
        {-2114564283}
    };
    std::vector<std::vector<float>> CtrFeatureBorders = {
        {3.99999905f, 13.999999f},
        {6.99999905f, 8.99999905f, 11.999999f, 12.999999f},
        {8.99999905f, 11.999999f, 13.999999f, 14.999999f},
        {2.99999905f, 3.99999905f, 4.99999905f, 5.99999905f, 6.99999905f, 7.99999905f, 8.99999905f, 9.99999905f, 12.999999f, 13.999999f},
        {9.99999905f},
        {8.99999905f}
    };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[887] = {
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
        0.01854745925478346, 0.007256784355642204
    };
    struct TCatboostCPPExportModelCtrs modelCtrs = {
        .UsedModelCtrsCount = 6,
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
