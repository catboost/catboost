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
    unsigned int BinaryFeatureCount = 74;
    unsigned int TreeCount = 20;
    std::vector<std::vector<float>> FloatFeatureBorders = {
        {17.5, 36.5, 61.5, 68.5},
        {38811, 51773, 59723, 204331, 449128.5, 553548.5},
        {10.5, 12.5, 13.5, 14.5},
        {3280},
        {1738, 1881.5, 2189.5},
        {46.5}
    };
    std::vector<unsigned int> TreeDepth = {6, 6, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
    std::vector<unsigned int> TreeSplits = {3, 87, 11, 52, 53, 92, 57, 12, 69, 2, 72, 74, 7, 92, 88, 47, 84, 94, 44, 8, 48, 56, 13, 80, 3, 62, 5, 42, 4, 58, 61, 39, 76, 91, 10, 78, 6, 83, 65, 51, 19, 16, 22, 5, 64, 25, 16, 28, 24, 3, 17, 63, 27, 40, 49, 61, 20, 96, 50, 85, 38, 35, 36, 59, 82, 23, 37, 46, 29, 32, 2, 61, 81, 95, 86, 45, 79, 81, 68, 96, 24, 71, 75, 16, 66, 30, 70, 55, 0, 17, 67, 21, 41, 46, 5, 19, 14, 15, 9, 44, 54, 18, 26, 33, 34, 60, 31, 91, 1, 93, 77, 13, 73, 22, 18, 97, 89, 90, 43};
    std::vector<unsigned char> TreeSplitIdxs = {4, 1, 2, 1, 1, 1, 2, 3, 1, 3, 1, 1, 4, 1, 1, 1, 1, 1, 2, 5, 1, 1, 4, 1, 4, 1, 2, 3, 1, 1, 2, 2, 1, 1, 1, 1, 3, 1, 1, 1, 255, 2, 2, 2, 1, 1, 2, 1, 2, 4, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 1, 2, 2, 2, 255, 1, 1, 6, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 4, 1, 2, 1, 1, 1, 1, 1};
    std::vector<unsigned short> TreeSplitFeatureIndex = {0, 63, 2, 32, 33, 68, 36, 2, 46, 0, 49, 51, 1, 68, 64, 27, 60, 70, 25, 1, 28, 36, 2, 57, 0, 39, 1, 24, 1, 37, 38, 23, 53, 67, 2, 55, 1, 59, 42, 31, 6, 4, 8, 1, 41, 10, 4, 13, 9, 0, 4, 40, 12, 24, 29, 38, 7, 72, 30, 61, 23, 20, 21, 37, 58, 9, 22, 26, 14, 17, 0, 38, 58, 71, 62, 26, 56, 58, 45, 72, 9, 48, 52, 4, 43, 15, 47, 35, 0, 4, 44, 8, 24, 26, 1, 6, 3, 4, 1, 25, 34, 5, 11, 18, 19, 38, 16, 67, 0, 69, 54, 2, 50, 8, 5, 73, 65, 66, 25};
    std::vector<unsigned char> TreeSplitXorMask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<unsigned int> CatFeaturesIndex = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<unsigned int> OneHotCatFeatureIndex = {9};
    std::vector<std::vector<int>> OneHotHashValues = {
        {-2114564283}
    };
    std::vector<std::vector<float>> CtrFeatureBorders = {
        {0.26406249f},
        {0.25f, 0.5f},
        {0.029411767f, 0.053921569f},
        {0.27508751f},
        {0.60013998f},
        {0.2179407f},
        {0.51108223f},
        {0.51108223f},
        {0.50011671f},
        {0.95919573f},
        {0.21651274f},
        {0.95919573f},
        {0.51108223f},
        {0.37680939f},
        {0.80058664f},
        {0.52022797f},
        {0.45833334f, 0.859375f},
        {0.29427084f, 0.42708331f, 0.47135416f},
        {0.625f, 0.70000005f},
        {0.014705883f, 0.23529413f},
        {0.95540446f},
        {0.91749161f},
        {0.18972945f},
        {-0.0098029412f},
        {0.80071992f},
        {0.21261059f},
        {0.25318801f},
        {0.90990901f},
        {0.66891891f},
        {0.48015201f, 0.66427362f},
        {0.46428573f, 0.60714287f},
        {0.024509804f, 0.38725489f},
        {0.98737419f},
        {0.97384858f},
        {0.40192059f},
        {0.77178001f},
        {0.68919784f},
        {0.39211765f},
        {0.17802803f},
        {1.0008999f},
        {0.32109454f},
        {-0.0098029412f},
        {0.80071992f},
        {0.068620592f},
        {0.80071992f},
        {0.82569247f},
        {1.0008999f},
        {0.33435699f},
        {0.97146165f},
        {0.98618072f},
        {0.375f},
        {0.014705883f, 0.14215687f},
        {1.0008999f},
        {0.9383437f},
        {0.7045455f},
        {0.73214281f},
        {0.65426791f},
        {0.81010002f},
        {0.50025004f},
        {0.52864581f},
        {0.6875f},
        {0.46078432f},
        {0.94023925f},
        {0.95540446f},
        {0.055147059f},
        {0.6875f},
        {0.46568626f}
    };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[1248] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007499999832361937, 0, 0, 0, 0, 0, 0, 0, 0.006000000052154064, 0, 0.02400000020861626, 0.007499999832361937, 0.007499999832361937, 0, 0.01200000010430813, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007499999832361937, 0, 0, 0, 0, 0, 0, 0, 0.0169565211981535, 0, 0.02571428567171097, 0, 0.003000000026077032, 0, 0.02285714261233807, 0,
        0.01847253181040287, 0.02758516184985638, 0.002167582279071212, 0.01473392825573683, 0, 0, 0, 0, -0.0001017391259665601, 0.01181785762310028, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0.01171976048499346, 0.005697329062968493, 0.008425761014223099, 0.01433677785098553, 0.02086312137544155, 0.0199800506234169, 0.02266760170459747, 0.02407447062432766, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002714372472837567, 0.01595684327185154, 0, 0, 0, 0, 0, 0, 0.01601459085941315, 0.0237090215086937, 0, 0.0158926397562027, 0, 0, 0, 0, -0.0001019500705297105, 0, 0, 0, 0, 0, 0, 0, 0, 0.005765947047621012, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01656651869416237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.006752429530024529, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005176381673663855, 0, 0, 0, -0.0004485874378588051, 0, 0, 0, 0.02199704386293888, 0.005424461793154478, 0, 0,
        0.006587451323866844, 0, 0, 0, 0, 0, 0, 0, 0.02450916729867458, 0.01598926819860935, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02330418303608894, 0.003080242080613971, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0.01144003495573997, 0, 0, 0.00825794879347086, 0.02251898869872093, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01130221504718065, 0, 0, 0, 0.02336438186466694, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004327624104917049, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0009059817530214787, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007223065011203289, 0, 0, 0, 0, 0, 0, -0.000899186881724745, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.001266310224309564, 0, 0, 0.004663164261728525, 0, 0.02390770800411701, 0, 0, 0, 0.01376825850456953, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004262709058821201,
        0, 0.005097512155771255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004652826115489006, 0.0203520804643631, 0.006441781762987375, -0.0004930952563881874, -0.0008924429421313107, -0.0002275376027682796, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0002729316474869847, -0.0005511741037480533, 0, -0.001586188096553087, 0, 0, 0.009997958317399025, 0.02263787016272545, 0, 0.002931951079517603, -0.0002258310705656186, 0.01264820899814367,
        -0.0002708846295718104, 0, 0.004903013817965984, 0, -0.0004062594089191407, 0, 0.01041687838733196, 0, 0.00460119592025876, 0, -0.002038162900134921, -0.0008738531614653766, 0, 0, 0.006326347123831511, 0.006288318429142237, 0, 0, 0, 0, 0, 0, 0, 0, 0.01234116964042187, 0.005861080251634121, 0.005976180545985699, 0.01634914427995682, 0, 0.01638057269155979, 0, 0.02053526230156422, 0, 0, -0.0006894372054375708, 0, -0.001100746681913733, 0, 0.01786068081855774, 0, -0.0006467309431172907, 0, -0.0009685574914328754, 0, -0.0002241373440483585, 0, 0, 0.006267681252211332, 0, 0, 0, 0, 0, 0, -0.0005842586397193372, 0, 0.003975818399339914, 0.005853517912328243, 0, 0.01158045418560505, 0, 0, 0, 0.01816232316195965,
        0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0008687424706295133, 0, 0, 0, -0.0008672992698848248, 0, 0.006264025811105967, 0, 0, 0, 0.007347192615270615, 0, 0, 0, 0, -0.000581549305934459, -0.0006842664442956448, 0, 0.01822366379201412, 0, 0, -0.001395853934809566, 0.0186829250305891, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0005798767087981105, 0, 0, 0, 0, 0, 0, 0, 0.007246622815728188, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0.006581226829439402, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00109198538120836, 0, 0.003929005935788155, 0, 0, 0.006169311702251434, 0.009158240631222725, 0, 0, 0.01284539978951216, 0, 0.005574367009103298, 0, 0.02088440023362637, 0.005921438336372375, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.001200675731524825, 0, -0.0008630764205008745, 0, 0, 0, 0, 0, 0.004742668475955725, 0, 0.009196614846587181, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007538412697613239, 0, 0, 0, 0, 0, 0, 0.003559011034667492, 0.01925214752554893,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00823186244815588, 0, -0.002024621469900012, 0, 0, 0, 0.007248329930007458, 0, 0, 0, 0.004668214824050665, 0, 0, 0, 0.01198064535856247, -0.001166830887086689, 0.0162948053330183, 0.004230910912156105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01953712664544582, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005415867082774639, 0, -0.001351873273961246, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0003595628077164292, 0, 0.005832122638821602, -0.0004880440246779472, -0.001394068705849349, 0, 0, 0, -0.0008550564525648952, 0, 0.01903237029910088, 0, 0.009482814930379391, 0, 0, 0, 0, 0, 0.005301141645759344, 0, 0, 0, 0, 0, 0, 0, 0.01694683730602264, -0.0009529920644126832, 0.01124119199812412, 0,
        0.008881011046469212, 0.004034095909446478, -0.0005869531887583435, 0, -0.001084911287762225, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01546690054237843, 0.01814764179289341, -0.003182707587257028, 0, 0.00406634621322155, 0.003149080090224743, 0, 0, 0.004973651375621557, 0.004905948415398598, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004733890295028687, 0.004846477881073952, 0, 0, 0, 0, 0, -0.0008636185666546226, 0.00578766968101263, 0, 0, 0, 0, 0, -0.001241610967554152, 0.005106562748551369, 0.01692573539912701, 0.002728525316342711, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0.005751724354922771, 0, -0.0009804080473259091, 0, 0, 0, 0, 0, 0.01701867580413818, 0.004851416684687138, 0.01527677476406097, 0, 0, 0, 0, 0, 0, 0, -0.0015314647462219, 0, 0, 0, 0, 0, 0, 0, 0.006127183325588703, 0, 0, 0, 0, 0, 0, 0, 0.006999357137829065, 0, 0, 0, 0, 0, 0, 0, -0.002223898656666279, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005480255465954542, -0.0004544198454823345, -0.001068010577000678, 0, 0, 0, 0, -0.0006257341592572629, 0.00445112120360136, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005006264429539442, 0, 0, 0, 0.009330349043011665, 0, 0, 0, -0.001663518021814525, 0, 0, 0, 0.009476527571678162, -0.001506267930381, -0.0002902766864281148, 0, 0.009738394990563393, 0, 0, -0.0009527976508252323, 0.01667401939630508, 0, 0.01057973131537437
    };
    struct TCatboostCPPExportModelCtrs modelCtrs = {
        .UsedModelCtrsCount = 67,
        .CompressedModelCtrs = {
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791580653471478ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 768791580653471478ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 8405694746487331134ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3},
                    .binarizedIndexes = {
                        {.BinIndex = 1, .CheckValueEqual = 0, .Value = 2}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 3001583246656978020ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3},
                    .binarizedIndexes = {
                        {.BinIndex = 5, .CheckValueEqual = 0, .Value = 1}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 4544184825393173621ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 4, 5},
                    .binarizedIndexes = {
                        {.BinIndex = 0, .CheckValueEqual = 0, .Value = 4},
                        {.BinIndex = 4, .CheckValueEqual = 0, .Value = 3}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 4414881145659723684ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 4, 5},
                    .binarizedIndexes = {
                        {.BinIndex = 1, .CheckValueEqual = 0, .Value = 2}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 3001583246125931243ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 4, 7, 8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952491747546147ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 5},
                    .binarizedIndexes = {
                        {.BinIndex = 4, .CheckValueEqual = 0, .Value = 2}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 2790902285321313792ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 5},
                    .binarizedIndexes = {
                        {.BinIndex = 5, .CheckValueEqual = 0, .Value = 1}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 13902559248212744134ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 5, 8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791582259504189ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 6},
                    .binarizedIndexes = {
                        {.BinIndex = 5, .CheckValueEqual = 0, .Value = 1}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 13902559248212744135ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 6, 7},
                    .binarizedIndexes = {
                        {.BinIndex = 5, .CheckValueEqual = 0, .Value = 1}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 4544184825161771334ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 7},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493224740167ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 17677952493224740167ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493224740166ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {4},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791580653471473ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 768791580653471473ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 768791580653471473ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 8405694746487331129ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {4},
                    .binarizedIndexes = {
                        {.BinIndex = 1, .CheckValueEqual = 0, .Value = 4}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 9867321491374199501ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {4, 5},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493261641996ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 17677952493261641996ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 10041049327410906820ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {4, 6},
                    .binarizedIndexes = {
                        {.BinIndex = 1, .CheckValueEqual = 0, .Value = 3}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 5840538188647484189ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {4, 7},
                    .binarizedIndexes = {
                        {.BinIndex = 0, .CheckValueEqual = 0, .Value = 4}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 5819498284355557857ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {4, 7, 8},
                    .binarizedIndexes = {
                        {.BinIndex = 0, .CheckValueEqual = 0, .Value = 4}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 5819498284603408945ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {4, 10},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493261641995ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791580653471472ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 768791580653471472ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 768791580653471472ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 8405694746487331128ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5},
                    .binarizedIndexes = {
                        {.BinIndex = 0, .CheckValueEqual = 0, .Value = 4}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 12627245789391619615ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5},
                    .binarizedIndexes = {
                        {.BinIndex = 0, .CheckValueEqual = 0, .Value = 4},
                        {.BinIndex = 4, .CheckValueEqual = 0, .Value = 3}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 4414881145133934893ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5},
                    .binarizedIndexes = {
                        {.BinIndex = 1, .CheckValueEqual = 0, .Value = 2}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 6317293569456956330ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5},
                    .binarizedIndexes = {
                        {.BinIndex = 1, .CheckValueEqual = 0, .Value = 3}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 12606205885276083426ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5},
                    .binarizedIndexes = {
                        {.BinIndex = 4, .CheckValueEqual = 0, .Value = 2}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 15655841788288703925ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5},
                    .binarizedIndexes = {
                        {.BinIndex = 4, .CheckValueEqual = 0, .Value = 3}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 8628341152511840406ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5, 6},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493260528854ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 17677952493260528854ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5, 6},
                    .binarizedIndexes = {
                        {.BinIndex = 4, .CheckValueEqual = 0, .Value = 2}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 2790902285205833619ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5, 6, 7},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 8405694746995314031ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5, 6, 8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791582220620454ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5, 6, 8},
                    .binarizedIndexes = {
                        {.BinIndex = 0, .CheckValueEqual = 0, .Value = 2}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 3863811882172310855ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5, 6, 8, 10},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493297533872ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 17677952493297533872ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5, 8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493260528848ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5, 8},
                    .binarizedIndexes = {
                        {.BinIndex = 0, .CheckValueEqual = 0, .Value = 2}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 13000966989535245561ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5, 10},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493260528850ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 17677952493260528850ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {6},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791580653471475ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 8405694746487331131ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {6},
                    .binarizedIndexes = {
                        {.BinIndex = 1, .CheckValueEqual = 0, .Value = 3}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 12606205885276083425ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {6, 8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493263343771ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {7},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791580653471474ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 768791580653471474ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {7},
                    .binarizedIndexes = {
                        {.BinIndex = 0, .CheckValueEqual = 0, .Value = 4}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 12627245789391619613ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {7},
                    .binarizedIndexes = {
                        {.BinIndex = 1, .CheckValueEqual = 0, .Value = 4}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 9867321491374199502ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {7},
                    .binarizedIndexes = {
                        {.BinIndex = 5, .CheckValueEqual = 0, .Value = 1}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 4544184825393173617ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791580653471469ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 768791580653471469ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 8405694746487331109ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {8},
                    .binarizedIndexes = {
                        {.BinIndex = 0, .CheckValueEqual = 0, .Value = 2}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 5445777084271881924ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {8, 10},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493265578087ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {10},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791580653471471ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 768791580653471471ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 8405694746487331111ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            }
        },
        .CtrData = {
            .LearnCtrs = {
                {
                    13000966989535245561ull,
                    {
                        .IndexHashViewer = {{1757986209816306368ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14449357781033878404ull, 5}, {17335471473308721348ull, 3}, {15684611358642908806ull, 14}, {18446744073709551615ull, 0}, {11580098970816741768ull, 2}, {80059902472028169ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15655322029177927125ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6489083244225771673ull, 13}, {12786063218960790489ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3619824434008635037ull, 1}, {2160949785446258526ull, 8}, {1968964319342822495ull, 9}, {4408800825433526368ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1539542015216389732ull, 0}, {3160296822215680932ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2730874518089248169ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6849001936407276463ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15200819853968089276ull, 6}, {6270639049625855037ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 23}, {.Sum = 2.66247e-44, .Count = 8}, {.Sum = 0, .Count = 6}, {.Sum = 2.8026e-45, .Count = 9}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 23, 19, 8, 0, 6, 2, 9, 0, 5, 0, 2, 0, 2, 0, 5, 0, 1, 0, 1, 0, 5, 0, 1, 0, 2, 0, 2, 0, 1, 0, 4, 1, 0, 0, 1, 0, 1}
                    }
                },
                {
                    6317293569456956330ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {3581428127016485793ull, 7}, {1236773280081879954ull, 6}, {17856817611009672707ull, 3}, {18446744073709551615ull, 0}, {14455983217430950149ull, 4}, {18446744073709551615ull, 0}, {18336378346035991543ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8312525161425951098ull, 2}, {5967870314491345259ull, 1}, {2436149079269713547ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 5.1848e-44, .Count = 41}, {.Sum = 1.68156e-44, .Count = 2}, {.Sum = 5.60519e-45, .Count = 3}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {37, 41, 12, 2, 4, 3, 1, 1}
                    }
                },
                {
                    17677952493224740166ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {1799168355831033313ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11936664559898134054ull, 11}, {14666845749088704071ull, 7}, {18429784838380727208ull, 1}, {17027374437435318793ull, 13}, {2862173265672040777ull, 3}, {16080065667299791243ull, 0}, {14677655266354382828ull, 12}, {12391839889973628461ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4082592020331177586ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {315676340386518075ull, 5}, {18446744073709551615ull, 0}, {10716245805238997245ull, 2}, {9313835404293588830ull, 9}, {17603450378469852574ull, 6}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 2.8026e-45, .Count = 6}, {.Sum = 1.4013e-45, .Count = 4}, {.Sum = 1.4013e-44, .Count = 46}, {.Sum = 0, .Count = 8}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 7.00649e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {2, 6, 1, 4, 10, 46, 0, 8, 1, 3, 0, 1, 5, 0, 2, 0, 0, 2, 0, 4, 0, 2, 0, 2, 1, 0, 0, 1}
                    }
                },
                {
                    17677952493224740167ull,
                    {
                        .IndexHashViewer = {{7515733889724454912ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2160905354121516547ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13659069800549444297ull, 3}, {7791826943727985930ull, 2}, {7884511582485373322ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18022007786552474063ull, 19}, {18446744073709551615ull, 0}, {6068383991325515601ull, 25}, {7524725216182310545ull, 24}, {17609669744399151123ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11681580651965248598ull, 15}, {576145588900686679ull, 22}, {13155646805788779928ull, 0}, {18446744073709551615ull, 0}, {5849831644443487770ull, 5}, {3372332782322797723ull, 17}, {18446744073709551615ull, 0}, {9865453060805390877ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9800431588293194596ull, 10}, {9048109927352371876ull, 11}, {16801589031893337254ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2099530300070748010ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4741992351141480365ull, 21}, {17321493568029573614ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2151914027663660914ull, 6}, {9012245698387122739ull, 20}, {3718664820244579636ull, 23}, {2925864759981622644ull, 1}, {15505365976869715893ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 0, .Count = 13}, {.Sum = 1.4013e-44, .Count = 15}, {.Sum = 0, .Count = 17}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 9}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 5.60519e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 4, 1, 2, 0, 13, 10, 15, 0, 17, 0, 1, 0, 9, 0, 2, 0, 1, 4, 0, 1, 0, 1, 0, 0, 2, 0, 2, 0, 3, 0, 1, 0, 1, 0, 2, 1, 0, 0, 1, 0, 1, 1, 0, 2, 0, 0, 1, 1, 0, 0, 1}
                    }
                },
                {
                    3001583246125931243ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3628762670213083650ull, 21}, {4152033245790959619ull, 48}, {18446744073709551615ull, 0}, {17407091705877351685ull, 20}, {6882106995078381574ull, 36}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12587226841112876431ull, 1}, {18317959509575665424ull, 25}, {742970608404689295ull, 15}, {16344700011017827602ull, 32}, {2035992302241612690ull, 40}, {18446744073709551615ull, 0}, {12338705865783753109ull, 23}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2314852681173712664ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16662434637249573787ull, 51}, {12814906903325799324ull, 38}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2339754841350749476ull, 39}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11760162732384441256ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10190834038247912619ull, 29}, {18446744073709551615ull, 0}, {808135643011091501ull, 37}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15248885571486568624ull, 8}, {18446744073709551615ull, 0}, {16951015140729532594ull, 10}, {4483191194562101811ull, 50}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12966982428170399929ull, 47}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7619784579124933820ull, 28}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14111457067786581057ull, 18}, {6827846041496707650ull, 30}, {14168626759096197955ull, 24}, {1232811508289654594ull, 34}, {16285601543817880901ull, 19}, {15931517322100805958ull, 46}, {8622225544353820353ull, 0}, {6088165380785275845ull, 14}, {12273342661996998085ull, 49}, {12095984437040160455ull, 41}, {18446744073709551615ull, 0}, {12637007560756539980ull, 11}, {8792735915257126348ull, 35}, {14339137129999503950ull, 3}, {18395773067821135182ull, 33}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16513281606030803794ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6497871174082847701ull, 31}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6558771518591634648ull, 42}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14981662407691145819ull, 9}, {18446744073709551615ull, 0}, {4209995588326251229ull, 27}, {18446744073709551615ull, 0}, {8877606263321985375ull, 52}, {18446744073709551615ull, 0}, {10579735832564949345ull, 43}, {18446744073709551615ull, 0}, {10636905523874566243ull, 4}, {8359469528752380003ull, 16}, {8842526021017453540ull, 2}, {7781340352808279782ull, 44}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14110664416560906345ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9105286325534908268ull, 5}, {18446744073709551615ull, 0}, {10807415894777872238ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9608040095590038645ull, 45}, {3924353183578036726ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15648596808374320252ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 3}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-45, .Count = 6}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 8}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 7}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 4.2039e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 3, 1, 0, 0, 2, 2, 6, 0, 7, 0, 8, 0, 5, 0, 1, 0, 1, 0, 4, 0, 1, 1, 7, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3, 0, 2, 0, 1, 0, 1, 2, 1, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 3, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1}
                    }
                },
                {
                    5840538188647484189ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14437050659463898499ull, 15}, {13712861078413872003ull, 27}, {18446744073709551615ull, 0}, {10471866573136752518ull, 39}, {3339193297886510343ull, 33}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6522474812938725258ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2143292629466310926ull, 41}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7089305956521872786ull, 40}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6692985183842031253ull, 38}, {18446744073709551615ull, 0}, {6568726319216336023ull, 35}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14184963790483298970ull, 17}, {8066257757148410395ull, 12}, {17298463301634926620ull, 2}, {5557686758182214811ull, 50}, {6932391217975877918ull, 5}, {151985887108509214ull, 25}, {8634520787218841888ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17067110762478402855ull, 34}, {18446744073709551615ull, 0}, {322496258011815209ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {905552284316133676ull, 32}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1455867562654053300ull, 21}, {18446744073709551615ull, 0}, {9563528327934404534ull, 4}, {15234196598318321335ull, 47}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11590587107156578237ull, 13}, {18446744073709551615ull, 0}, {8031909129594746559ull, 16}, {6922172069111294656ull, 48}, {9734038698837710529ull, 30}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12449993409784492100ull, 20}, {18446744073709551615ull, 0}, {14152122979027456070ull, 42}, {8600131001622206919ull, 45}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8140346942904631631ull, 24}, {12703712228892337104ull, 51}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9181895285552204755ull, 29}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8504011841664173526ull, 9}, {18446744073709551615ull, 0}, {10206141410907137496ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9505565985399112924ull, 23}, {2305054716417383901ull, 19}, {9352405656455510750ull, 6}, {15202963607546217823ull, 31}, {7650276087212546780ull, 44}, {13923650265588858465ull, 46}, {13307679510447017953ull, 49}, {12613343166795003362ull, 8}, {5168754957168326627ull, 1}, {1511139700538854501ull, 36}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13671344409206559848ull, 10}, {5002941664428245224ull, 43}, {15373473978449523818ull, 28}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3661062170767697137ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17506660474440175860ull, 37}, {15791627448755489013ull, 52}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2129442972428039162ull, 14}, {18446744073709551615ull, 0}, {3831572541671003132ull, 22}, {18446744073709551615ull, 0}, {8194820753884735230ull, 26}, {6592600901344044030ull, 53}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 2.8026e-45, .Count = 5}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 4}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 5.60519e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {2, 5, 2, 0, 2, 3, 0, 2, 0, 2, 0, 4, 1, 2, 0, 2, 0, 1, 0, 1, 1, 4, 1, 1, 0, 1, 0, 1, 2, 2, 4, 0, 1, 3, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 5, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 5, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1}
                    }
                },
                {
                    3001583246656978020ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12337359831519453058ull, 0}, {18446744073709551615ull, 0}, {6973539969458659060ull, 2}, {13860744542689514389ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16503206593760246744ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2242442935049193755ull, 7}, {18446744073709551615ull, 0}, {8193958724117795869ull, 6}, {10924139913308365886ull, 5}, {14687079002600389023ull, 1}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 4.2039e-45, .Count = 6}, {.Sum = 1.4013e-45, .Count = 6}, {.Sum = 1.26117e-44, .Count = 57}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 7.00649e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {3, 6, 1, 6, 9, 57, 1, 5, 5, 0, 2, 0, 0, 4, 1, 1}
                    }
                },
                {
                    2790902285205833619ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {2188507106676080001ull, 5}, {637210402677728642ull, 19}, {15993786133470765187ull, 11}, {9069587651555262340ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5055294682867474183ull, 24}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12323226651178604938ull, 22}, {8215851897103635594ull, 37}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14236920219097648662ull, 9}, {17912585353630038166ull, 27}, {18446744073709551615ull, 0}, {1109313114747155609ull, 32}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8703931276011015453ull, 28}, {18446744073709551615ull, 0}, {255577360295528863ull, 29}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3288025018294948642ull, 4}, {4141994062948909859ull, 35}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7706109298484694183ull, 18}, {2695018782319127976ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6276645682289541931ull, 10}, {8021551920895572396ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2327253922091514671ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11721681120454478260ull, 33}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2894085065674662199ull, 26}, {18446744073709551615ull, 0}, {3760127730364375609ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13181488319220572861ull, 13}, {9803449187629884094ull, 34}, {2906391975912748863ull, 6}, {18446744073709551615ull, 0}, {5556431424490156097ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6123262568073303625ull, 39}, {3404434568201661641ull, 38}, {8927460297906761931ull, 23}, {7497967027866966732ull, 36}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10275551899699311061ull, 20}, {16042900961391600982ull, 21}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15849784945427779545ull, 7}, {18446744073709551615ull, 0}, {5368307437087193947ull, 14}, {18446744073709551615ull, 0}, {15832302934837792861ull, 30}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17051461319339267937ull, 25}, {9516124139116033121ull, 40}, {1848716790044246113ull, 41}, {17984436564768411617ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9672412035561384938ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {246971503299269366ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {813802646882416894ull, 31}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 5}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-45, .Count = 7}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 8.40779e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 7.00649e-45, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 4}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 5, 1, 0, 0, 4, 0, 2, 0, 5, 0, 7, 0, 3, 0, 1, 0, 2, 0, 2, 2, 7, 0, 1, 0, 1, 6, 1, 1, 0, 0, 2, 0, 2, 0, 1, 0, 2, 1, 2, 0, 1, 5, 1, 0, 3, 0, 4, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 2, 4, 0, 3, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}
                    }
                },
                {
                    4544184825393173617ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5844492600899280932ull, 0}, {18446744073709551615ull, 0}, {1034166431492604838ull, 2}, {18446744073709551615ull, 0}, {6203552979315789704ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1113395566489815627ull, 3}, {13957701839509617452ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9226604805100152147ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13302932820562179799ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15316838452862012827ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5765263465902070143ull, 1}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 4}, {.Sum = 1.4013e-44, .Count = 1}, {.Sum = 1.26117e-44, .Count = 16}, {.Sum = 0, .Count = 18}, {.Sum = 0, .Count = 22}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 12}, {.Sum = 2.8026e-45, .Count = 3}, {.Sum = 0, .Count = 2}},
                        .CtrTotal = {0, 4, 10, 1, 9, 16, 0, 18, 0, 22, 1, 1, 0, 12, 2, 3, 0, 2}
                    }
                },
                {
                    12627245789391619613ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9226604805100152147ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1034166431492604838ull, 1}, {13302932820562179799ull, 2}, {6203552979315789704ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1113395566489815627ull, 0}, {13957701839509617452ull, 7}, {15316838452862012827ull, 3}, {18446744073709551615ull, 0}, {5765263465902070143ull, 6}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 22}, {.Sum = 2.52234e-44, .Count = 17}, {.Sum = 0, .Count = 22}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 13}, {.Sum = 2.8026e-45, .Count = 3}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 22, 18, 17, 0, 22, 1, 1, 0, 13, 2, 3, 1, 0, 0, 1}
                    }
                },
                {
                    12627245789391619615ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {3581428127016485793ull, 2}, {1236773280081879954ull, 1}, {16151796118569799858ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18336378346035991543ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13125720576600207402ull, 3}, {5967870314491345259ull, 6}, {9724886183021484844ull, 4}, {18446744073709551615ull, 0}, {13605281311626526238ull, 5}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 37}, {.Sum = 2.94273e-44, .Count = 20}, {.Sum = 0, .Count = 13}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 37, 21, 20, 0, 13, 0, 2, 0, 4, 0, 2, 1, 0, 0, 1}
                    }
                },
                {
                    17677952493261641995ull,
                    {
                        .IndexHashViewer = {{7458091914254611456ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4709287016198198532ull, 9}, {11891385945082349892ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16624566716182634315ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8188814934051861073ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4569428324804022359ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5629641527707403930ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11910326597643169058ull, 11}, {16012272658388189795ull, 14}, {7930141458505850467ull, 19}, {16604351646315406629ull, 16}, {17723738371509991206ull, 4}, {1862677700213432292ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16566219115744069547ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11902478327942383792ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13377843633458007987ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7687100899529582134ull, 1}, {10629038735401595063ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9943717546119900283ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6610044300938801023ull, 6}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 2.8026e-45, .Count = 15}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 9.80909e-45, .Count = 3}, {.Sum = 5.60519e-45, .Count = 15}, {.Sum = 5.60519e-45, .Count = 22}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {2, 15, 2, 0, 7, 3, 4, 15, 4, 22, 0, 5, 0, 2, 0, 1, 0, 3, 0, 2, 0, 4, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1}
                    }
                },
                {
                    4544184825393173621ull,
                    {
                        .IndexHashViewer = {{11772109559350781439ull, 4}, {18446744073709551615ull, 0}, {12337359831519453058ull, 0}, {18446744073709551615ull, 0}, {3462861689708330564ull, 10}, {6193042878898900581ull, 7}, {9955981968190923718ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7606262797109987753ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6973539969458659060ull, 2}, {13860744542689514389ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2242442935049193755ull, 3}, {9129647508280049084ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14687079002600389023ull, 1}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 7.00649e-45, .Count = 5}, {.Sum = 7.00649e-45, .Count = 53}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 0, .Count = 5}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 5}, {.Sum = 4.2039e-45, .Count = 0}, {.Sum = 0, .Count = 4}},
                        .CtrTotal = {1, 1, 1, 1, 5, 5, 5, 53, 1, 5, 0, 5, 2, 0, 2, 0, 2, 5, 3, 0, 0, 4}
                    }
                },
                {
                    17677952493261641996ull,
                    {
                        .IndexHashViewer = {{16259707375369223360ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13847085545544291780ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7654620248348398600ull, 2}, {18446744073709551615ull, 0}, {9243796653651753418ull, 5}, {18446744073709551615ull, 0}, {1681026541770505292ull, 22}, {1292491219513334285ull, 21}, {13677090684479491854ull, 23}, {6494991755595340494ull, 15}, {7494438315637327440ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18410271455579776277ull, 14}, {6336919059871405781ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9974519673449003035ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5899717636280359390ull, 13}, {18446744073709551615ull, 0}, {15904544917366469984ull, 1}, {18446744073709551615ull, 0}, {862592111642406882ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18161161563956788133ull, 11}, {18446744073709551615ull, 0}, {3340544229935902247ull, 12}, {18446744073709551615ull, 0}, {14827488318775688873ull, 16}, {15675535932091499306ull, 3}, {18446744073709551615ull, 0}, {15230422751883885548ull, 24}, {18446744073709551615ull, 0}, {1662085889209686126ull, 27}, {18446744073709551615ull, 0}, {1062699037197581552ull, 4}, {14072903496117963889ull, 17}, {18446744073709551615ull, 0}, {15434641073738489523ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14277121817972567864ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18160464660109825851ull, 9}, {16406258951888748923ull, 18}, {17480885798804750972ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 12}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 3}, {.Sum = 5.60519e-45, .Count = 7}, {.Sum = 0, .Count = 11}, {.Sum = 0, .Count = 8}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 5}, {.Sum = 5.60519e-45, .Count = 7}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 9.80909e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 4.2039e-45, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {0, 12, 2, 0, 0, 3, 4, 7, 0, 11, 0, 8, 0, 1, 0, 5, 4, 7, 0, 2, 0, 1, 7, 0, 0, 2, 0, 2, 0, 3, 0, 2, 1, 1, 0, 2, 3, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0}
                    }
                },
                {
                    5445777084271881924ull,
                    {
                        .IndexHashViewer = {{17151879688829397503ull, 2}, {18446744073709551615ull, 0}, {14474606344715696898ull, 3}, {14282620878612260867ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12420782654419932198ull, 4}, {18446744073709551615ull, 0}, {15473953381485119304ull, 6}, {18446744073709551615ull, 0}, {9551523844202795562ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10742856347075653999ull, 5}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 2.8026e-45, .Count = 40}, {.Sum = 2.66247e-44, .Count = 21}, {.Sum = 0, .Count = 6}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 6}, {.Sum = 1.4013e-45, .Count = 4}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {2, 40, 19, 21, 0, 6, 0, 1, 0, 6, 1, 4, 0, 1}
                    }
                },
                {
                    17677952493263343771ull,
                    {
                        .IndexHashViewer = {{15330345801530070271ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13871343560304450565ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17989766274143549768ull, 22}, {18334501489220455433ull, 24}, {17271881404906880906ull, 17}, {1327065643761606346ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5745149923951351887ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18147836298725285973ull, 23}, {11919737177904201494ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {220392991226246300ull, 8}, {11009125960592947549ull, 19}, {16732756202475478686ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1799168355831033313ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17299587346058626214ull, 16}, {945432601379406567ull, 4}, {18446744073709551615ull, 0}, {227547732142737705ull, 3}, {8878683662908522218ull, 5}, {18371399316525749547ull, 15}, {18446744073709551615ull, 0}, {12391839889973628461ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4342739523005943472ull, 21}, {18446744073709551615ull, 0}, {10362267276645262642ull, 1}, {6966500923373419635ull, 7}, {9445514806491669746ull, 18}, {10820219266285332853ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17559172457516014783ull, 14}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 8.40779e-45, .Count = 8}, {.Sum = 0, .Count = 6}, {.Sum = 2.8026e-45, .Count = 5}, {.Sum = 0, .Count = 8}, {.Sum = 2.8026e-45, .Count = 8}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 2.8026e-45, .Count = 11}, {.Sum = 0, .Count = 1}, {.Sum = 9.80909e-45, .Count = 6}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {6, 8, 0, 6, 2, 5, 0, 8, 2, 8, 0, 1, 1, 3, 2, 11, 0, 1, 7, 6, 0, 2, 1, 3, 0, 2, 0, 2, 0, 3, 0, 1, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}
                    }
                },
                {
                    15655841788288703925ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {3581428127016485793ull, 2}, {1236773280081879954ull, 3}, {16151796118569799858ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13125720576600207402ull, 4}, {5967870314491345259ull, 1}, {9724886183021484844ull, 5}, {18446744073709551615ull, 0}, {13605281311626526238ull, 6}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 37}, {.Sum = 4.2039e-45, .Count = 1}, {.Sum = 0, .Count = 13}, {.Sum = 2.66247e-44, .Count = 19}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 3}},
                        .CtrTotal = {0, 37, 3, 1, 0, 13, 19, 19, 0, 2, 0, 4, 0, 3}
                    }
                },
                {
                    8405694746487331109ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8473802870189803490ull, 0}, {7071392469244395075ull, 3}, {18446744073709551615ull, 0}, {8806438445905145973ull, 2}, {619730330622847022ull, 1}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 1.14906e-43, .Count = 12}, {.Sum = 1.4013e-45, .Count = 6}},
                        .CtrTotal = {82, 12, 1, 6}
                    }
                },
                {
                    2790902285321313792ull,
                    {
                        .IndexHashViewer = {{8975491433706742463ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14435487234778955461ull, 22}, {26794562384612742ull, 19}, {18446744073709551615ull, 0}, {4411634050168915016ull, 2}, {11361933621181601929ull, 1}, {15118949489711741514ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {488596013123191629ull, 6}, {2041917558348994126ull, 16}, {18446744073709551615ull, 0}, {3099115351550504912ull, 23}, {13955926499752636625ull, 5}, {6798076237643774482ull, 17}, {10555092106173914067ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4633306462361102487ull, 11}, {18446744073709551615ull, 0}, {16982002041722229081ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14612285549902308191ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {779318031744854123ull, 10}, {18446744073709551615ull, 0}, {4020317248344823341ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6630836586772136624ull, 13}, {18446744073709551615ull, 0}, {15927023829150890738ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2066979203234309177ull, 3}, {16388825279889469625ull, 15}, {18446744073709551615ull, 0}, {6364972095279429180ull, 12}, {18446744073709551615ull, 0}, {18348953501661188798ull, 9}, {18144006785123939903ull, 21}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 8}, {.Sum = 1.26117e-44, .Count = 18}, {.Sum = 0, .Count = 26}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 5.60519e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 4.2039e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}},
                        .CtrTotal = {0, 4, 1, 1, 0, 8, 9, 18, 0, 26, 0, 2, 0, 2, 0, 1, 0, 3, 4, 0, 2, 0, 1, 0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 2, 1, 0, 0, 1, 0, 3, 1, 0, 3, 0, 0, 1, 0, 2}
                    }
                },
                {
                    8405694746487331111ull,
                    {
                        .IndexHashViewer = {{2136296385601851904ull, 0}, {7428730412605434673ull, 5}, {9959754109938180626ull, 2}, {14256903225472974739ull, 3}, {8056048104805248435ull, 1}, {18446744073709551615ull, 0}, {12130603730978457510ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10789443546307262781ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 1.30321e-43, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 2}},
                        .CtrTotal = {93, 2, 1, 1, 1, 2, 1}
                    }
                },
                {
                    5819498284603408945ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {401572100562674692ull, 2}, {18446744073709551615ull, 0}, {15483923052928748550ull, 39}, {12879637026568809095ull, 41}, {793550578637923848ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11367447619088669583ull, 11}, {5265189619257386128ull, 7}, {4243019055446252944ull, 43}, {7714913839382636178ull, 16}, {18446744073709551615ull, 0}, {2395930809040249492ull, 13}, {116261182353282069ull, 47}, {6322089685280714644ull, 38}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14730630066036795803ull, 46}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13687336289042331679ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5278875108181112996ull, 26}, {4457098312139029797ull, 12}, {12062999459536534438ull, 6}, {18446744073709551615ull, 0}, {2409616297963976360ull, 3}, {18446744073709551615ull, 0}, {6401305903214724138ull, 22}, {18446744073709551615ull, 0}, {13010046892757689900ull, 30}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2457936645703814959ull, 34}, {11036119054636294576ull, 21}, {9928946807531223473ull, 33}, {2486846435837533490ull, 24}, {18035421909196229939ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14292590550055890113ull, 36}, {13993739391568110402ull, 27}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17239079498146583879ull, 15}, {18446744073709551615ull, 0}, {1488087840841630025ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10635501563021545804ull, 49}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1670208868805258833ull, 28}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5733202716806891092ull, 18}, {12366794655858091989ull, 42}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2328590962904621532ull, 25}, {18446744073709551615ull, 0}, {7642631318464954334ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2966914650778391649ull, 23}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16422441931114309222ull, 37}, {3772880582916128230ull, 35}, {18446744073709551615ull, 0}, {8454321268541353705ull, 40}, {13553183120897172586ull, 0}, {6965341922180312555ull, 32}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12647497933473231982ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8531551406414910835ull, 8}, {3948120307064453107ull, 31}, {1150935252797962357ull, 20}, {18446744073709551615ull, 0}, {1078861496847316471ull, 9}, {780010338359536760ull, 48}, {18446744073709551615ull, 0}, {12321202553328987770ull, 29}, {13267872202687779963ull, 45}, {18341677509001906300ull, 44}, {12802406888695251965ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 7}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 5.60519e-45, .Count = 5}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 5}, {.Sum = 4.2039e-45, .Count = 6}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 7.00649e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {0, 7, 2, 0, 0, 1, 4, 5, 0, 5, 0, 5, 0, 5, 0, 1, 0, 1, 0, 5, 3, 6, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 4, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 2, 0, 0, 1, 0, 1, 0, 1, 5, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0}
                    }
                },
                {
                    768791580653471469ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8473802870189803490ull, 0}, {7071392469244395075ull, 3}, {18446744073709551615ull, 0}, {8806438445905145973ull, 2}, {619730330622847022ull, 1}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 2.94273e-44, .Count = 61}, {.Sum = 0, .Count = 12}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 5}},
                        .CtrTotal = {21, 61, 0, 12, 0, 1, 1, 5}
                    }
                },
                {
                    768791580653471471ull,
                    {
                        .IndexHashViewer = {{2136296385601851904ull, 0}, {7428730412605434673ull, 5}, {9959754109938180626ull, 2}, {14256903225472974739ull, 3}, {8056048104805248435ull, 1}, {18446744073709551615ull, 0}, {12130603730978457510ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10789443546307262781ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 2.8026e-44, .Count = 73}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {20, 73, 0, 2, 1, 0, 0, 1, 1, 0, 0, 2, 0, 1}
                    }
                },
                {
                    17677952493260528848ull,
                    {
                        .IndexHashViewer = {{3632340108106778112ull, 12}, {84580555217079201ull, 5}, {1856503610704726976ull, 8}, {12055230997206289283ull, 2}, {16771526449192646880ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3152779373080459276ull, 4}, {14225011642249373260ull, 9}, {18198689053211288334ull, 6}, {16796278652265879919ull, 13}, {4201158457639332815ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9710576150271683444ull, 1}, {6178854915050051732ull, 0}, {8308165749326275029ull, 11}, {4776444514104643317ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 28}, {.Sum = 2.94273e-44, .Count = 17}, {.Sum = 0, .Count = 11}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 28, 21, 17, 0, 11, 0, 2, 0, 2, 0, 1, 0, 2, 0, 5, 0, 3, 0, 2, 0, 4, 1, 0, 0, 1, 0, 1}
                    }
                },
                {
                    17677952493265578087ull,
                    {
                        .IndexHashViewer = {{18171586759681088672ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14557079040021784102ull, 1}, {1894223316800506727ull, 9}, {18446744073709551615ull, 0}, {11879805695908083497ull, 2}, {11687820229804647466ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12879152732677505903ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4426716004344559893ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8230941806183355321ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1064533880431424572ull, 5}, {17607571949008043997ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 2.8026e-44, .Count = 58}, {.Sum = 0, .Count = 11}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 3}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {20, 58, 0, 11, 0, 1, 0, 2, 0, 3, 1, 0, 0, 1, 1, 0, 0, 2, 0, 1}
                    }
                },
                {
                    768791580653471472ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13987540656699198946ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18089724839685297862ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10353740403438739754ull, 1}, {3922001124998993866ull, 0}, {13686716744772876732ull, 4}, {18293943161539901837ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 37}, {.Sum = 3.08286e-44, .Count = 20}, {.Sum = 0, .Count = 13}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 3}},
                        .CtrTotal = {0, 37, 22, 20, 0, 13, 0, 2, 0, 4, 0, 3}
                    }
                },
                {
                    768791580653471473ull,
                    {
                        .IndexHashViewer = {{7537614347373541888ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5903587924673389870ull, 4}, {18278593470046426063ull, 9}, {10490918088663114479ull, 8}, {18446744073709551615ull, 0}, {407784798908322194ull, 5}, {5726141494028968211ull, 6}, {1663272627194921140ull, 7}, {8118089682304925684ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15431483020081801594ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1403990565605003389ull, 0}, {3699047549849816830ull, 1}, {14914630290137473119ull, 2}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 4.2039e-45, .Count = 16}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 9.80909e-45, .Count = 3}, {.Sum = 5.60519e-45, .Count = 16}, {.Sum = 5.60519e-45, .Count = 24}, {.Sum = 0, .Count = 5}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {3, 16, 2, 0, 7, 3, 4, 16, 4, 24, 0, 5, 1, 3, 0, 3, 0, 3, 0, 4, 0, 1, 1, 1}
                    }
                },
                {
                    17677952493260528850ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {5717724512618337697ull, 2}, {18446744073709551615ull, 0}, {5133782457465682915ull, 12}, {11196527390020060580ull, 8}, {11961955270333222981ull, 9}, {5761100149665496677ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15262016962202059306ull, 3}, {18446744073709551615ull, 0}, {11861182568623336748ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12026216826389142735ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3373069665683731858ull, 1}, {18288092504171651762ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13367377011060337464ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17153616595626919517ull, 11}, {15741577697228378142ull, 6}, {17780934287826733279ull, 5}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 34}, {.Sum = 2.8026e-44, .Count = 19}, {.Sum = 0, .Count = 13}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 34, 20, 19, 0, 13, 0, 2, 0, 2, 0, 1, 0, 3, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1}
                    }
                },
                {
                    768791580653471474ull,
                    {
                        .IndexHashViewer = {{3607388709394294015ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18356215166324018775ull, 4}, {18365206492781874408ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14559146096844143499ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11416626865500250542ull, 1}, {5549384008678792175ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 22}, {.Sum = 2.66247e-44, .Count = 17}, {.Sum = 0, .Count = 22}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 14}, {.Sum = 2.8026e-45, .Count = 3}},
                        .CtrTotal = {0, 22, 19, 17, 0, 22, 1, 1, 0, 14, 2, 3}
                    }
                },
                {
                    768791580653471475ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {14452488454682494753ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1388452262538353895ull, 8}, {8940247467966214344ull, 2}, {4415016594903340137ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {41084306841859596ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8678739366408346384ull, 1}, {18446744073709551615ull, 0}, {4544226147037566482ull, 11}, {14256903225472974739ull, 5}, {16748601451484174196ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5913522704362245435ull, 3}, {1466902651052050075ull, 7}, {2942073219785550491ull, 12}, {15383677753867481021ull, 6}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 8.40779e-45, .Count = 10}, {.Sum = 0, .Count = 6}, {.Sum = 2.8026e-45, .Count = 8}, {.Sum = 0, .Count = 11}, {.Sum = 2.8026e-45, .Count = 9}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 2.8026e-45, .Count = 14}, {.Sum = 0, .Count = 2}, {.Sum = 9.80909e-45, .Count = 6}, {.Sum = 1.4013e-45, .Count = 4}, {.Sum = 0, .Count = 3}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {6, 10, 0, 6, 2, 8, 0, 11, 2, 9, 1, 5, 2, 14, 0, 2, 7, 6, 1, 4, 0, 3, 1, 0, 0, 1}
                    }
                },
                {
                    17677952493260528854ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17985374731566054150ull, 24}, {18446744073709551615ull, 0}, {4969880389554839688ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1883285504791108373ull, 36}, {14139902777924824981ull, 35}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17540248381108153753ull, 27}, {18446744073709551615ull, 0}, {2120068639763588379ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1277857586923739550ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9915512646490226338ull, 3}, {18446744073709551615ull, 0}, {5780999427119446436ull, 30}, {15493676505554854693ull, 29}, {14453653496344422438ull, 28}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3622512433858345389ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9415440463389949361ull, 19}, {18446744073709551615ull, 0}, {15689261734764374707ull, 26}, {17838331352489460532ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18403625549429831228ull, 12}, {18446744073709551615ull, 0}, {16192880425411659454ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6383791411268594626ull, 20}, {18033916581698980546ull, 34}, {18446744073709551615ull, 0}, {11961955270333222981ull, 8}, {18446744073709551615ull, 0}, {11191788834073534919ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5730630563994427981ull, 23}, {647125264645798733ull, 37}, {16620451033949360975ull, 10}, {17618698769621849933ull, 38}, {7150295984444125389ull, 17}, {18446744073709551615ull, 0}, {12157540499542742995ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1072059942279220057ull, 25}, {10177020748048094298ull, 14}, {18446744073709551615ull, 0}, {9494950831378731228ull, 33}, {18446744073709551615ull, 0}, {518361807174415198ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {592499207252901221ull, 7}, {4098784705883188966ull, 31}, {10062654256758136807ull, 32}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3618574749222493677ull, 5}, {18446744073709551615ull, 0}, {13088729798727729263ull, 21}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2625225542620233849ull, 13}, {6645299512826462586ull, 4}, {5651789874985220091ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 5}, {.Sum = 8.40779e-45, .Count = 1}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-45, .Count = 7}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 9.80909e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 4}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 5, 6, 1, 0, 4, 0, 2, 0, 5, 0, 7, 0, 3, 0, 1, 0, 2, 0, 2, 2, 7, 0, 1, 0, 1, 7, 1, 2, 2, 0, 2, 0, 2, 0, 1, 0, 2, 0, 1, 0, 3, 0, 4, 1, 1, 0, 1, 0, 1, 0, 1, 2, 4, 0, 3, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}
                    }
                },
                {
                    768791580653471478ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {15379737126276794113ull, 5}, {18446744073709551615ull, 0}, {14256903225472974739ull, 3}, {18048946643763804916ull, 6}, {2051959227349154549ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7024059537692152076ull, 4}, {18446744073709551615ull, 0}, {15472181234288693070ull, 1}, {8864790892067322495ull, 2}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 4.2039e-45, .Count = 6}, {.Sum = 1.4013e-45, .Count = 6}, {.Sum = 1.4013e-44, .Count = 58}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 7.00649e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 4}},
                        .CtrTotal = {3, 6, 1, 6, 10, 58, 1, 5, 5, 0, 2, 0, 0, 4}
                    }
                },
                {
                    768791582220620454ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {8729380230485332353ull, 34}, {9977784445157143938ull, 47}, {10895282230322158083ull, 22}, {11761827888484752132ull, 4}, {12615796933138713349ull, 42}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16179912168674497673ull, 19}, {11168821652508931466ull, 17}, {18446744073709551615ull, 0}, {10475827037446056716ull, 45}, {14750448552479345421ull, 11}, {16495354791085375886ull, 13}, {10135854469738880143ull, 48}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12233930600554179099ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3208547115700824735ull, 14}, {18277252057819687584ull, 40}, {11380194846102552353ull, 6}, {18446744073709551615ull, 0}, {14030234294679959587ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14597065438263107115ull, 46}, {1433532977505263916ull, 37}, {17401263168096565421ull, 24}, {15971769898056770222ull, 43}, {2808237437298927023ull, 20}, {1256940733300575664ull, 21}, {18446744073709551615ull, 0}, {9689317982178109362ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6069959757871852856ull, 1}, {7318363972543664441ull, 41}, {18446744073709551615ull, 0}, {5876843741908031419ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5859361731318044735ull, 36}, {6636790901455000384ull, 30}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7078520115819519811ull, 28}, {10322519660234049603ull, 49}, {8011495361248663491ull, 18}, {9259899575920475076ull, 35}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9323661606633862475ull, 32}, {18146214905751188428ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3907755348917795664ull, 33}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8720774373489072856ull, 0}, {6896376012912388953ull, 27}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10662309976865883491ull, 5}, {9111013272867532132ull, 15}, {10359417487539343717ull, 29}, {17543390521745065830ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13529097553057277673ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2350285447658856812ull, 23}, {16689654767293439084ull, 44}, {18446744073709551615ull, 0}, {6176161755113003119ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {346329515989321719ull, 39}, {4263979015577900536ull, 10}, {6353480505666359544ull, 12}, {9547190628529608953ull, 31}, {9583115984936959099ull, 38}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 4}, {.Sum = 8.40779e-45, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-45, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 9.80909e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 3}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 4, 6, 1, 0, 3, 0, 2, 0, 3, 0, 5, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 2, 5, 0, 1, 0, 1, 7, 1, 2, 1, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 3, 0, 3, 1, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 4, 0, 1, 0, 3, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}
                    }
                },
                {
                    8405694746487331128ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13987540656699198946ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18089724839685297862ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10353740403438739754ull, 1}, {3922001124998993866ull, 0}, {13686716744772876732ull, 4}, {18293943161539901837ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 5.1848e-44, .Count = 42}, {.Sum = 1.82169e-44, .Count = 2}, {.Sum = 5.60519e-45, .Count = 3}},
                        .CtrTotal = {37, 42, 13, 2, 4, 3}
                    }
                },
                {
                    5819498284355557857ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16651102300929268102ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17241896836922419469ull, 14}, {18446744073709551615ull, 0}, {10511965914255575823ull, 39}, {9263222292810378768ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6908362001823204373ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10984690017285232025ull, 34}, {18446744073709551615ull, 0}, {13013334951445741211ull, 25}, {18446744073709551615ull, 0}, {11118050854346541341ull, 38}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1178507314092247330ull, 7}, {18124759156733634467ull, 19}, {11481715753106083236ull, 10}, {5594842188654002339ull, 29}, {13183845322349047206ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15878007324930404523ull, 30}, {18446744073709551615ull, 0}, {5342579366432390957ull, 28}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13092564086442952000ull, 0}, {12955372608910449601ull, 32}, {11197279989343752130ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2759029251343736904ull, 15}, {11560944888103294025ull, 9}, {863745154244537034ull, 24}, {13263074457346257995ull, 31}, {6835357266805764556ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1227410053004078929ull, 16}, {5421808501429601746ull, 2}, {2929539622247042899ull, 33}, {18446744073709551615ull, 0}, {5303738068466106581ull, 4}, {18446744073709551615ull, 0}, {7005867637709070551ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13535017740039938266ull, 27}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6084421232781469407ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1416006194736185826ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14197630471391805927ull, 18}, {17162667701925208680ull, 23}, {18446744073709551615ull, 0}, {9529215433346522346ull, 36}, {18273958486853833579ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2701066908808445294ull, 11}, {13605543448808549998ull, 35}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17744431985855206516ull, 21}, {11659615095675342580ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3609240935860829562ull, 40}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18182721499268926077ull, 37}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 9}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 5.60519e-45, .Count = 7}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 6}, {.Sum = 0, .Count = 5}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 6}, {.Sum = 4.2039e-45, .Count = 6}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 7.00649e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {0, 9, 2, 0, 0, 1, 4, 7, 0, 7, 0, 6, 0, 5, 1, 1, 0, 1, 0, 6, 3, 6, 0, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 4, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 2, 0, 1, 2, 0, 0, 1, 0, 1, 5, 0, 0, 1, 0, 1, 0, 2, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0}
                    }
                },
                {
                    17677952493297533872ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {228832412018222341ull, 6}, {18446744073709551615ull, 0}, {11579036573410064263ull, 40}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2142920538933900555ull, 20}, {18446744073709551615ull, 0}, {11420714090427158285ull, 19}, {18446744073709551615ull, 0}, {17720405802426315535ull, 5}, {3215834049561110672ull, 33}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {346575239343974036ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13139983920087306647ull, 32}, {14860408764928037144ull, 1}, {286844492446271769ull, 41}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10925792178412610972ull, 23}, {12726869934920056605ull, 27}, {11945848411936959644ull, 46}, {18446744073709551615ull, 0}, {11343638620497380128ull, 42}, {9857611124702919969ull, 11}, {15541558334966787106ull, 50}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10990677728635501222ull, 45}, {4919457811166910375ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4237122415554814250ull, 37}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {339035928827901487ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8200830002684883256ull, 0}, {6893797804197340345ull, 13}, {1058988547593232698ull, 16}, {11714417785040418747ull, 14}, {18446744073709551615ull, 0}, {6067291172676902717ull, 31}, {16636473811085647678ull, 26}, {18446744073709551615ull, 0}, {483329372556896832ull, 30}, {3198032362459766081ull, 24}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12661894127993305031ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4340360739111205579ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1471101928894068943ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {464994231589622356ull, 35}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14915048362378503384ull, 10}, {5278641733246315480ull, 12}, {1537907742216832473ull, 29}, {5054839022797264859ull, 38}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6888411174261376229ull, 34}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16939687026671270763ull, 51}, {14120581721888279787ull, 36}, {18080292852670312173ull, 25}, {7952734526884932333ull, 47}, {8723830392309106799ull, 28}, {9875412811804264560ull, 21}, {15038402360561546607ull, 52}, {16771855022716002162ull, 17}, {5933240490959917807ull, 18}, {7006154001587127924ull, 15}, {8813616260402002415ull, 39}, {18446744073709551615ull, 0}, {5540766610232480247ull, 48}, {18446744073709551615ull, 0}, {16586264761736307193ull, 44}, {18446744073709551615ull, 0}, {6712598941894663547ull, 49}, {17585370940655764860ull, 3}, {9392162505557741693ull, 43}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 3}, {.Sum = 7.00649e-45, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-45, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 9.80909e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 3}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 3, 5, 1, 0, 3, 0, 2, 0, 3, 0, 5, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 2, 5, 0, 1, 0, 1, 7, 1, 2, 1, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 3, 0, 3, 1, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 1, 0, 0, 3, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}
                    }
                },
                {
                    8405694746487331129ull,
                    {
                        .IndexHashViewer = {{7537614347373541888ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5903587924673389870ull, 4}, {18278593470046426063ull, 9}, {10490918088663114479ull, 8}, {18446744073709551615ull, 0}, {407784798908322194ull, 5}, {5726141494028968211ull, 6}, {1663272627194921140ull, 7}, {8118089682304925684ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15431483020081801594ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1403990565605003389ull, 0}, {3699047549849816830ull, 1}, {14914630290137473119ull, 2}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 2.66247e-44, .Count = 2}, {.Sum = 1.4013e-44, .Count = 20}, {.Sum = 3.92364e-44, .Count = 5}, {.Sum = 5.60519e-45, .Count = 3}, {.Sum = 4.2039e-45, .Count = 4}, {.Sum = 1.4013e-45, .Count = 2}},
                        .CtrTotal = {19, 2, 10, 20, 28, 5, 4, 3, 3, 4, 1, 2}
                    }
                },
                {
                    8405694746487331131ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {14452488454682494753ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1388452262538353895ull, 8}, {8940247467966214344ull, 2}, {4415016594903340137ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {41084306841859596ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8678739366408346384ull, 1}, {18446744073709551615ull, 0}, {4544226147037566482ull, 11}, {14256903225472974739ull, 5}, {16748601451484174196ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5913522704362245435ull, 3}, {1466902651052050075ull, 7}, {2942073219785550491ull, 12}, {15383677753867481021ull, 6}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 2.24208e-44, .Count = 6}, {.Sum = 1.4013e-44, .Count = 11}, {.Sum = 1.54143e-44, .Count = 6}, {.Sum = 2.24208e-44, .Count = 2}, {.Sum = 1.82169e-44, .Count = 5}, {.Sum = 4.2039e-45, .Count = 1}},
                        .CtrTotal = {16, 6, 10, 11, 11, 6, 16, 2, 13, 5, 3, 1, 1}
                    }
                },
                {
                    8405694746487331134ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {15379737126276794113ull, 5}, {18446744073709551615ull, 0}, {14256903225472974739ull, 3}, {18048946643763804916ull, 6}, {2051959227349154549ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7024059537692152076ull, 4}, {18446744073709551615ull, 0}, {15472181234288693070ull, 1}, {8864790892067322495ull, 2}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 1.26117e-44, .Count = 7}, {.Sum = 9.52883e-44, .Count = 6}, {.Sum = 7.00649e-45, .Count = 2}},
                        .CtrTotal = {9, 7, 68, 6, 5, 2, 4}
                    }
                },
                {
                    4414881145659723684ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1968438200869838210ull, 51}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17679689593011713800ull, 19}, {17770863836821556104ull, 0}, {5873407800382985098ull, 52}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15354859179876249743ull, 29}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8241769915270219922ull, 49}, {18446744073709551615ull, 0}, {4781221601667622548ull, 28}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13259958884694735255ull, 41}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4656770217989207578ull, 33}, {3123998877235734427ull, 44}, {2880537555826393628ull, 14}, {1045758839855420444ull, 50}, {6453812626951551388ull, 20}, {7373521406379420317ull, 10}, {15208993820173816478ull, 47}, {5036837526151520545ull, 46}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {785637260644879140ull, 39}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14201924189220883111ull, 1}, {1883760134919382184ull, 45}, {7171489145492152617ull, 3}, {2203248159541751849ull, 34}, {5114067664025077675ull, 7}, {7763215270077623596ull, 24}, {18433555596902063917ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7248719283365709747ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14392356253141541821ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3334767067226870465ull, 37}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9822303761109372617ull, 6}, {1918034629091685706ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2657477670942228686ull, 15}, {14131880866619031375ull, 13}, {9892630029936032464ull, 23}, {18446744073709551615ull, 0}, {6432081716333435090ull, 38}, {12606536426880398291ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1240745232807043927ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9144531774667173594ull, 30}, {6557251021933574875ull, 27}, {1262915927860373596ull, 21}, {18446744073709551615ull, 0}, {7116775155420360158ull, 53}, {12404504165993130591ull, 11}, {10606002133962586720ull, 48}, {63527192270722015ull, 36}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {133853461097381862ull, 43}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11727214769708488812ull, 32}, {9928712737677944941ull, 31}, {18048189026166942061ull, 35}, {15146535830480538223ull, 25}, {17409370781001408239ull, 40}, {303226289080229489ull, 12}, {9082896331950655090ull, 4}, {12760211465443864178ull, 17}, {18446744073709551615ull, 0}, {7611634590235858933ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4722706790762200317ull, 42}, {18446744073709551615ull, 0}, {15055318781610350591ull, 5}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 3}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 4.2039e-45, .Count = 6}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 8}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 7}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 4.2039e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 3, 1, 0, 0, 2, 3, 6, 0, 7, 0, 8, 0, 5, 0, 1, 0, 1, 0, 4, 0, 1, 1, 7, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3, 0, 2, 0, 1, 0, 1, 1, 0, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 3, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 2, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1}
                    }
                },
                {
                    4414881145133934893ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15744222165914032517ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5649296009765293714ull, 4}, {12515044663515391091ull, 1}, {7217896454495632755ull, 8}, {18446744073709551615ull, 0}, {16950186414058081238ull, 6}, {16498401743253028919ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10434762244723145786ull, 0}, {7783947629105925786ull, 3}, {6133087514440113244ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 37}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 13}, {.Sum = 2.66247e-44, .Count = 20}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 37, 2, 0, 0, 13, 19, 20, 0, 2, 0, 4, 0, 2, 1, 0, 0, 1}
                    }
                },
                {
                    4544184825161771334ull,
                    {
                        .IndexHashViewer = {{158999094665252608ull, 33}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8367117909645166341ull, 19}, {18446744073709551615ull, 0}, {8702450991728868615ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6262126705356135693ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14946871803847873040ull, 34}, {13913159826293879825ull, 29}, {3752585126949001232ull, 64}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14375335943966472213ull, 32}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6045632691652965145ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12440274563092365353ull, 46}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6619783738664723247ull, 23}, {18446744073709551615ull, 0}, {4905776570447084337ull, 37}, {18446744073709551615ull, 0}, {8130996685331913523ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14732992977275059767ull, 56}, {18446744073709551615ull, 0}, {2585940615564665401ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3128199045106796348ull, 47}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7735182141653903170ull, 7}, {17619157723446594371ull, 44}, {11241408283717132868ull, 48}, {13574756925474066500ull, 53}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18314399080722540106ull, 27}, {4146295242583377226ull, 43}, {18446744073709551615ull, 0}, {3172219588709525325ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17343663287863212120ull, 20}, {174679536037619032ull, 24}, {18446744073709551615ull, 0}, {79769959668041819ull, 51}, {16685972765223635547ull, 54}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16843846084406882659ull, 11}, {518059473081761380ull, 52}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6182897570358924904ull, 45}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6510311249307667563ull, 21}, {12533704194145800556ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11009716383731513464ull, 41}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13808447517377610367ull, 42}, {18446744073709551615ull, 0}, {7824087488121779841ull, 1}, {13795416998968128130ull, 55}, {7469564332260859522ull, 59}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12443445925107193993ull, 50}, {6540554603667512458ull, 30}, {18446744073709551615ull, 0}, {18123001185196525196ull, 13}, {18446744073709551615ull, 0}, {8051767550334702734ull, 40}, {2891096590926338447ull, 62}, {18446744073709551615ull, 0}, {6116316705811167633ull, 0}, {9269497864691699089ull, 63}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8490057063748422295ull, 58}, {18446744073709551615ull, 0}, {4919885718948249ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {522424025960426143ull, 57}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17603342537294645418ull, 8}, {16803678464185371818ull, 61}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7617439314328151475ull, 6}, {18446744073709551615ull, 0}, {3320670515925237429ull, 26}, {13992388961291090614ull, 4}, {18446744073709551615ull, 0}, {1385219671401702328ull, 31}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6662268631026693053ull, 5}, {16764616949409671870ull, 12}, {6124861826650175934ull, 14}, {9498428910012038848ull, 36}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7162655135050725840ull, 35}, {12072581429775906513ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11977671853406329300ull, 39}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {197512786993793514ull, 49}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10586639264769695215ull, 28}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {635802300407513075ull, 25}, {18446744073709551615ull, 0}, {6475377084227676405ull, 60}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12735534006750400250ull, 38}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 6}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 1, 1, 0, 0, 1, 0, 2, 0, 2, 0, 6, 0, 4, 0, 3, 0, 1, 0, 3, 0, 2, 0, 3, 0, 7, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 2, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 2, 1, 0, 1, 2, 0, 0, 2, 0, 1, 0, 1, 1, 1, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 2, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1}
                    }
                },
                {
                    17677952491747546147ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17787954881284471813ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7454420046185256717ull, 20}, {16256335682944813838ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1636731659193698578ull, 48}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5922800847845598742ull, 22}, {14182197490569975831ull, 27}, {7624930417088562712ull, 52}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10422205982269444643ull, 44}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3411314423057176877ull, 30}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4522605207985801776ull, 29}, {18446744073709551615ull, 0}, {13192676729576349746ull, 62}, {16466569643076362291ull, 8}, {18300934243650069811ull, 58}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4431368220400894274ull, 60}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14233673023285815109ull, 50}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2899749022061236299ull, 53}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8023290181753164880ull, 65}, {9933882341717515345ull, 66}, {3233597379123467602ull, 47}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8402263143377857370ull, 37}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17651556054977644126ull, 21}, {15680080812126751838ull, 55}, {17708725746287261024ull, 28}, {18446744073709551615ull, 0}, {1780070264439091554ull, 19}, {15773274901763725923ull, 0}, {16328374789029446500ull, 51}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16177106547947603049ull, 13}, {18446744073709551615ull, 0}, {17879236117190567019ull, 3}, {3489127981302646635ull, 41}, {14241655703424067948ull, 56}, {15943785272667031918ull, 43}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9771448801094703501ull, 67}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11530748061647284369ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5994047302704556692ull, 57}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10117199296271121559ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9999128863307626394ull, 5}, {18446744073709551615ull, 0}, {11701258432550590364ull, 6}, {7854656800704835228ull, 39}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {118997543255608737ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10779812027622989220ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6111396989577705639ull, 24}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16127325828303939500ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5576091289432675759ull, 49}, {14224606228188042159ull, 59}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14966077412008197812ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10025163577623610551ull, 32}, {1755789550731085240ull, 64}, {7501413217152384697ull, 14}, {16355005890516862393ull, 46}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14797650915799523780ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13730933025438975688ull, 35}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {724243645116964305ull, 42}, {18446744073709551615ull, 0}, {11702735195037717203ull, 31}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16072394239031333591ull, 45}, {18446744073709551615ull, 0}, {11159883566315996889ull, 34}, {11603752796664724186ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16142728109259286750ull, 10}, {18446744073709551615ull, 0}, {17844857678502250720ull, 12}, {18446744073709551615ull, 0}, {9628264367976338914ull, 16}, {15813441649188061154ull, 61}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2145056323740669926ull, 40}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9516068082126538479ull, 54}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10037970161273910770ull, 38}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17560274819071548920ull, 23}, {11038948726666369272ull, 25}, {18446744073709551615ull, 0}, {8596718462362217979ull, 63}, {18446744073709551615ull, 0}, {10298848031605181949ull, 33}, {16215728555360712189ull, 36}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 4.2039e-45, .Count = 4}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 6}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 4.2039e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 2, 1, 0, 0, 1, 3, 4, 0, 4, 0, 3, 0, 4, 0, 3, 0, 1, 0, 1, 0, 1, 0, 4, 0, 1, 1, 6, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1}
                    }
                },
                {
                    8405694746995314031ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {14806117600143412865ull, 45}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9012783753182299908ull, 1}, {1339560154066889221ull, 11}, {18446744073709551615ull, 0}, {174039779367851655ull, 44}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16747653203520223500ull, 5}, {9579614896765447436ull, 31}, {18446744073709551615ull, 0}, {10954319356559110543ull, 20}, {18446744073709551615ull, 0}, {8837409026973740817ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8435628674313196821ull, 25}, {4328253920238227477ull, 46}, {13255493307163486358ull, 22}, {8244402790997920151ull, 47}, {2642294827352083864ull, 3}, {5465902517104623256ull, 32}, {13570935929574364571ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3313905660816128800ull, 48}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7407438120481433891ull, 43}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17847171115139092141ull, 7}, {15352833196308676269ull, 41}, {18446744073709551615ull, 0}, {11105815433168948272ull, 2}, {15759256038042246704ull, 9}, {12053837268177979184ull, 27}, {18446744073709551615ull, 0}, {15741774027452260020ull, 49}, {16471921548367724725ull, 28}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11672646576752095800ull, 52}, {18446744073709551615ull, 0}, {14476844306585554106ull, 33}, {18446744073709551615ull, 0}, {16788877028614866876ull, 16}, {18446744073709551615ull, 0}, {15961363380463923774ull, 51}, {18446744073709551615ull, 0}, {9163820133884450112ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18319273827208519108ull, 10}, {15824935908378103236ull, 24}, {18446744073709551615ull, 0}, {12525939980247406151ull, 39}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17465538072756892362ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11672204225795779405ull, 35}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4154101254308508496ull, 30}, {15065508147794825296ull, 53}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5039862321041353814ull, 26}, {18446744073709551615ull, 0}, {2097978199290547160ull, 23}, {17693272547789792473ull, 12}, {15903257226085232346ull, 13}, {8979058744169729499ull, 29}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12232697743793072097ull, 50}, {4691563912245177186ull, 40}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5796355511978061541ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11944704957972384744ull, 37}, {18446744073709551615ull, 0}, {555672821750051434ull, 17}, {6151371111011271787ull, 15}, {16407862886888389612ull, 54}, {14146391311712115821ull, 21}, {6363186655561209069ull, 38}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14618971660234054515ull, 18}, {8613402368625482612ull, 34}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3197496110909415801ull, 36}, {4051465155563377018ull, 42}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2604489874933595135ull, 19}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 5.60519e-45, .Count = 5}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 6}, {.Sum = 4.2039e-45, .Count = 3}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.26117e-44, .Count = 1}, {.Sum = 1.4013e-45, .Count = 7}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {4, 5, 2, 2, 1, 6, 3, 3, 1, 1, 2, 1, 9, 1, 1, 7, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 3, 1, 5, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1}
                    }
                },
                {
                    10041049327410906820ull,
                    {
                        .IndexHashViewer = {{16259707375369223360ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13847085545544291780ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7654620248348398600ull, 2}, {18446744073709551615ull, 0}, {9243796653651753418ull, 5}, {18446744073709551615ull, 0}, {1681026541770505292ull, 22}, {1292491219513334285ull, 21}, {13677090684479491854ull, 23}, {6494991755595340494ull, 15}, {7494438315637327440ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18410271455579776277ull, 14}, {6336919059871405781ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9974519673449003035ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5899717636280359390ull, 13}, {18446744073709551615ull, 0}, {15904544917366469984ull, 1}, {18446744073709551615ull, 0}, {862592111642406882ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18161161563956788133ull, 11}, {18446744073709551615ull, 0}, {3340544229935902247ull, 12}, {18446744073709551615ull, 0}, {14827488318775688873ull, 16}, {15675535932091499306ull, 3}, {18446744073709551615ull, 0}, {15230422751883885548ull, 24}, {18446744073709551615ull, 0}, {1662085889209686126ull, 27}, {18446744073709551615ull, 0}, {1062699037197581552ull, 4}, {14072903496117963889ull, 17}, {18446744073709551615ull, 0}, {15434641073738489523ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14277121817972567864ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18160464660109825851ull, 9}, {16406258951888748923ull, 18}, {17480885798804750972ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 1.68156e-44, .Count = 2}, {.Sum = 4.2039e-45, .Count = 11}, {.Sum = 1.54143e-44, .Count = 8}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 1.54143e-44, .Count = 2}, {.Sum = 1.4013e-45, .Count = 7}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 4.2039e-45, .Count = 2}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 5.60519e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {12, 2, 3, 11, 11, 8, 1, 5, 11, 2, 1, 7, 2, 2, 3, 2, 2, 2, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1}
                    }
                },
                {
                    9867321491374199501ull,
                    {
                        .IndexHashViewer = {{5321795528652759552ull, 3}, {1871794946608052991ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2572990630596346628ull, 14}, {9755089559480497988ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14488270330580782411ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10281901548337195535ull, 19}, {18446744073709551615ull, 0}, {6052518548450009169ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12538518194927513684ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9204844949746414424ull, 15}, {10052892563062224857ull, 6}, {3493345142105552026ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14486186593889963293ull, 21}, {7304087665005811933ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {772623291280696100ull, 20}, {18446744073709551615ull, 0}, {15587441985908139302ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10783615582859474474ull, 4}, {14429922730142217643ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8224442176515017331ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5550804513927730230ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7807421160518048379ull, 7}, {14505127246450782459ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4473747915336949119ull, 10}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 4.2039e-45, .Count = 7}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 4.2039e-45, .Count = 9}, {.Sum = 0, .Count = 9}, {.Sum = 2.8026e-45, .Count = 11}, {.Sum = 1.4013e-45, .Count = 7}, {.Sum = 7.00649e-45, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-45, .Count = 13}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {3, 7, 1, 0, 2, 1, 3, 9, 0, 9, 2, 11, 1, 7, 5, 2, 0, 2, 2, 13, 0, 1, 0, 1, 0, 3, 0, 1, 0, 2, 1, 2, 0, 4, 0, 1, 0, 1, 1, 0, 0, 2, 1, 0}
                    }
                },
                {
                    9867321491374199502ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {10934650013725255009ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5844492600899280932ull, 2}, {18446744073709551615ull, 0}, {1034166431492604838ull, 1}, {18446744073709551615ull, 0}, {6203552979315789704ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1113395566489815627ull, 0}, {13957701839509617452ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18034029854971645104ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9226604805100152147ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13302932820562179799ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15316838452862012827ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5765263465902070143ull, 7}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 10}, {.Sum = 1.54143e-44, .Count = 7}, {.Sum = 0, .Count = 12}, {.Sum = 0, .Count = 12}, {.Sum = 0, .Count = 10}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 8}, {.Sum = 1.12104e-44, .Count = 10}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 0, .Count = 6}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 10, 11, 7, 0, 12, 0, 12, 0, 10, 1, 1, 0, 8, 8, 10, 2, 2, 0, 6, 0, 1}
                    }
                },
                {
                    3863811882172310855ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16539280100125922053ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2933836577635514888ull, 24}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6624153332048651ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15584109416824463631ull, 5}, {1079537663959258768ull, 37}, {10795630651492496657ull, 40}, {18446744073709551615ull, 0}, {3653789196400846099ull, 17}, {16657022927451673748ull, 7}, {14309218433848032148ull, 15}, {5255101148688962965ull, 55}, {784530386183709463ull, 29}, {12724112379326185240ull, 41}, {3078130021364510233ull, 33}, {5792833011267379482ull, 49}, {18446744073709551615ull, 0}, {8789495792810759068ull, 25}, {9809552026335107740ull, 53}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6935161387918818980ull, 8}, {18446744073709551615ull, 0}, {8854381343033649318ull, 51}, {2783161425565058471ull, 10}, {4065902577701682344ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7873442382053928881ull, 13}, {17509849011186116785ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6064533617083031352ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9578121399438566843ull, 35}, {18446744073709551615ull, 0}, {3930994787075050813ull, 34}, {9483211823068989630ull, 38}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1061735976857914177ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2228349427768373958ull, 48}, {10525597742391453127ull, 9}, {8528041139767531208ull, 19}, {18446744073709551615ull, 0}, {17730612602998780490ull, 57}, {919911597814063947ull, 54}, {18446744073709551615ull, 0}, {9600954650394741325ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16775441919697322068ull, 39}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12778751976776651480ull, 11}, {17848355430324532185ull, 32}, {18446744073709551615ull, 0}, {2918542637195412955ull, 42}, {13003457108638579292ull, 43}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3568846733801557215ull, 14}, {14173837222217677664ull, 44}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4752114788659524325ull, 52}, {14015514739234771686ull, 21}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14803390641069418859ull, 58}, {18446744073709551615ull, 0}, {2941375888151587437ull, 4}, {15943996467068460269ull, 28}, {6587534006707254895ull, 31}, {7739116426202412656ull, 23}, {15734784568894920048ull, 36}, {14635558637114150258ull, 18}, {6602984835177365875ull, 45}, {4869857615985276020ull, 27}, {12902105974959694703ull, 59}, {17455209413735650545ull, 1}, {15321670583727670006ull, 30}, {3404470224630628343ull, 56}, {13938439269304993529ull, 46}, {12452411773510533370ull, 12}, {14449968376134455289ull, 50}, {15449074555053912956ull, 3}, {7255866119955889789ull, 47}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 4.2039e-45, .Count = 5}, {.Sum = 4.2039e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 9.80909e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 7}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 4.2039e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 5.60519e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {3, 5, 3, 2, 1, 5, 2, 2, 1, 1, 2, 1, 7, 1, 1, 7, 2, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1}
                    }
                },
                {
                    12606205885276083425ull,
                    {
                        .IndexHashViewer = {{14591795653440117248ull, 7}, {3812458928802352640ull, 15}, {14931585970071951136ull, 3}, {16031103881690819777ull, 2}, {18446744073709551615ull, 0}, {10918373804899154693ull, 14}, {2002444088171013702ull, 9}, {18446744073709551615ull, 0}, {11300006847281354472ull, 13}, {6619561440864924457ull, 1}, {3223795087593081450ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16939442126893167761ull, 11}, {18446744073709551615ull, 0}, {8668830525758017779ull, 12}, {18446744073709551615ull, 0}, {12990050366695140501ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16503206593760246744ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10128637724524112380ull, 8}, {13556881510278288029ull, 10}, {15649470839308619998ull, 4}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 8.40779e-45, .Count = 10}, {.Sum = 0, .Count = 6}, {.Sum = 1.4013e-45, .Count = 8}, {.Sum = 0, .Count = 11}, {.Sum = 2.8026e-45, .Count = 8}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 2.8026e-45, .Count = 13}, {.Sum = 0, .Count = 2}, {.Sum = 9.80909e-45, .Count = 6}, {.Sum = 1.4013e-45, .Count = 4}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {6, 10, 0, 6, 1, 8, 0, 11, 2, 8, 1, 5, 2, 13, 0, 2, 7, 6, 1, 4, 0, 3, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1}
                    }
                },
                {
                    12606205885276083426ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {3581428127016485793ull, 8}, {18446744073709551615ull, 0}, {17856817611009672707ull, 3}, {18446744073709551615ull, 0}, {14455983217430950149ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5967870314491345259ull, 1}, {2436149079269713547ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1236773280081879954ull, 7}, {16151796118569799858ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18336378346035991543ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8312525161425951098ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 36}, {.Sum = 2.94273e-44, .Count = 20}, {.Sum = 0, .Count = 12}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 36, 21, 20, 0, 12, 0, 2, 0, 4, 0, 3, 0, 1, 1, 0, 0, 1}
                    }
                },
                {
                    768791582259504189ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {4639344346382560065ull, 0}, {6768655180658783362ull, 24}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17601732372345076103ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9253120901934657613ull, 11}, {18446744073709551615ull, 0}, {12494120118534626831ull, 23}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15104639456961940114ull, 1}, {10170507820794899987ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17626484575418309142ull, 27}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10540782073424112667ull, 3}, {5606650437257072540ull, 18}, {18446744073709551615ull, 0}, {14838774965469232670ull, 17}, {18446744073709551615ull, 0}, {16546754159773737760ull, 20}, {8171065581604191777ull, 28}, {8376012298141440672ull, 10}, {17449294303896545953ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2686709533857156199ull, 15}, {8500597432574416232ull, 21}, {4462546031259207335ull, 25}, {12885436920358718506ull, 2}, {6984702425902276202ull, 12}, {17008555610512316647ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8962398883312995119ull, 7}, {10515720428538797616ull, 19}, {18446744073709551615ull, 0}, {11572918221740308402ull, 29}, {3982985296232888499ull, 6}, {646524893007459764ull, 22}, {582150902654165941ull, 9}, {5031364380791762038ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7009060838202480955ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5478643861907335871ull, 8}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 0, .Count = 7}, {.Sum = 1.4013e-44, .Count = 15}, {.Sum = 0, .Count = 19}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 7.00649e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 4, 1, 2, 0, 7, 10, 15, 0, 19, 0, 1, 0, 2, 0, 2, 0, 1, 0, 2, 5, 0, 2, 0, 0, 1, 0, 1, 0, 4, 0, 3, 0, 1, 0, 1, 0, 3, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1}
                    }
                },
                {
                    8628341152511840406ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {3581428127016485793ull, 2}, {1236773280081879954ull, 3}, {16151796118569799858ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13125720576600207402ull, 4}, {5967870314491345259ull, 1}, {9724886183021484844ull, 5}, {18446744073709551615ull, 0}, {13605281311626526238ull, 6}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 5.1848e-44, .Count = 2}, {.Sum = 1.82169e-44, .Count = 40}, {.Sum = 2.8026e-45, .Count = 4}},
                        .CtrTotal = {37, 2, 13, 40, 2, 4, 3}
                    }
                },
                {
                    13902559248212744134ull,
                    {
                        .IndexHashViewer = {{8975491433706742463ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14435487234778955461ull, 25}, {26794562384612742ull, 22}, {18446744073709551615ull, 0}, {4411634050168915016ull, 5}, {11361933621181601929ull, 1}, {15118949489711741514ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {488596013123191629ull, 7}, {2041917558348994126ull, 19}, {18446744073709551615ull, 0}, {3099115351550504912ull, 26}, {13955926499752636625ull, 6}, {6798076237643774482ull, 20}, {10555092106173914067ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4633306462361102487ull, 17}, {4428359745823853592ull, 29}, {16982002041722229081ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14612285549902308191ull, 15}, {18446744073709551615ull, 0}, {9142731084578380321ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {240279460452550314ull, 28}, {779318031744854123ull, 11}, {15286189140583379372ull, 16}, {4020317248344823341ull, 23}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6630836586772136624ull, 18}, {18446744073709551615ull, 0}, {3266355002422142770ull, 27}, {15927023829150890738ull, 21}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {896638510602221880ull, 0}, {2066979203234309177ull, 3}, {16388825279889469625ull, 14}, {18446744073709551615ull, 0}, {6364972095279429180ull, 12}, {18446744073709551615ull, 0}, {18348953501661188798ull, 10}, {18144006785123939903ull, 24}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 7.00649e-45, .Count = 18}, {.Sum = 0, .Count = 24}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 4.2039e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 7.00649e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {0, 1, 1, 1, 0, 1, 5, 18, 0, 24, 0, 7, 0, 1, 0, 2, 0, 1, 0, 2, 2, 0, 2, 0, 0, 2, 0, 1, 0, 1, 0, 3, 0, 1, 3, 0, 0, 1, 0, 2, 5, 0, 0, 1, 0, 3, 1, 0, 2, 0, 0, 1, 0, 2, 0, 2, 0, 1, 1, 0}
                    }
                },
                {
                    13902559248212744135ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16479676762461049221ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14314906987178377226ull, 24}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12130212770433783695ull, 4}, {18446744073709551615ull, 0}, {4054001010745510673ull, 30}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9965442995151111700ull, 13}, {16548297050623529236ull, 17}, {1889231235462838678ull, 20}, {18446744073709551615ull, 0}, {11147526993393187224ull, 28}, {18446744073709551615ull, 0}, {14555653527613724826ull, 8}, {12522231453186850331ull, 23}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10958843647676541603ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8794073872393869608ull, 35}, {8589127155856620713ull, 36}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11748579728051583916ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18384113673385397171ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17769648050045596984ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13820256289847569724ull, 2}, {13621749364805718972ull, 26}, {1878905203052656190ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11450539798027648834ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15254761925720908613ull, 10}, {18446744073709551615ull, 0}, {2398222922681060807ull, 29}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6227746613267076298ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18358311891254868174ull, 37}, {4062976837984404303ull, 11}, {3858030121447155408ull, 32}, {7449767364017353680ull, 33}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13653016638975931866ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14219847782559079394ull, 21}, {9089159255438104419ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16861309804843249000ull, 31}, {6719442763618183529ull, 18}, {16481986878556141930ull, 25}, {9655990399021251947ull, 34}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11030694858814915054ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2718670329607888375ull, 3}, {7719283207639011575ull, 27}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4940085441777621244ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 9}, {.Sum = 0, .Count = 8}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 0, .Count = 12}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 4}, {.Sum = 1.4013e-45, .Count = 4}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 1, 1, 0, 2, 1, 0, 5, 0, 7, 0, 9, 0, 8, 2, 2, 1, 5, 0, 12, 0, 2, 1, 0, 1, 0, 1, 0, 2, 4, 0, 1, 0, 2, 1, 4, 1, 4, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 2, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1}
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
