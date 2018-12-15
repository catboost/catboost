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
    unsigned int BinaryFeatureCount = 74;
    unsigned int TreeCount = 20;
    std::vector<std::vector<float>> FloatFeatureBorders = {
        {33.5, 45.5, 57.5, 61.5, 68.5},
        {51773, 59723, 126119, 553548.5},
        {13.5},
        {3280},
        {1738, 1881.5, 2189.5},
        {46.5}
    };
    std::vector<unsigned int> TreeDepth = {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
    std::vector<unsigned int> TreeSplits = {1, 78, 9, 10, 56, 53, 59, 9, 68, 3, 71, 72, 6, 85, 49, 14, 77, 67, 79, 5, 82, 83, 57, 74, 2, 16, 24, 45, 66, 60, 9, 44, 8, 84, 81, 55, 7, 85, 40, 16, 41, 12, 78, 8, 86, 20, 82, 15, 64, 30, 51, 26, 69, 3, 45, 50, 64, 61, 90, 52, 16, 63, 38, 32, 62, 76, 18, 42, 48, 28, 33, 3, 10, 75, 88, 78, 70, 37, 9, 21, 89, 75, 39, 5, 5, 80, 91, 19, 58, 4, 13, 87, 16, 45, 47, 43, 15, 10, 11, 5, 16, 34, 14, 22, 35, 36, 64, 31, 84, 0, 65, 73, 9, 54, 17, 23, 27, 25, 29, 46};
    std::vector<unsigned char> TreeSplitIdxs = {2, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 3, 1, 1, 1, 1, 2, 1, 4, 1, 1, 1, 255, 1, 1, 2, 1, 1, 4, 1, 1, 1, 2, 2, 1, 1, 4, 1, 1, 3, 2, 1, 1, 2, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 3, 1, 1, 1, 1, 1, 255, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1};
    std::vector<unsigned short> TreeSplitFeatureIndex = {0, 62, 2, 3, 44, 41, 47, 2, 53, 0, 56, 57, 1, 68, 38, 5, 61, 52, 62, 1, 65, 66, 45, 59, 0, 7, 14, 35, 51, 48, 2, 34, 1, 67, 64, 43, 1, 68, 30, 7, 31, 4, 62, 1, 69, 10, 65, 6, 49, 20, 39, 16, 54, 0, 35, 39, 49, 48, 72, 40, 7, 48, 28, 22, 48, 60, 8, 32, 37, 18, 23, 0, 3, 60, 71, 62, 55, 27, 2, 11, 72, 60, 29, 1, 1, 63, 73, 9, 46, 0, 4, 70, 7, 35, 37, 33, 6, 3, 4, 1, 7, 24, 5, 12, 25, 26, 49, 21, 67, 0, 50, 58, 2, 42, 7, 13, 17, 15, 19, 36};
    std::vector<unsigned char> TreeSplitXorMask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<unsigned int> CatFeaturesIndex = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<unsigned int> OneHotCatFeatureIndex = {9};
    std::vector<std::vector<int>> OneHotHashValues = {
        {-1291328762}
    };
    std::vector<std::vector<float>> CtrFeatureBorders = {
        {0.5f, 0.75f},
        {0.029411767f},
        {0.52022797f},
        {0.26579341f},
        {0.50773066f},
        {0.37749559f},
        {0.95085484f},
        {0.63317007f},
        {0.29146251f},
        {0.51108223f},
        {-0.0098029412f},
        {1.0008999f},
        {0.94529426f},
        {0.27624238f},
        {0.95919573f},
        {0.93417329f},
        {0.27482411f},
        {-0.0098029412f},
        {0.95919573f},
        {0.70002997f},
        {0.96515352f},
        {0.29418236f},
        {0.34988168f},
        {0.05838583f},
        {0.88244677f},
        {0.52022797f},
        {0.88451618f},
        {0.28645834f},
        {0.47135416f},
        {0.47500002f},
        {0.19117647f, 0.23529413f},
        {0.47973868f},
        {0.30810887f, 0.42648831f},
        {-0.0098029412f},
        {0.25318801f},
        {0.9383437f},
        {0.80071992f},
        {-0.0098029412f},
        {0.15519114f},
        {0.85135138f},
        {0.68614864f},
        {0.51249999f, 0.58749998f, 0.66250002f, 0.70000005f},
        {0.38725489f},
        {0.68919784f},
        {0.97309709f},
        {0.30659601f},
        {0.33336666f},
        {-0.0098029412f},
        {-0.0098029412f},
        {0.80071992f},
        {0.80071992f},
        {0.41202542f},
        {0.65625f},
        {0.014705883f, 0.14215687f},
        {0.9383437f},
        {0.578125f, 0.859375f},
        {-0.0098029412f},
        {0.019605882f},
        {0.94529432f},
        {1.0008999f},
        {0.6484375f},
        {0.46078432f},
        {0.90990901f},
        {0.91749161f},
        {0.05372807f},
        {0.6875f, 0.72916669f},
        {0.014705883f}
    };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[1280] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01928571425378323, 0.01200000010430813, 0.02769230678677559, 0.02210526168346405, 0, 0.004999999888241291, 0.006000000052154064, 0.007499999832361937, 0, 0, 0, 0, 0, 0, 0, 0,
        0.007310356944799423, 0, -3.749999814317562e-05, 0, 0.01944969967007637, 0.02751540020108223, 0.003624374745413661, 0.01489500049501657, 0, 0, 0, 0, 0, 0.01173473615199327, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.499999874951754e-07, 0, 0, 0, 0.003353543812409043, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007332037203013897, 0, 0.004870899952948093, 0, 0.02301016822457314, 0.01133750658482313, 0.02305624820291996, 0, 0, 0, 0.007332037203013897, 0, 0, -0.0002358727360842749, 0.0127288494259119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 2.778749887966114e-07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004790153820067644, 0.021476861089468, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.778749887966114e-07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0002341036888537928, 0.007000911049544811, 0.01596527919173241, 0.02494806423783302, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 2.757909101092082e-07, 0, 0, 0, -0.0003855169634334743, 0, -0.0006043236935511231, 0, 0, 0, -0.0008890924509614706, -0.000290763913653791, 0.02044864743947983, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005533884279429913, 0, 0, 0, 0, 0, 0.01079553086310625, 0.01138555351644754, 0, 0, 0, 0, 0.008730918169021606, -0.0003536756848916411, 0.02351492643356323, 0.01478569023311138, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0.009009950794279575, 0, 0, 0, 0, 0.01047955267131329, 0, 0.01074251346290112, 0, 0.006549893412739038, 0, 0, 0, 0, 0, 0.01506465207785368, 0.004555108025670052, 0.0214995089918375, 2.752712362052989e-06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -0.0006891480879858136, -0.0006831123027950525, 0.002532587619498372, 0, 0, 0, 0.01053583156317472, 0.01340770628303289, 0.01291835680603981, -0.0008755214512348175, 0.0128707168623805, 0.006532303988933563, 0.0181859340518713, 0.01853357627987862, 0.02083073742687702, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01464985031634569, 0.01465335488319397, 0.005536741111427546, 0, 0, 0, -0.0007610375760123134, 0, 0, 0, 0, 0, 0, 0, 0.006955127231776714, 0, 0, 2.530771325837122e-07, -0.0001295684051001444, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0.005009821150451899, 0, 0, 0, 0, 0, 0, 0.006620839703828096, 0.02210220322012901, 0, 0.006314720492810011, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00134974648244679, -0.0008378521306440234, 0, 0, 0, 0, 0, 0, 0.01231416780501604, 0.02082100510597229, 0, 0.006361422594636679, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0003167019167449325, 0, 0.004155758768320084, 0, -0.000792903418187052, 0, 0, 0, -0.0004587008152157068, 0, 0.002350897993892431, 0, -0.001474804827012122, 0.02351617068052292, 0.01556101720780134, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01014057919383049, -0.001142604975029826,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0005645044147968292, 0.006540014874190092, 0, 0.01045290939509869, 0, 0, 0, 0.01210945099592209, 0, 0, 0, 0, 0, 0, 0, 0, -0.000780347443651408, 0.006219714879989624, 0, 0.01121984422206879, 0, 0, 0.01606840454041958, 0.02258607558906078, 0, 0, 0, 0.002059697639197111,
        -0.001083423150703311, 0, 0, 0, -0.001294439774937928, 0.01125763077288866, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02174467965960503, 0, 0, 0, 0, 0, 0, 0, 0, -0.0005394094041548669, 0, 0, 0, -0.0007409536046907306, 0.01166711375117302, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01244788896292448, 0.01787867955863476, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -0.001033345819450915, 0, 0, -0.0007353964610956609, -0.001157999970018864, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.009978396818041801, -0.0004744013422168791, 0.01737690530717373, 0, 0, 0, 0.01960894465446472, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005792170763015747, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0008675626595504582, 0, 0.005921269301325083, -0.001245560240931809, 0, 0, 0.01212776266038418, -0.002071471884846687, 0, 0, 0.01053064502775669, 0, 0.005483436863869429, 0, 0.01919993944466114, -0.0004708433407358825, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.003398054046556354, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0005120777059346437, -0.0008021456887945533, 0, 0, 0, 0, 0, 0, -0.0008610559743829072, 0, 0.005442311055958271, 0, 0, 0.0043126055970788, 0.01135652232915163, 0, 0.009072449058294296, 0.006167028099298477, 0.01893139630556107, -0.001384882023558021,
        0, 0, 0, -0.0008545980672352016, 0, 0, 0, -0.002044330816715956, 0, 0, 0, 0.003485335968434811, 0, 0, -0.001310376217588782, 0.01504929549992085, 0, 0, 0, 0, 0, 0, 0.005506972782313824, 0.01166494376957417, 0, 0, 0, 0.00523595092818141, 0, 0, 0, 0.01927082426846027, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.003729796968400478, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0.004615754820406437, 0, 0, 0, 0.006164980120956898, -0.0006440034485422075, 0, 0, 0.001854024361819029, 0, 0, 0, 0.01686272956430912, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002609467599540949, -0.001049422658979893, 0, 0, 0.01684010773897171, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0114445723593235, -0.001957590458914638, 0, -0.000506809854414314, -0.001041551935486495, 0, 0, 0, 0, 0.003129880642518401, 0, 0, 0, 0, 0, 0, 0.01739063300192356, 0.01720453053712845, 0, -0.003156541148200631, 0.004927132744342089, 0.002337179146707058, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0.006116759032011032, 0, 0.01855400204658508, 0.01366560813039541, 0, 0, 0, 0, 0, 0, 0.004433132708072662, -0.0007501640357077122, 0, 0, 0, 0, 0, -0.0005030087777413428, 0.01064703427255154, -0.001125634298659861, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0.006904707290232182, 0, 0, 0, 0, 0, 0.008625911548733711, 0.00545165129005909, 0.005367254838347435, 0.004848805256187916, 0.01684729009866714, 0.004983052611351013, 0.01679569482803345, 0, 0, 0, -0.0005853155162185431, 0, 0, 0, 0, 0, 0, 0, -0.0002005159039981663, 0, 0, 0, 0.008737414143979549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, -0.0005809256108477712, 0, 0, 0, 0, 0, 0, 0.006005352828651667, 0.0041582016274333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005525874439626932, 0, 0, 0, 0, 0, 0, 0.006589753553271294, 0.01514835096895695, 0, 0.004515335895121098, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007012525573372841
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
                    {.BaseHash = 768791580653471478ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 8405694746487331134ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3},
                    .binarizedIndexes = {
                        {.BinIndex = 1, .CheckValueEqual = 0, .Value = 1}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 3001583246656978020ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3},
                    .binarizedIndexes = {
                        {.BinIndex = 1, .CheckValueEqual = 0, .Value = 4}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 9858262772220957788ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3},
                    .binarizedIndexes = {
                        {.BinIndex = 2, .CheckValueEqual = 0, .Value = 1}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 8360067430238894485ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
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
                    .transposedCatFeatureIndexes = {3, 4},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493224740170ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 17677952493224740170ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 4, 5, 6},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952491745844311ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 4, 5, 10},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952491745844307ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 4, 6},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 8405694747035287472ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
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
                    .transposedCatFeatureIndexes = {3, 4, 8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791582260355194ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 5},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493224740165ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
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
                    .transposedCatFeatureIndexes = {3, 5, 7, 10},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791580996533100ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
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
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 10041049327393282700ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
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
                    .transposedCatFeatureIndexes = {3, 6, 10},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791582296144838ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 7},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493224740167ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 7},
                    .binarizedIndexes = {
                        {.BinIndex = 2, .CheckValueEqual = 0, .Value = 1}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 10086676643306191396ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493224740166ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 17677952493224740166ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 17677952493224740166ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3, 8, 10},
                    .binarizedIndexes = {
                        {.BinIndex = 4, .CheckValueEqual = 0, .Value = 3}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 698704012360858366ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
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
                        {.BinIndex = 1, .CheckValueEqual = 0, .Value = 2}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 12606205885276083427ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {4, 5},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493261641996ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 10041049327410906820ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {4, 5, 7},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493035932493ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {4, 5, 8},
                    .binarizedIndexes = {
                        {.BinIndex = 0, .CheckValueEqual = 0, .Value = 1}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 2144226895133286195ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {4, 6},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493261641999ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {4, 7},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 10041049327410906822ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {4, 7, 8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493039731422ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 1}
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
                        {.BinIndex = 0, .CheckValueEqual = 0, .Value = 1}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 16302517178123948687ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5},
                    .binarizedIndexes = {
                        {.BinIndex = 0, .CheckValueEqual = 0, .Value = 3}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 18092064124022228906ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5},
                    .binarizedIndexes = {
                        {.BinIndex = 1, .CheckValueEqual = 0, .Value = 2},
                        {.BinIndex = 5, .CheckValueEqual = 0, .Value = 1}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 1148160072997466383ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5, 6},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493260528854ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 10041049327412019998ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5, 6, 7, 10},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 10041049327446511649ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
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
                    .transposedCatFeatureIndexes = {5, 6, 8, 10},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493297533872ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5, 7, 8},
                    .binarizedIndexes = {
                        {.BinIndex = 0, .CheckValueEqual = 0, .Value = 1}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 16302517177565331990ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
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
                    {.BaseHash = 768791580653471474ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {7},
                    .binarizedIndexes = {
                        {.BinIndex = 1, .CheckValueEqual = 0, .Value = 1}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 6317293569456956328ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {7},
                    .binarizedIndexes = {
                        {.BinIndex = 1, .CheckValueEqual = 0, .Value = 4},
                        {.BinIndex = 2, .CheckValueEqual = 0, .Value = 1}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 14067340392369360370ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {7, 8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 17677952493262230690ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {7, 8, 10},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791582178526348ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 768791580653471469ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 1},
                    {.BaseHash = 8405694746487331109ull, .BaseCtrType = ECatboostCPPExportModelCtrType::FeatureFreq, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {8},
                    .binarizedIndexes = {
                        {.BinIndex = 1, .CheckValueEqual = 0, .Value = 4}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 9858262772220957767ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {8},
                    .binarizedIndexes = {
                        {.BinIndex = 4, .CheckValueEqual = 0, .Value = 3}
                    },
                },
                .ModelCtrs = {
                    {.BaseHash = 698704010916237123ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 1}
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
                    698704010916237123ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14282620878612260867ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12420782654419932198ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9743509310306231593ull, 3}, {9551523844202795562ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10742856347075653999ull, 2}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 2.66247e-44, .Count = 61}, {.Sum = 0, .Count = 12}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}},
                        .CtrTotal = {19, 61, 0, 12, 1, 5, 0, 1, 2, 0}
                    }
                },
                {
                    698704012360858366ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {13998490437458917697ull, 17}, {18446744073709551615ull, 0}, {689933260599154435ull, 7}, {1850735554476988931ull, 11}, {18446744073709551615ull, 0}, {1559498166707215238ull, 13}, {7677328319700576455ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8050484539827566986ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13364628511990177423ull, 21}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7480393485741629395ull, 6}, {2453198338332991508ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5570950726739079329ull, 3}, {7235014621520909985ull, 4}, {11251207028491766434ull, 14}, {1101613637306031585ull, 1}, {7245212883257691941ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9267393403049452392ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5798630788952002222ull, 9}, {3740172899584216879ull, 8}, {2946231285291111150ull, 16}, {18446744073709551615ull, 0}, {438518358812245618ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9495073465262375285ull, 0}, {1197922782738558518ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15638672711214035641ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 1.26117e-44, .Count = 44}, {.Sum = 0, .Count = 8}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 2.8026e-45, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 5.60519e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {9, 44, 0, 8, 0, 2, 0, 1, 0, 1, 1, 3, 2, 5, 0, 1, 2, 0, 0, 2, 0, 4, 0, 2, 0, 1, 0, 1, 0, 1, 0, 2, 4, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0}
                    }
                },
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
                    768791580653471471ull,
                    {
                        .IndexHashViewer = {{2136296385601851904ull, 0}, {7428730412605434673ull, 2}, {9959754109938180626ull, 6}, {14256903225472974739ull, 1}, {8056048104805248435ull, 3}, {18446744073709551615ull, 0}, {12130603730978457510ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10789443546307262781ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 2.8026e-44, .Count = 73}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {20, 73, 0, 1, 0, 2, 0, 2, 0, 1, 1, 0, 1, 0}
                    }
                },
                {
                    768791580653471472ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13987540656699198946ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18089724839685297862ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10353740403438739754ull, 1}, {3922001124998993866ull, 2}, {13686716744772876732ull, 4}, {18293943161539901837ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 13}, {.Sum = 3.08286e-44, .Count = 20}, {.Sum = 0, .Count = 37}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 3}},
                        .CtrTotal = {0, 13, 22, 20, 0, 37, 0, 2, 0, 4, 0, 3}
                    }
                },
                {
                    768791580653471473ull,
                    {
                        .IndexHashViewer = {{7537614347373541888ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5903587924673389870ull, 2}, {18278593470046426063ull, 6}, {10490918088663114479ull, 3}, {18446744073709551615ull, 0}, {407784798908322194ull, 7}, {5726141494028968211ull, 1}, {1663272627194921140ull, 10}, {8118089682304925684ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15431483020081801594ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1403990565605003389ull, 5}, {3699047549849816830ull, 11}, {14914630290137473119ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 9.80909e-45, .Count = 3}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 5.60519e-45, .Count = 24}, {.Sum = 0, .Count = 3}, {.Sum = 5.60519e-45, .Count = 16}, {.Sum = 4.2039e-45, .Count = 16}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 5}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 2.8026e-45, .Count = 0}},
                        .CtrTotal = {7, 3, 1, 3, 4, 24, 0, 3, 4, 16, 3, 16, 0, 4, 0, 5, 1, 1, 0, 1, 0, 3, 2, 0}
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
                    768791580653471475ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {14452488454682494753ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1388452262538353895ull, 9}, {8940247467966214344ull, 1}, {4415016594903340137ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {41084306841859596ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8678739366408346384ull, 2}, {18446744073709551615ull, 0}, {4544226147037566482ull, 11}, {14256903225472974739ull, 8}, {16748601451484174196ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5913522704362245435ull, 6}, {1466902651052050075ull, 10}, {2942073219785550491ull, 12}, {15383677753867481021ull, 3}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 8.40779e-45, .Count = 10}, {.Sum = 2.8026e-45, .Count = 8}, {.Sum = 0, .Count = 6}, {.Sum = 2.8026e-45, .Count = 14}, {.Sum = 1.4013e-45, .Count = 4}, {.Sum = 2.8026e-45, .Count = 9}, {.Sum = 0, .Count = 11}, {.Sum = 0, .Count = 3}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 9.80909e-45, .Count = 6}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {6, 10, 2, 8, 0, 6, 2, 14, 1, 4, 2, 9, 0, 11, 0, 3, 1, 5, 7, 6, 0, 2, 1, 0, 0, 1}
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
                    768791580996533100ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {5161318391331352961ull, 27}, {18446744073709551615ull, 0}, {10972200855320061955ull, 17}, {16421466290801465092ull, 35}, {12345384918675283717ull, 1}, {13489896567580910725ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5604367391183497098ull, 29}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13200822063136670481ull, 24}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6948318308639175958ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2867287228037455385ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {627156190880834716ull, 33}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7175998370852098851ull, 5}, {18446744073709551615ull, 0}, {8941269328832924965ull, 36}, {14987821980444493478ull, 23}, {15155625736707958567ull, 38}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9722100677553049393ull, 2}, {18446744073709551615ull, 0}, {15215502042657416371ull, 0}, {18446744073709551615ull, 0}, {14999456095188427573ull, 42}, {13798428693015077045ull, 13}, {18446744073709551615ull, 0}, {9792434547781002552ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4881967207558201275ull, 6}, {13729498278813835708ull, 8}, {11524119110386893629ull, 21}, {18446744073709551615ull, 0}, {8958295223020228927ull, 30}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {563266583087864771ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7891362141945783750ull, 11}, {18446744073709551615ull, 0}, {4118321463043573832ull, 32}, {13233785125119014088ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16237524846887984845ull, 31}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1634155322308596306ull, 39}, {15703769826543348434ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9678362059253606747ull, 7}, {18446744073709551615ull, 0}, {1421097805173940445ull, 41}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15829145531557509728ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10330704939154537827ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13319597616803759207ull, 28}, {9906042121466529640ull, 9}, {3421155239616419303ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18019251360076866160ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3648835301829342196ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2912357214899525111ull, 34}, {18446744073709551615ull, 0}, {13811113861149180281ull, 37}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5796542738704019582ull, 40}, {11025566453509861631ull, 15}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 1.26117e-44, .Count = 14}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 15}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 5.60519e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {0, 3, 0, 3, 0, 1, 0, 3, 0, 2, 9, 14, 0, 3, 0, 1, 0, 1, 0, 7, 0, 2, 0, 2, 0, 15, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 4, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0}
                    }
                },
                {
                    768791582178526348ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {7595587498916228289ull, 0}, {7296736340428448578ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8786920001789086726ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12459205017610939725ull, 9}, {10179535390923972302ull, 1}, {18446744073709551615ull, 0}, {4138709075294936144ull, 5}, {9589946207393803089ull, 11}, {7310276580706835666ull, 6}, {6898253095073778579ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8241084747278133087ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17028616130751002788ull, 10}, {4365760407529725413ull, 16}, {4626577027618828517ull, 17}, {18446744073709551615ull, 0}, {14159357320533866152ull, 3}, {1949303683505127912ull, 8}, {1757318217401691881ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12589170367432031347ull, 18}, {13515339218119624820ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8974662647772726777ull, 20}, {3536070971160643258ull, 21}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10464846309133364925ull, 15}, {6105403837555590141ull, 7}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 14}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 10}, {.Sum = 2.38221e-44, .Count = 14}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 0, .Count = 18}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {0, 14, 0, 1, 0, 10, 17, 14, 0, 1, 0, 1, 2, 2, 0, 18, 0, 1, 0, 1, 0, 2, 1, 0, 0, 2, 0, 1, 0, 2, 0, 4, 0, 1, 0, 2, 0, 1, 1, 0, 0, 1, 1, 0}
                    }
                },
                {
                    768791582220620454ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {8729380230485332353ull, 7}, {9977784445157143938ull, 44}, {10895282230322158083ull, 41}, {11761827888484752132ull, 2}, {12615796933138713349ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16179912168674497673ull, 12}, {11168821652508931466ull, 34}, {18446744073709551615ull, 0}, {10475827037446056716ull, 27}, {14750448552479345421ull, 20}, {16495354791085375886ull, 28}, {10135854469738880143ull, 48}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12233930600554179099ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3208547115700824735ull, 23}, {18277252057819687584ull, 43}, {11380194846102552353ull, 29}, {18446744073709551615ull, 0}, {14030234294679959587ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14597065438263107115ull, 15}, {1433532977505263916ull, 11}, {17401263168096565421ull, 31}, {15971769898056770222ull, 8}, {2808237437298927023ull, 47}, {1256940733300575664ull, 1}, {18446744073709551615ull, 0}, {9689317982178109362ull, 42}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6069959757871852856ull, 6}, {7318363972543664441ull, 10}, {18446744073709551615ull, 0}, {5876843741908031419ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5859361731318044735ull, 16}, {6636790901455000384ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10322519660234049603ull, 32}, {7078520115819519811ull, 37}, {8011495361248663491ull, 33}, {9259899575920475076ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9323661606633862475ull, 40}, {18146214905751188428ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3907755348917795664ull, 49}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8720774373489072856ull, 14}, {6896376012912388953ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10662309976865883491ull, 24}, {9111013272867532132ull, 39}, {10359417487539343717ull, 46}, {17543390521745065830ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13529097553057277673ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2350285447658856812ull, 3}, {16689654767293439084ull, 38}, {18446744073709551615ull, 0}, {6176161755113003119ull, 21}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {346329515989321719ull, 45}, {6353480505666359544ull, 17}, {4263979015577900536ull, 30}, {9547190628529608953ull, 35}, {9583115984936959099ull, 36}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 8.40779e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 9.80909e-45, .Count = 1}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 3, 0, 1, 0, 3, 0, 3, 0, 1, 1, 1, 6, 1, 2, 4, 0, 1, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 0, 4, 0, 1, 0, 3, 0, 1, 0, 1, 0, 1, 2, 5, 0, 1, 0, 2, 7, 1, 0, 5, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 1, 0, 1, 0, 2, 0, 1, 1, 0, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1}
                    }
                },
                {
                    768791582259504189ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {4639344346382560065ull, 9}, {6768655180658783362ull, 29}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17601732372345076103ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9253120901934657613ull, 14}, {18446744073709551615ull, 0}, {12494120118534626831ull, 23}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15104639456961940114ull, 16}, {10170507820794899987ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17626484575418309142ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10540782073424112667ull, 5}, {5606650437257072540ull, 15}, {18446744073709551615ull, 0}, {14838774965469232670ull, 11}, {18446744073709551615ull, 0}, {16546754159773737760ull, 24}, {8171065581604191777ull, 12}, {8376012298141440672ull, 26}, {17449294303896545953ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2686709533857156199ull, 1}, {8500597432574416232ull, 2}, {4462546031259207335ull, 28}, {12885436920358718506ull, 0}, {6984702425902276202ull, 27}, {17008555610512316647ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8962398883312995119ull, 8}, {10515720428538797616ull, 21}, {18446744073709551615ull, 0}, {11572918221740308402ull, 19}, {3982985296232888499ull, 4}, {646524893007459764ull, 7}, {582150902654165941ull, 18}, {5031364380791762038ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7009060838202480955ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5478643861907335871ull, 10}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 19}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-44, .Count = 15}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 3}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 7.00649e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {0, 7, 0, 3, 0, 2, 0, 19, 0, 2, 10, 15, 0, 1, 0, 1, 0, 2, 0, 4, 0, 1, 0, 1, 2, 0, 0, 1, 2, 0, 0, 3, 1, 2, 0, 1, 0, 2, 0, 1, 0, 1, 0, 2, 0, 4, 1, 0, 0, 1, 0, 1, 5, 0, 0, 1, 0, 1, 1, 0}
                    }
                },
                {
                    768791582260355194ull,
                    {
                        .IndexHashViewer = {{742917631823268096ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17111429953860281092ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16848063936164177159ull, 11}, {18446744073709551615ull, 0}, {14625804321994992265ull, 4}, {14741713462040360202ull, 14}, {14946660178577609097ull, 33}, {18446744073709551615ull, 0}, {11876999423938579341ull, 23}, {18446744073709551615ull, 0}, {12256087830175071375ull, 38}, {5042296304595411984ull, 37}, {18446744073709551615ull, 0}, {227505594222800146ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5195152315211825698ull, 21}, {12797353935447784739ull, 15}, {8398044750651334563ull, 19}, {18446744073709551615ull, 0}, {11008564089078647846ull, 24}, {7502454802225285414ull, 45}, {7240525494885412904ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17371395578289429165ull, 30}, {17656687951877322286ull, 20}, {6444706705540820399ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5287187449774898740ull, 29}, {4279936930258148404ull, 42}, {15823768782370826038ull, 18}, {971926617098238774ull, 36}, {4074990213720899509ull, 27}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14854813307269962943ull, 32}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9257357414293324624ull, 16}, {1473640651620517713ull, 25}, {16894328577540058706ull, 40}, {8993991396597220691ull, 7}, {5923684169112225364ull, 1}, {5345535050213463508ull, 3}, {6771731782428035797ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4022926884371622873ull, 43}, {15356527341792241882ull, 10}, {18446744073709551615ull, 0}, {7936243299912755676ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13191757566509569887ull, 35}, {12986810849972320992ull, 9}, {17787251304587411297ull, 6}, {5157045534051365345ull, 44}, {6896619440991342435ull, 34}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12375346307733773417ull, 22}, {13223393921049583850ull, 41}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10005629815913852527ull, 39}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17833197029028008052ull, 31}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17037378239683415547ull, 28}, {18446744073709551615ull, 0}, {16579142367105506429ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 4.2039e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 4.2039e-45, .Count = 9}, {.Sum = 1.4013e-45, .Count = 17}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 2.8026e-45, .Count = 9}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {3, 1, 0, 1, 0, 1, 0, 3, 3, 9, 1, 17, 0, 1, 0, 1, 1, 1, 0, 3, 2, 9, 0, 1, 0, 1, 0, 1, 2, 1, 0, 4, 0, 1, 0, 2, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 2, 0, 0, 1, 1, 0, 0, 1}
                    }
                },
                {
                    768791582296144838ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14266509156035635599ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {237849362515829524ull, 5}, {12101739380752963604ull, 30}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16691949913215576730ull, 10}, {14658527838788702235ull, 23}, {14775490868423431964ull, 27}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10774718434413136810ull, 19}, {6957686145156610347ull, 9}, {13884876113653435820ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11720106338370763825ull, 24}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1459200361937897272ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15758045750407570876ull, 2}, {17741174660364776893ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17391058311322760517ull, 14}, {5530283389519412293ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8364042998868928202ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2047864203147168462ull, 32}, {6199273223586256207ull, 29}, {5994326507049007312ull, 21}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15789313024577783770ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {730566066947466973ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16356144168160931298ull, 17}, {11225455641039956323ull, 0}, {18446744073709551615ull, 0}, {17730848627954594405ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8855739149220035433ull, 11}, {171539190448442218ull, 31}, {11792286784623103851ull, 12}, {4091279200649295081ull, 22}, {14528826653659518317ull, 28}, {13166991244416766958ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {602169291666733429ull, 33}, {18446744073709551615ull, 0}, {4854966715209740279ull, 3}, {9855579593240863479ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7076381827379473148ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 4.2039e-45, .Count = 3}, {.Sum = 1.4013e-45, .Count = 7}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 14}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 1.4013e-45, .Count = 7}, {.Sum = 0, .Count = 9}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 1.4013e-45, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 4.2039e-45, .Count = 4}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {3, 3, 1, 7, 0, 1, 0, 5, 0, 14, 1, 3, 1, 7, 0, 9, 1, 2, 0, 1, 1, 5, 1, 4, 0, 1, 0, 1, 0, 2, 3, 4, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 1, 0, 0, 1, 1, 0}
                    }
                },
                {
                    1148160072997466383ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6011932206351584034ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15744222165914032517ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {648103084612310536ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14363750123912396847ull, 8}, {2782754703952942608ull, 5}, {5433569319570162608ull, 3}, {1131894589287130066ull, 10}, {12515044663515391091ull, 12}, {7217896454495632755ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16498401743253028919ull, 1}, {702472285160697303ull, 2}, {14847541628587216377ull, 7}, {10434762244723145786ull, 9}, {10743029240761049339ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 1}, {.Sum = 1.68156e-44, .Count = 19}, {.Sum = 0, .Count = 33}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 1.26117e-44, .Count = 1}, {.Sum = 0, .Count = 11}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 3}},
                        .CtrTotal = {0, 1, 12, 19, 0, 33, 0, 3, 0, 1, 9, 1, 0, 11, 0, 3, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 3}
                    }
                },
                {
                    2144226895133286195ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13715390106631652101ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12502692833016322569ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9633434022799185933ull, 14}, {4913143281939397646ull, 37}, {8405633602493730575ull, 39}, {6327393565113416592ull, 34}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {28371208966466454ull, 47}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15322382680967285401ull, 21}, {15605856472458881434ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12453123870750148765ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12530354008623705895ull, 31}, {18446744073709551615ull, 0}, {11564173954830761897ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5331759292516153391ull, 32}, {11634500223657421744ull, 42}, {1227246904689986353ull, 29}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15759531511022347321ull, 33}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5587375217000051388ull, 4}, {15103938486367368765ull, 41}, {7199669354349681854ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2232265062098166593ull, 43}, {18446744073709551615ull, 0}, {966771518167475907ull, 26}, {7722026836340683460ull, 8}, {3406608024258179780ull, 44}, {5664605354873608518ull, 17}, {11713716441591431238ull, 38}, {9391638768282176711ull, 20}, {537349214041043144ull, 28}, {18362891160282708680ull, 35}, {14714737143401074507ull, 25}, {6522379958065040075ull, 0}, {1343315146178779725ull, 23}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6544550653118369744ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17686138891251126739ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16797188975331739871ull, 40}, {17008849494966484960ull, 6}, {18446744073709551615ull, 0}, {17233789867425787874ull, 46}, {4244261432549852771ull, 36}, {10372841451957903460ull, 13}, {2468572319940216549ull, 27}, {14364531057208651238ull, 22}, {13312426531001699943ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6982619407181965933ull, 45}, {18446744073709551615ull, 0}, {3917966052405608559ull, 24}, {4759468243375931759ull, 30}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1890209433158795123ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1813453618708904439ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12955041856841661434ull, 11}, {14287668101921523323ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 5.60519e-45, .Count = 4}, {.Sum = 9.80909e-45, .Count = 0}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 8}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 5.60519e-45, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 2, 0, 1, 0, 7, 0, 2, 0, 1, 4, 4, 7, 0, 0, 4, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 5, 0, 8, 0, 1, 0, 1, 0, 1, 4, 2, 0, 1, 0, 1, 0, 2, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1}
                    }
                },
                {
                    3001583246656978020ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12337359831519453058ull, 4}, {18446744073709551615ull, 0}, {6973539969458659060ull, 0}, {13860744542689514389ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16503206593760246744ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2242442935049193755ull, 6}, {18446744073709551615ull, 0}, {8193958724117795869ull, 1}, {10924139913308365886ull, 5}, {14687079002600389023ull, 2}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 1.26117e-44, .Count = 57}, {.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 6}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 4.2039e-45, .Count = 6}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 7.00649e-45, .Count = 0}},
                        .CtrTotal = {9, 57, 0, 4, 1, 6, 1, 5, 3, 6, 2, 0, 1, 1, 5, 0}
                    }
                },
                {
                    4544184825161771334ull,
                    {
                        .IndexHashViewer = {{158999094665252608ull, 45}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8367117909645166341ull, 41}, {18446744073709551615ull, 0}, {8702450991728868615ull, 27}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6262126705356135693ull, 48}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3752585126949001232ull, 36}, {14946871803847873040ull, 50}, {13913159826293879825ull, 53}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14375335943966472213ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6045632691652965145ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12440274563092365353ull, 52}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6619783738664723247ull, 13}, {18446744073709551615ull, 0}, {4905776570447084337ull, 56}, {18446744073709551615ull, 0}, {8130996685331913523ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14732992977275059767ull, 31}, {18446744073709551615ull, 0}, {2585940615564665401ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3128199045106796348ull, 55}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7735182141653903170ull, 17}, {17619157723446594371ull, 11}, {11241408283717132868ull, 4}, {13574756925474066500ull, 59}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4146295242583377226ull, 18}, {18314399080722540106ull, 39}, {18446744073709551615ull, 0}, {3172219588709525325ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {174679536037619032ull, 24}, {17343663287863212120ull, 37}, {18446744073709551615ull, 0}, {79769959668041819ull, 7}, {16685972765223635547ull, 38}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16843846084406882659ull, 34}, {518059473081761380ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6182897570358924904ull, 44}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6510311249307667563ull, 40}, {12533704194145800556ull, 60}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11009716383731513464ull, 61}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13808447517377610367ull, 2}, {18446744073709551615ull, 0}, {7824087488121779841ull, 51}, {7469564332260859522ull, 16}, {13795416998968128130ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12443445925107193993ull, 28}, {6540554603667512458ull, 5}, {18446744073709551615ull, 0}, {18123001185196525196ull, 19}, {18446744073709551615ull, 0}, {8051767550334702734ull, 6}, {2891096590926338447ull, 43}, {18446744073709551615ull, 0}, {6116316705811167633ull, 63}, {9269497864691699089ull, 64}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8490057063748422295ull, 62}, {18446744073709551615ull, 0}, {4919885718948249ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {522424025960426143ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17603342537294645418ull, 23}, {16803678464185371818ull, 54}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7617439314328151475ull, 33}, {18446744073709551615ull, 0}, {3320670515925237429ull, 58}, {13992388961291090614ull, 26}, {18446744073709551615ull, 0}, {1385219671401702328ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6662268631026693053ull, 29}, {16764616949409671870ull, 22}, {6124861826650175934ull, 32}, {9498428910012038848ull, 42}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7162655135050725840ull, 30}, {12072581429775906513ull, 57}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11977671853406329300ull, 49}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {197512786993793514ull, 46}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10586639264769695215ull, 35}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {635802300407513075ull, 1}, {18446744073709551615ull, 0}, {6475377084227676405ull, 21}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12735534006750400250ull, 47}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 6}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 2, 0, 1, 0, 0, 3, 0, 2, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 0, 2, 0, 1, 0, 3, 0, 1, 0, 1, 0, 1, 1, 0, 0, 7, 0, 1, 2, 1, 0, 3, 0, 2, 1, 0, 0, 1, 0, 6, 0, 1, 0, 1, 0, 1, 0, 4, 0, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 2, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1}
                    }
                },
                {
                    4544184825393173621ull,
                    {
                        .IndexHashViewer = {{11772109559350781439ull, 4}, {18446744073709551615ull, 0}, {12337359831519453058ull, 8}, {18446744073709551615ull, 0}, {3462861689708330564ull, 2}, {6193042878898900581ull, 6}, {9955981968190923718ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7606262797109987753ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6973539969458659060ull, 0}, {13860744542689514389ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2242442935049193755ull, 1}, {9129647508280049084ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14687079002600389023ull, 7}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 7.00649e-45, .Count = 5}, {.Sum = 7.00649e-45, .Count = 53}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 5}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 2.8026e-45, .Count = 5}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 4.2039e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}},
                        .CtrTotal = {5, 5, 5, 53, 0, 4, 0, 5, 1, 5, 2, 5, 2, 0, 1, 1, 1, 1, 3, 0, 2, 0}
                    }
                },
                {
                    6317293569456956328ull,
                    {
                        .IndexHashViewer = {{18034029854971645104ull, 4}, {10934650013725255009ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5844492600899280932ull, 0}, {1601191413561926516ull, 5}, {1034166431492604838ull, 7}, {13302932820562179799ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13957701839509617452ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5765263465902070143ull, 3}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 3.08286e-44, .Count = 5}, {.Sum = 1.96182e-44, .Count = 35}, {.Sum = 2.94273e-44, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {22, 5, 14, 35, 21, 2, 1, 1}
                    }
                },
                {
                    8360067430238894485ull,
                    {
                        .IndexHashViewer = {{11772109559350781439ull, 4}, {18446744073709551615ull, 0}, {12337359831519453058ull, 6}, {18446744073709551615ull, 0}, {3462861689708330564ull, 2}, {6193042878898900581ull, 11}, {9955981968190923718ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7606262797109987753ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6973539969458659060ull, 0}, {13860744542689514389ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2242442935049193755ull, 1}, {9129647508280049084ull, 10}, {18446744073709551615ull, 0}, {10924139913308365886ull, 7}, {14687079002600389023ull, 8}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 5.60519e-45, .Count = 2}, {.Sum = 8.40779e-45, .Count = 56}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 5}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 5.60519e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {4, 2, 6, 56, 0, 4, 0, 5, 1, 5, 1, 5, 2, 1, 1, 0, 1, 1, 1, 0, 4, 0, 1, 0}
                    }
                },
                {
                    8405694746487331109ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8473802870189803490ull, 0}, {7071392469244395075ull, 2}, {18446744073709551615ull, 0}, {8806438445905145973ull, 3}, {619730330622847022ull, 1}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 1.14906e-43, .Count = 12}, {.Sum = 8.40779e-45, .Count = 1}},
                        .CtrTotal = {82, 12, 6, 1}
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
                    8405694746487331128ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13987540656699198946ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18089724839685297862ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10353740403438739754ull, 1}, {3922001124998993866ull, 2}, {13686716744772876732ull, 4}, {18293943161539901837ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 1.82169e-44, .Count = 42}, {.Sum = 5.1848e-44, .Count = 2}, {.Sum = 5.60519e-45, .Count = 3}},
                        .CtrTotal = {13, 42, 37, 2, 4, 3}
                    }
                },
                {
                    8405694746487331129ull,
                    {
                        .IndexHashViewer = {{7537614347373541888ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5903587924673389870ull, 2}, {18278593470046426063ull, 6}, {10490918088663114479ull, 3}, {18446744073709551615ull, 0}, {407784798908322194ull, 7}, {5726141494028968211ull, 1}, {1663272627194921140ull, 10}, {8118089682304925684ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15431483020081801594ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1403990565605003389ull, 5}, {3699047549849816830ull, 11}, {14914630290137473119ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 1.4013e-44, .Count = 4}, {.Sum = 3.92364e-44, .Count = 3}, {.Sum = 2.8026e-44, .Count = 19}, {.Sum = 5.60519e-45, .Count = 5}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 4.2039e-45, .Count = 2}},
                        .CtrTotal = {10, 4, 28, 3, 20, 19, 4, 5, 2, 1, 3, 2}
                    }
                },
                {
                    8405694746487331131ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {14452488454682494753ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1388452262538353895ull, 9}, {8940247467966214344ull, 1}, {4415016594903340137ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {41084306841859596ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8678739366408346384ull, 2}, {18446744073709551615ull, 0}, {4544226147037566482ull, 11}, {14256903225472974739ull, 8}, {16748601451484174196ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5913522704362245435ull, 6}, {1466902651052050075ull, 10}, {2942073219785550491ull, 12}, {15383677753867481021ull, 3}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 2.24208e-44, .Count = 10}, {.Sum = 8.40779e-45, .Count = 16}, {.Sum = 7.00649e-45, .Count = 11}, {.Sum = 1.54143e-44, .Count = 3}, {.Sum = 8.40779e-45, .Count = 13}, {.Sum = 2.8026e-45, .Count = 1}},
                        .CtrTotal = {16, 10, 6, 16, 5, 11, 11, 3, 6, 13, 2, 1, 1}
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
                },
                {
                    8405694747035287472ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5550471546432006402ull, 3}, {18446744073709551615ull, 0}, {6923808778444297988ull, 62}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3884426539713262344ull, 7}, {10026079346208831497ull, 33}, {4554092286624377098ull, 14}, {15561463838010784776ull, 39}, {9762713328512727564ull, 17}, {13263798533224195593ull, 58}, {18446744073709551615ull, 0}, {7656362854388910607ull, 42}, {10567018046708528912ull, 5}, {12197062333452576271ull, 45}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2888468852575381529ull, 43}, {4746381528034559258ull, 64}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16498818186995172646ull, 29}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10756943142184875818ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2785254884385905453ull, 28}, {8678711390512337198ull, 0}, {3915498648679126062ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6308994898692416308ull, 20}, {18446744073709551615ull, 0}, {2385920430254357046ull, 23}, {18446744073709551615ull, 0}, {11334713873079255864ull, 18}, {15641330396316250425ull, 61}, {1654888376477695802ull, 52}, {8738418174719376443ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7478463493707023934ull, 57}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14719505286835191105ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14181145105934505812ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9808977969888609111ull, 59}, {18446744073709551615ull, 0}, {16629381539705261657ull, 37}, {18446744073709551615ull, 0}, {1286748774927000155ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10985952127459846496ull, 55}, {18446744073709551615ull, 0}, {329295446230924386ull, 44}, {18446744073709551615ull, 0}, {3088935131963118180ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {340130233906705256ull, 34}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15092248919771403119ull, 46}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11487666161982125435ull, 36}, {10237073769620226684ull, 21}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13354581589218497930ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7540453714343542670ull, 48}, {9718970433392718479ull, 31}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9824797691387988883ull, 66}, {18446744073709551615ull, 0}, {6911151303317231253ull, 54}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6676118443748996249ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17806100171598922420ull, 24}, {13469950896506196404ull, 30}, {3819658151760367797ull, 40}, {14830740818213535159ull, 8}, {18446744073709551615ull, 0}, {3823369266827127481ull, 56}, {15436383679779001530ull, 38}, {12081935920157122235ull, 32}, {12066920417729042876ull, 65}, {4218443878319275962ull, 47}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15822971939568652736ull, 16}, {18446744073709551615ull, 0}, {8402687897689166530ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2157745832778131912ull, 6}, {2478601689360748744ull, 50}, {2087048262131918536ull, 51}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13023625850168584153ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6649643201759363293ull, 60}, {18446744073709551615ull, 0}, {8271176734140792287ull, 27}, {4001663869940894431ull, 35}, {18446744073709551615ull, 0}, {12065524156167434210ull, 53}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9316719258111021286ull, 63}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6422094743922019049ull, 49}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14244201306455592686ull, 1}, {12423392290033511662ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1260484745415910654ull, 41}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 4.2039e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 7.00649e-45, .Count = 1}, {.Sum = 4.2039e-45, .Count = 4}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 4.2039e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 5.60519e-45, .Count = 2}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 4.2039e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {3, 1, 1, 1, 5, 1, 3, 4, 2, 1, 1, 5, 1, 2, 3, 1, 2, 1, 1, 2, 2, 1, 4, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1}
                    }
                },
                {
                    9858262772220957767ull,
                    {
                        .IndexHashViewer = {{17151879688829397503ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14282620878612260867ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12420782654419932198ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9743509310306231593ull, 4}, {9551523844202795562ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10742856347075653999ull, 3}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 2.94273e-44, .Count = 60}, {.Sum = 0, .Count = 11}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {21, 60, 0, 11, 0, 1, 1, 5, 0, 1, 0, 1}
                    }
                },
                {
                    9858262772220957788ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3462861689708330564ull, 1}, {6193042878898900581ull, 6}, {9955981968190923718ull, 3}, {6973539969458659060ull, 2}, {18446744073709551615ull, 0}, {7606262797109987753ull, 5}, {18446744073709551615ull, 0}, {2242442935049193755ull, 0}, {9129647508280049084ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11772109559350781439ull, 4}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 1.4013e-44, .Count = 56}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 6}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 4.2039e-45, .Count = 6}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 7.00649e-45, .Count = 0}},
                        .CtrTotal = {10, 56, 0, 4, 0, 2, 1, 6, 1, 5, 3, 6, 2, 0, 5, 0}
                    }
                },
                {
                    10041049327393282700ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {15869504672553169153ull, 18}, {3630895197587547650ull, 15}, {18446744073709551615ull, 0}, {6657459529952533892ull, 5}, {18069894976246263428ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12403109157674544908ull, 2}, {7581495141437254476ull, 16}, {18446744073709551615ull, 0}, {544300312823816335ull, 12}, {8994715059648341648ull, 19}, {18446744073709551615ull, 0}, {7582268711114204562ull, 10}, {9997066275032783314ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8155965639411439190ull, 7}, {18446744073709551615ull, 0}, {17626120688916674776ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5135391889252992221ull, 21}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11422884619242973793ull, 14}, {3129976559056829986ull, 26}, {10518099770818402979ull, 24}, {11182690403015408099ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2283527241891053351ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10921182301457540139ull, 3}, {4851313952246684459ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7647347103951847349ull, 11}, {5184516154834744246ull, 27}, {18446744073709551615ull, 0}, {1764719067482953144ull, 9}, {6066581188437978489ull, 22}, {8257839345965546298ull, 17}, {12150488944147554235ull, 23}, {16694931389731688508ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9376384394070575999ull, 20}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 9.80909e-45, .Count = 8}, {.Sum = 1.4013e-45, .Count = 6}, {.Sum = 1.96182e-44, .Count = 5}, {.Sum = 1.4013e-44, .Count = 9}, {.Sum = 4.2039e-45, .Count = 1}, {.Sum = 8.40779e-45, .Count = 6}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 7}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {7, 8, 1, 6, 14, 5, 10, 9, 3, 1, 6, 6, 1, 1, 2, 7, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1}
                    }
                },
                {
                    10041049327410906820ull,
                    {
                        .IndexHashViewer = {{16259707375369223360ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13847085545544291780ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7654620248348398600ull, 0}, {18446744073709551615ull, 0}, {9243796653651753418ull, 8}, {18446744073709551615ull, 0}, {1681026541770505292ull, 15}, {1292491219513334285ull, 20}, {13677090684479491854ull, 14}, {6494991755595340494ull, 17}, {7494438315637327440ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18410271455579776277ull, 3}, {6336919059871405781ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9974519673449003035ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5899717636280359390ull, 18}, {18446744073709551615ull, 0}, {15904544917366469984ull, 24}, {18446744073709551615ull, 0}, {862592111642406882ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18161161563956788133ull, 6}, {18446744073709551615ull, 0}, {3340544229935902247ull, 21}, {18446744073709551615ull, 0}, {14827488318775688873ull, 1}, {15675535932091499306ull, 5}, {18446744073709551615ull, 0}, {15230422751883885548ull, 26}, {18446744073709551615ull, 0}, {1662085889209686126ull, 27}, {18446744073709551615ull, 0}, {1062699037197581552ull, 2}, {14072903496117963889ull, 23}, {18446744073709551615ull, 0}, {15434641073738489523ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14277121817972567864ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18160464660109825851ull, 11}, {16406258951888748923ull, 25}, {17480885798804750972ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 4.2039e-45, .Count = 2}, {.Sum = 1.54143e-44, .Count = 3}, {.Sum = 1.4013e-45, .Count = 11}, {.Sum = 9.80909e-45, .Count = 5}, {.Sum = 1.12104e-44, .Count = 12}, {.Sum = 1.54143e-44, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 2.8026e-45, .Count = 4}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {3, 2, 11, 3, 1, 11, 7, 5, 8, 12, 11, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2, 2, 4, 1, 1}
                    }
                },
                {
                    10041049327410906822ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14339393822756684802ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9042729150784344201ull, 33}, {18446744073709551615ull, 0}, {1434197551787351435ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15505904404980462094ull, 9}, {17132136727440490127ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2690081920877379861ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1532562665111458202ull, 31}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14397741423195249570ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18052491238949695525ull, 29}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5961989641064476328ull, 26}, {777303952308747305ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15890374780837199661ull, 35}, {16738422394153010094ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11699844009042731185ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5970980967522331961ull, 38}, {1590910265550022970ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11601902557128801344ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14909972007605802568ull, 28}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5867491582615089871ull, 22}, {2743913003832016080ull, 23}, {18446744073709551615ull, 0}, {7716892253132515538ull, 27}, {18446744073709551615ull, 0}, {8557324777698838228ull, 18}, {18446744073709551615ull, 0}, {4383219007951416278ull, 14}, {5231266621267226711ull, 4}, {10600672353715374294ull, 21}, {7399805521932916569ull, 30}, {18446744073709551615ull, 0}, {2482461723210813787ull, 37}, {2164920571584601052ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7725883579590371171ull, 16}, {16967431379427980772ull, 32}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4392210334409271911ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13356805169196840554ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10871179537331551727ull, 7}, {18446744073709551615ull, 0}, {3402816234720019185ull, 17}, {2724972351271196914ull, 36}, {8122374639275138803ull, 24}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11414809869912342394ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15496913078522606461ull, 2}, {18446744073709551615ull, 0}, {17469145413950259711ull, 34}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 1.4013e-45, .Count = 1}, {.Sum = 5.60519e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 11}, {.Sum = 7.00649e-45, .Count = 2}, {.Sum = 1.26117e-44, .Count = 2}, {.Sum = 1.4013e-45, .Count = 6}, {.Sum = 8.40779e-45, .Count = 5}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.26117e-44, .Count = 2}, {.Sum = 9.80909e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 4.2039e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {1, 1, 4, 1, 1, 11, 5, 2, 9, 2, 1, 6, 6, 5, 2, 2, 1, 1, 9, 2, 7, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 3, 1, 1, 1, 1}
                    }
                },
                {
                    10041049327412019998ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17985374731566054150ull, 22}, {18446744073709551615ull, 0}, {4969880389554839688ull, 30}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1883285504791108373ull, 14}, {14139902777924824981ull, 23}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17540248381108153753ull, 15}, {18446744073709551615ull, 0}, {2120068639763588379ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1277857586923739550ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9915512646490226338ull, 9}, {18446744073709551615ull, 0}, {5780999427119446436ull, 37}, {15493676505554854693ull, 31}, {14453653496344422438ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3622512433858345389ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9415440463389949361ull, 35}, {18446744073709551615ull, 0}, {15689261734764374707ull, 7}, {17838331352489460532ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18403625549429831228ull, 24}, {18446744073709551615ull, 0}, {16192880425411659454ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6383791411268594626ull, 3}, {18033916581698980546ull, 33}, {18446744073709551615ull, 0}, {11961955270333222981ull, 13}, {18446744073709551615ull, 0}, {11191788834073534919ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17618698769621849933ull, 28}, {5730630563994427981ull, 32}, {16620451033949360975ull, 19}, {647125264645798733ull, 38}, {7150295984444125389ull, 29}, {18446744073709551615ull, 0}, {12157540499542742995ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1072059942279220057ull, 34}, {10177020748048094298ull, 1}, {18446744073709551615ull, 0}, {9494950831378731228ull, 8}, {18446744073709551615ull, 0}, {518361807174415198ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {592499207252901221ull, 18}, {4098784705883188966ull, 36}, {10062654256758136807ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3618574749222493677ull, 21}, {18446744073709551615ull, 0}, {13088729798727729263ull, 27}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2625225542620233849ull, 20}, {6645299512826462586ull, 2}, {5651789874985220091ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 5.60519e-45, .Count = 4}, {.Sum = 7.00649e-45, .Count = 3}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 9.80909e-45, .Count = 6}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 7.00649e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 9}, {.Sum = 1.12104e-44, .Count = 7}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 2.8026e-45, .Count = 4}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {4, 4, 5, 3, 1, 2, 7, 6, 1, 2, 5, 1, 2, 2, 1, 3, 1, 2, 1, 9, 8, 7, 1, 1, 1, 3, 2, 4, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1}
                    }
                },
                {
                    10041049327446511649ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {14590129323275859969ull, 39}, {4761737228040236034ull, 18}, {3061679160699873539ull, 43}, {4729919983694621444ull, 54}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8806247999156649096ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5682453444175299852ull, 52}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {285702114596126864ull, 44}, {5632299938741154192ull, 57}, {18446744073709551615ull, 0}, {8333726745003022227ull, 36}, {15952973246181437460ull, 11}, {6343051352217846420ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15549382406229051928ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16446935983613358747ull, 40}, {343067263211379228ull, 34}, {1340253711992729245ull, 22}, {7024200922256596382ull, 42}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16261345722633663778ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14166509076554175142ull, 58}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17479732393699636012ull, 26}, {18446744073709551615ull, 0}, {2982585812714178350ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3980241388377606578ull, 31}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10355191035340545717ull, 25}, {18446744073709551615ull, 0}, {3197060372330228023ull, 24}, {17395486763323672120ull, 56}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3025070705076148156ull, 27}, {21656200241330621ull, 2}, {10412716033556257724ull, 28}, {18446744073709551615ull, 0}, {7236416537553286208ull, 23}, {4097984215703358273ull, 16}, {2905510342784400193ull, 50}, {3540440309374370371ull, 12}, {9945806070767526596ull, 47}, {10600873139309967557ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14677201154771995209ull, 37}, {15697257388296343881ull, 45}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10665758689532857807ull, 8}, {9024898650572774224ull, 53}, {6436482984909430481ull, 29}, {3658183136700122066ull, 1}, {18446744073709551615ull, 0}, {8670866787526294612ull, 14}, {7478392914607336532ull, 48}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18209445798681454937ull, 10}, {16902413600193912026ull, 30}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8198345533372667743ull, 49}, {18446744073709551615ull, 0}, {16817797835260737121ull, 7}, {13206648158456337762ull, 38}, {9272675415381189347ull, 51}, {18446744073709551615ull, 0}, {11952238979044267493ull, 55}, {18446744073709551615ull, 0}, {16311554771983004263ull, 19}, {18446744073709551615ull, 0}, {9562935439960121449ull, 5}, {18446744073709551615ull, 0}, {15862627151959278699ull, 35}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6949441338819150318ull, 33}, {18446744073709551615ull, 0}, {11479717724890640624ull, 0}, {12336975088890661616ull, 21}, {16935540662586488816ull, 46}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11512437900041031286ull, 20}, {18446744073709551615ull, 0}, {9068013527945574136ull, 9}, {6476920084665523449ull, 32}, {1146182889791425528ull, 41}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 2.8026e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 5.60519e-45, .Count = 4}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 4.2039e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 4.2039e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.26117e-44, .Count = 1}, {.Sum = 9.80909e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 6}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 4.2039e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {2, 2, 1, 1, 1, 2, 4, 4, 1, 2, 3, 1, 1, 2, 2, 1, 3, 2, 1, 1, 2, 1, 9, 1, 7, 1, 1, 6, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
                    }
                },
                {
                    10086676643306191396ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {3817710399755548225ull, 23}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11763422251894688132ull, 28}, {6428229738182861508ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6674698524051710155ull, 0}, {18446744073709551615ull, 0}, {1864372354645034061ull, 4}, {9395139445057763342ull, 15}, {1143736165224900685ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18146346653071913682ull, 26}, {17941399936534664787ull, 27}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12418191270842125785ull, 21}, {18446744073709551615ull, 0}, {2431397336714355739ull, 16}, {18446744073709551615ull, 0}, {11159326772592326813ull, 18}, {18446744073709551615ull, 0}, {3435138084014180127ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6595469389054499366ull, 5}, {11548347322624516647ull, 2}, {3896939534752759014ull, 24}, {6507458873180072297ull, 6}, {18446744073709551615ull, 0}, {15624675338086544299ull, 12}, {4225752897234578476ull, 22}, {4430699613771827371ull, 25}, {12010148773363095534ull, 14}, {7033758902468218927ull, 1}, {18446744073709551615ull, 0}, {14620668111790408817ull, 13}, {16086476788825123186ull, 9}, {1943601489642244850ull, 8}, {4868989127185546932ull, 29}, {250252053542884853ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18020629071531875576ull, 10}, {18446744073709551615ull, 0}, {10056810728252581370ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14133138743714609022ull, 11}, {576711183155579007ull, 30}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 9}, {.Sum = 8.40779e-45, .Count = 15}, {.Sum = 5.60519e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 12}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 17}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 4.2039e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {0, 1, 0, 3, 0, 2, 0, 9, 6, 15, 4, 0, 0, 2, 0, 1, 0, 12, 0, 2, 0, 4, 0, 17, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 2, 1, 0, 3, 0, 1, 0, 0, 1, 1, 0, 1, 0}
                    }
                },
                {
                    12606205885276083427ull,
                    {
                        .IndexHashViewer = {{5321795528652759552ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {772623291280696100ull, 3}, {714275690842131332ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10783615582859474474ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10281901548337195535ull, 14}, {18446744073709551615ull, 0}, {6052518548450009169ull, 11}, {18446744073709551615ull, 0}, {8224442176515017331ull, 7}, {12538518194927513684ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9204844949746414424ull, 1}, {10052892563062224857ull, 4}, {3493345142105552026ull, 12}, {14505127246450782459ull, 9}, {18446744073709551615ull, 0}, {14486186593889963293ull, 8}, {7304087665005811933ull, 10}, {1871794946608052991ull, 2}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 9.80909e-45, .Count = 3}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 5.60519e-45, .Count = 24}, {.Sum = 0, .Count = 3}, {.Sum = 4.2039e-45, .Count = 16}, {.Sum = 4.2039e-45, .Count = 15}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}},
                        .CtrTotal = {7, 3, 1, 3, 4, 24, 0, 3, 3, 16, 3, 15, 0, 4, 0, 4, 1, 1, 0, 1, 0, 3, 0, 1, 0, 1, 1, 0, 2, 0}
                    }
                },
                {
                    13902559248212744134ull,
                    {
                        .IndexHashViewer = {{8975491433706742463ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14435487234778955461ull, 27}, {26794562384612742ull, 2}, {18446744073709551615ull, 0}, {4411634050168915016ull, 6}, {11361933621181601929ull, 19}, {15118949489711741514ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {488596013123191629ull, 9}, {2041917558348994126ull, 20}, {18446744073709551615ull, 0}, {3099115351550504912ull, 7}, {13955926499752636625ull, 14}, {6798076237643774482ull, 5}, {10555092106173914067ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4633306462361102487ull, 26}, {4428359745823853592ull, 21}, {16982002041722229081ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14612285549902308191ull, 10}, {18446744073709551615ull, 0}, {9142731084578380321ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {240279460452550314ull, 4}, {779318031744854123ull, 15}, {15286189140583379372ull, 23}, {4020317248344823341ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6630836586772136624ull, 16}, {18446744073709551615ull, 0}, {3266355002422142770ull, 3}, {15927023829150890738ull, 24}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {896638510602221880ull, 29}, {2066979203234309177ull, 1}, {16388825279889469625ull, 25}, {18446744073709551615ull, 0}, {6364972095279429180ull, 12}, {18446744073709551615ull, 0}, {18348953501661188798ull, 28}, {18144006785123939903ull, 13}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 1}, {.Sum = 7.00649e-45, .Count = 18}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 7.00649e-45, .Count = 0}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 24}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 4.2039e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 1, 5, 18, 0, 3, 0, 2, 0, 1, 5, 0, 0, 7, 0, 2, 0, 24, 0, 2, 0, 3, 0, 1, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1, 0, 2, 0, 1, 1, 1, 0, 2, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 3, 0, 0, 1, 2, 0, 0, 1}
                    }
                },
                {
                    13902559248212744135ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16479676762461049221ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14314906987178377226ull, 27}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12130212770433783695ull, 1}, {18446744073709551615ull, 0}, {4054001010745510673ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16548297050623529236ull, 5}, {9965442995151111700ull, 34}, {1889231235462838678ull, 28}, {18446744073709551615ull, 0}, {11147526993393187224ull, 10}, {18446744073709551615ull, 0}, {14555653527613724826ull, 12}, {12522231453186850331ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10958843647676541603ull, 32}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8794073872393869608ull, 31}, {8589127155856620713ull, 24}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11748579728051583916ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18384113673385397171ull, 29}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17769648050045596984ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13820256289847569724ull, 0}, {13621749364805718972ull, 2}, {1878905203052656190ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11450539798027648834ull, 36}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15254761925720908613ull, 16}, {18446744073709551615ull, 0}, {2398222922681060807ull, 30}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6227746613267076298ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18358311891254868174ull, 37}, {4062976837984404303ull, 33}, {7449767364017353680ull, 3}, {3858030121447155408ull, 23}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13653016638975931866ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14219847782559079394ull, 21}, {9089159255438104419ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16861309804843249000ull, 25}, {6719442763618183529ull, 13}, {16481986878556141930ull, 35}, {9655990399021251947ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11030694858814915054ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2718670329607888375ull, 8}, {7719283207639011575ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4940085441777621244ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 2.8026e-45, .Count = 1}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 4}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 9}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 8}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 1.4013e-45, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 12}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 2.8026e-45, .Count = 4}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {2, 1, 0, 7, 0, 1, 0, 1, 0, 2, 1, 4, 1, 1, 0, 9, 0, 5, 0, 2, 0, 1, 0, 8, 1, 5, 1, 4, 0, 1, 0, 1, 0, 2, 0, 12, 2, 2, 2, 4, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1}
                    }
                },
                {
                    14067340392369360370ull,
                    {
                        .IndexHashViewer = {{2633223011998620928ull, 9}, {353553385311653505ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14069200424611739816ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {638864303521046128ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12864327681460494291ull, 7}, {4622221383258683956ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7202634125138683991ull, 4}, {14354511342821132439ull, 6}, {13247339095716061336ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17230696175453699164ull, 5}, {18446744073709551615ull, 0}, {2471537090729218686ull, 3}, {8516242061306596031ull, 2}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 101,
                        .CtrMeanHistory = {{.Sum = 1.4013e-45, .Count = 4}, {.Sum = 1.68156e-44, .Count = 28}, {.Sum = 1.12104e-44, .Count = 1}, {.Sum = 2.8026e-44, .Count = 22}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {1, 4, 12, 28, 8, 1, 20, 22, 1, 2, 1, 1}
                    }
                },
                {
                    16302517177565331990ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13861830344683952963ull, 4}, {14504803952193703940ull, 26}, {3457347844175700611ull, 3}, {18446744073709551615ull, 0}, {11385118328825847751ull, 30}, {11635545141976567304ull, 28}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8515859518608711115ull, 2}, {18446744073709551615ull, 0}, {3487195925702945293ull, 13}, {4402382706704580365ull, 23}, {15224756570959035151ull, 27}, {5150911380664593808ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12355497760741898515ull, 0}, {4486542962472367699ull, 19}, {2281652570447457172ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15733203948979746011ull, 31}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17223387610340384159ull, 8}, {17219115279436609951ull, 15}, {12863945138762609375ull, 11}, {18446744073709551615ull, 0}, {14354128800123247523ull, 12}, {14055277641635467812ull, 18}, {18446744073709551615ull, 0}, {9133479741114045670ull, 20}, {18446744073709551615ull, 0}, {15545461302996105960ull, 7}, {18446744073709551615ull, 0}, {9130733310274487658ull, 6}, {18446744073709551615ull, 0}, {8666534670679167532ull, 25}, {7012749604856922477ull, 14}, {9292419231543889900ull, 17}, {10322065813147346095ull, 29}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6134217064971795062ull, 24}, {18446744073709551615ull, 0}, {3507740571040297848ull, 22}, {18446744073709551615ull, 0}, {7624400726332433210ull, 21}, {6517228479227362107ull, 5}, {638481760823161212ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16731089154901089599ull, 16}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 2.38221e-44, .Count = 10}, {.Sum = 0, .Count = 5}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 17}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 4, 0, 1, 0, 3, 0, 2, 17, 10, 0, 5, 1, 5, 0, 1, 0, 1, 0, 2, 0, 5, 0, 17, 0, 3, 0, 1, 2, 0, 0, 1, 0, 2, 1, 0, 0, 3, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1}
                    }
                },
                {
                    16302517178123948687ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {3581428127016485793ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14455983217430950149ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13125720576600207402ull, 3}, {5967870314491345259ull, 4}, {9724886183021484844ull, 8}, {2436149079269713547ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1236773280081879954ull, 1}, {16151796118569799858ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18336378346035991543ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8312525161425951098ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 11}, {.Sum = 2.8026e-45, .Count = 8}, {.Sum = 0, .Count = 30}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-44, .Count = 12}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}},
                        .CtrTotal = {0, 11, 2, 8, 0, 30, 0, 2, 20, 12, 0, 7, 0, 3, 0, 2, 0, 1, 0, 3}
                    }
                },
                {
                    17677952491745844307ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12386861758883532418ull, 7}, {18446744073709551615ull, 0}, {1615194939518637828ull, 37}, {18446744073709551615ull, 0}, {6282805614514371974ull, 28}, {18446744073709551615ull, 0}, {7984935183757335944ull, 8}, {18446744073709551615ull, 0}, {8042104875066952842ull, 13}, {6247725372209840139ull, 0}, {5764668879944766602ull, 44}, {5186539704000666381ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11515863767753292944ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6510485676727294867ull, 19}, {1048747059899601044ull, 50}, {8212615245970258837ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1329552534770423325ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12117686985652909478ull, 49}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1033962021405470249ull, 27}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14812291057069738284ull, 3}, {4287306346270768173ull, 54}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {248609998944785840ull, 35}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16594914033306627510ull, 20}, {15723158860768052023ull, 34}, {9992426192305263030ull, 47}, {13749899362210214201ull, 40}, {17887935727143550905ull, 36}, {18446744073709551615ull, 0}, {9743905216976139708ull, 41}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18166796106075650879ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14067633988441960386ull, 51}, {10220106254518185923ull, 42}, {11802919703730877636ull, 48}, {18446744073709551615ull, 0}, {18067813707702657862ull, 38}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18191698266252687691ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9165362083576827855ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7596033389440299218ull, 23}, {18446744073709551615ull, 0}, {16660079067913029716ull, 2}, {18446744073709551615ull, 0}, {526803805060551766ull, 25}, {12654084922678955223ull, 21}, {18446744073709551615ull, 0}, {14356214491921919193ull, 11}, {1888390545754488410ull, 17}, {6849666623986928987ull, 24}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11865561032414258910ull, 9}, {18446744073709551615ull, 0}, {10372181779362786528ull, 31}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5024983930317320419ull, 45}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11516656418978967656ull, 30}, {17084754933191592809ull, 22}, {4233045392689094249ull, 33}, {11573826110288584554ull, 53}, {13690800895010267500ull, 29}, {6027424895546206952ull, 12}, {9678542013189384684ull, 39}, {3493364731977662444ull, 52}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10042206911948926579ull, 10}, {18446744073709551615ull, 0}, {11744336481191890549ull, 5}, {15800972419013521781ull, 43}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13918480957223190393ull, 32}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {950539760915361660ull, 55}, {3903070525275234300ull, 6}, {18446744073709551615ull, 0}, {3963970869784021247ull, 46}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 4.2039e-45, .Count = 6}, {.Sum = 4.2039e-45, .Count = 0}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 6}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 7}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {0, 2, 0, 1, 0, 1, 0, 3, 0, 1, 3, 6, 3, 0, 0, 4, 0, 1, 0, 1, 1, 6, 0, 1, 0, 3, 0, 7, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1, 0, 7, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 5, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 2, 0, 1, 1, 1, 0, 1, 0}
                    }
                },
                {
                    17677952491745844311ull,
                    {
                        .IndexHashViewer = {{11909113549668261632ull, 70}, {4102723187689520639ull, 69}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1807858446515241992ull, 6}, {18446744073709551615ull, 0}, {12942588402337221386ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4152513293449847831ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5017286076567840540ull, 2}, {17885275719815889437ull, 63}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5613411879964570145ull, 41}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13931387008599280677ull, 54}, {8847881709250651429ull, 81}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4842844206505004080ull, 23}, {18446744073709551615ull, 0}, {17443854143189189682ull, 18}, {6544973775747968050ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2907989964166681148ull, 3}, {101543489846701373ull, 57}, {18446744073709551615ull, 0}, {7187499053439609919ull, 38}, {18446744073709551615ull, 0}, {1140546422690990401ull, 52}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3761594381613069894ull, 34}, {2842742169623030343ull, 49}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11561798493305122124ull, 53}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3417767983137710927ull, 60}, {1311122971283372368ull, 46}, {18446744073709551615ull, 0}, {14846055957431315282ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18319814850117655384ull, 1}, {18446744073709551615ull, 0}, {9294362788885428570ull, 26}, {13314436759091657307ull, 82}, {10996492358128392540ull, 59}, {15016566328334621277ull, 75}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13170636834159692384ull, 47}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4811188340165863780ull, 78}, {18446744073709551615ull, 0}, {18239922540704047718ull, 21}, {142773302120580199ull, 32}, {11937261703073749095ull, 55}, {18446744073709551615ull, 0}, {6327947688840526442ull, 29}, {18446744073709551615ull, 0}, {8030077258083490412ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7294260752003454833ull, 30}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17527125680970607732ull, 58}, {2137833313929101941ull, 48}, {9478614031528592246ull, 77}, {18446744073709551615ull, 0}, {11595588816250275192ull, 42}, {18446744073709551615ull, 0}, {5762641553663796858ull, 72}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16584649892755421059ull, 74}, {18446744073709551615ull, 0}, {18286779461998385029ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3835952806284834451ull, 35}, {5889723104997343636ull, 10}, {3911654907320017812ull, 12}, {5613784476562981782ull, 7}, {9670420414384613014ull, 62}, {17091011266371226517ull, 13}, {18446744073709551615ull, 0}, {14584547855873447322ull, 45}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15847069910243129501ull, 28}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18272365298582949536ull, 64}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15293882737739361187ull, 43}, {2415237178242434724ull, 33}, {14755058226776753317ull, 51}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4479179307745462185ull, 56}, {13319789865378462377ull, 73}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1149495066663649715ull, 36}, {379933672098386100ull, 68}, {10297315814342024884ull, 79}, {2082063241341350070ull, 37}, {18446744073709551615ull, 0}, {660158800220621496ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16164088077643925949ull, 8}, {18446744073709551615ull, 0}, {14357259258425458367ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7897214113545270722ull, 67}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9624337656299288519ull, 44}, {10767921952148383687ull, 66}, {18446744073709551615ull, 0}, {18433921072266295498ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10287711995487688398ull, 31}, {8124894175758193615ull, 61}, {11989841564730652368ull, 65}, {18446744073709551615ull, 0}, {11840658872237222098ull, 20}, {2745520608914164435ull, 76}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12320927121250414812ull, 27}, {18446744073709551615ull, 0}, {14023056690493378782ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3155226402211736291ull, 39}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14281343389671099366ull, 40}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17534687714536397290ull, 80}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3932212816786214644ull, 15}, {10523796155324129524ull, 50}, {8552422751056303094ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8789205886028783100ull, 24}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4080365869422862847ull, 71}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 4, 0, 1, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 4, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 2, 2, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1}
                    }
                },
                {
                    17677952491747546147ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17787954881284471813ull, 24}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7454420046185256717ull, 40}, {16256335682944813838ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1636731659193698578ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5922800847845598742ull, 7}, {14182197490569975831ull, 48}, {7624930417088562712ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10422205982269444643ull, 64}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3411314423057176877ull, 42}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4522605207985801776ull, 34}, {18446744073709551615ull, 0}, {13192676729576349746ull, 19}, {16466569643076362291ull, 22}, {18300934243650069811ull, 28}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4431368220400894274ull, 65}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14233673023285815109ull, 56}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2899749022061236299ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8023290181753164880ull, 43}, {9933882341717515345ull, 66}, {3233597379123467602ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8402263143377857370ull, 39}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17651556054977644126ull, 37}, {15680080812126751838ull, 45}, {17708725746287261024ull, 60}, {18446744073709551615ull, 0}, {1780070264439091554ull, 51}, {15773274901763725923ull, 13}, {16328374789029446500ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16177106547947603049ull, 21}, {18446744073709551615ull, 0}, {17879236117190567019ull, 5}, {14241655703424067948ull, 38}, {3489127981302646635ull, 50}, {15943785272667031918ull, 55}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9771448801094703501ull, 35}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11530748061647284369ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5994047302704556692ull, 31}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10117199296271121559ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9999128863307626394ull, 27}, {18446744073709551615ull, 0}, {11701258432550590364ull, 33}, {7854656800704835228ull, 47}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {118997543255608737ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10779812027622989220ull, 32}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6111396989577705639ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16127325828303939500ull, 53}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14224606228188042159ull, 3}, {5576091289432675759ull, 44}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14966077412008197812ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10025163577623610551ull, 23}, {1755789550731085240ull, 57}, {7501413217152384697ull, 17}, {16355005890516862393ull, 49}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14797650915799523780ull, 63}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13730933025438975688ull, 30}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {724243645116964305ull, 29}, {18446744073709551615ull, 0}, {11702735195037717203ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16072394239031333591ull, 2}, {18446744073709551615ull, 0}, {11159883566315996889ull, 52}, {11603752796664724186ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16142728109259286750ull, 26}, {18446744073709551615ull, 0}, {17844857678502250720ull, 12}, {18446744073709551615ull, 0}, {15813441649188061154ull, 46}, {9628264367976338914ull, 59}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2145056323740669926ull, 67}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9516068082126538479ull, 61}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10037970161273910770ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17560274819071548920ull, 36}, {11038948726666369272ull, 62}, {18446744073709551615ull, 0}, {8596718462362217979ull, 54}, {18446744073709551615ull, 0}, {16215728555360712189ull, 41}, {10298848031605181949ull, 58}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 4.2039e-45, .Count = 4}, {.Sum = 4.2039e-45, .Count = 0}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 6}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 3, 4, 3, 0, 0, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 4, 0, 3, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 6, 0, 1, 0, 2, 0, 4, 1, 0, 0, 1, 0, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 2, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1}
                    }
                },
                {
                    17677952493035932493ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {5463648634093988865ull, 17}, {10335157342305182722ull, 32}, {6738126486361565313ull, 4}, {10791805669449703300ull, 37}, {2663236581938862725ull, 29}, {18446744073709551615ull, 0}, {8062072459088193799ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11155470568209245195ull, 41}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1131617383599204750ull, 13}, {4832104736055316622ull, 38}, {8806694637943861776ull, 33}, {14839272646072616465ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15228474649950355220ull, 43}, {13076441876572047509ull, 1}, {15233604573492176790ull, 9}, {13439225070635064087ull, 0}, {9152574924415960088ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2585548803766369181ull, 44}, {3108819379344245150ull, 15}, {13701985375152518815ull, 20}, {16363877839430637216ull, 3}, {15404114944395482785ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11497229771350565927ull, 21}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2435711933190352810ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1798551784282379183ull, 27}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5796622587947976498ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6185824591825715253ull, 40}, {18446744073709551615ull, 0}, {4290540494726515383ull, 35}, {3557046681785410616ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4265003389608318523ull, 23}, {18446744073709551615ull, 0}, {5967132958851282493ull, 12}, {4654205393486057278ull, 19}, {18446744073709551615ull, 0}, {6356334962729021248ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10716948865937726787ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16961813080521916615ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17235223807996045521ull, 28}, {6812569332418330194ull, 39}, {18446744073709551615ull, 0}, {12566808769950761940ull, 24}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6127862249290423643ull, 42}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5151892753385378284ull, 36}, {15470067739584089325ull, 30}, {15822331420583681518ull, 34}, {13574783642484889455ull, 16}, {13134404219107339119ull, 45}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13938448541244431350ull, 7}, {18446744073709551615ull, 0}, {11602785020767681144ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7799312154570739071ull, 31}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 5.60519e-45, .Count = 7}, {.Sum = 7.00649e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 7}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 6}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 4.2039e-45, .Count = 6}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 4.2039e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 4, 7, 5, 0, 0, 2, 0, 2, 0, 7, 1, 1, 0, 1, 0, 6, 0, 3, 0, 5, 0, 2, 0, 1, 1, 0, 0, 1, 3, 6, 0, 2, 0, 1, 1, 0, 0, 7, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1, 3, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1}
                    }
                },
                {
                    17677952493039731422ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5923468399088953603ull, 48}, {9652310184282050820ull, 22}, {1580021004445293060ull, 43}, {1508774549586335110ull, 19}, {18446744073709551615ull, 0}, {3210904118829299080ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1887722852881101454ull, 7}, {13816382236622194447ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14817360802014652949ull, 46}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12375130537710501656ull, 41}, {18446744073709551615ull, 0}, {14077260106953465626ull, 45}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3119622882923203874ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11232832121533540394ull, 31}, {1588003684583545899ull, 13}, {9337548024434340524ull, 16}, {3290133253826509869ull, 42}, {15309160136995568046ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9772459378052840369ull, 27}, {18446744073709551615ull, 0}, {9701212923193882419ull, 2}, {13895611371619405236ull, 0}, {11403342492436846389ull, 4}, {18446744073709551615ull, 0}, {13777540938655910071ull, 23}, {18446744073709551615ull, 0}, {15479670507898874041ull, 11}, {18446744073709551615ull, 0}, {17861627167545266491ull, 39}, {3562076536520190140ull, 32}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14558224102971272897ull, 28}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9889809064925989316ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7189726498405460554ull, 33}, {18446744073709551615ull, 0}, {18003018303536325836ull, 3}, {8301017283334085453ull, 29}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1798237644715094352ull, 20}, {3632602245288801872ull, 24}, {297745413646929873ull, 1}, {18446744073709551615ull, 0}, {13803575652971894228ull, 21}, {18446744073709551615ull, 0}, {1686673892155594454ull, 38}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12083043806050633052ull, 44}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8209780295749177951ull, 47}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11507505354713591267ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6678161097409519976ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13712294417065799022ull, 8}, {4502655720465247982ull, 25}, {11817010319966599152ull, 37}, {539024710735827697ull, 17}, {17737025163000182258ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12180675218726141047ull, 30}, {15382164872013007863ull, 18}, {1483475484867384056ull, 36}, {18446744073709551615ull, 0}, {1011748813765483899ull, 34}, {18446744073709551615ull, 0}, {3040393747925993085ull, 40}, {18446744073709551615ull, 0}, {1145109650826793215ull, 35}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 5.60519e-45, .Count = 5}, {.Sum = 7.00649e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 4.2039e-45, .Count = 6}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 4.2039e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 1, 0, 1, 0, 4, 0, 1, 0, 1, 4, 5, 5, 0, 0, 1, 0, 2, 1, 1, 0, 1, 0, 5, 0, 7, 0, 5, 0, 5, 0, 1, 0, 1, 1, 0, 0, 1, 3, 6, 0, 1, 0, 2, 1, 0, 0, 5, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1, 3, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1}
                    }
                },
                {
                    17677952493224740165ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5195954639254248834ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12596183338487933509ull, 1}, {11415090325326527685ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15694110684349775305ull, 6}, {3105842455684076810ull, 17}, {18446744073709551615ull, 0}, {11619308647181131660ull, 9}, {18446744073709551615ull, 0}, {7384862814707324430ull, 2}, {16546783282337640335ull, 11}, {13877983093189917584ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9803181056021273939ull, 14}, {17960003200548727507ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15929159679822070487ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1885423701024940001ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6164444060048187621ull, 3}, {1036643838009237222ull, 19}, {18446744073709551615ull, 0}, {2089642022879543976ull, 0}, {3679105889079969577ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1862978297920111856ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11528263922108981619ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7453461884940337974ull, 15}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16229983591748392701ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 8}, {.Sum = 1.4013e-44, .Count = 18}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 26}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 4.2039e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 7.00649e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 8, 10, 18, 0, 3, 0, 26, 0, 2, 0, 2, 0, 2, 0, 4, 0, 1, 0, 2, 3, 0, 2, 0, 1, 2, 0, 3, 0, 1, 0, 2, 1, 0, 0, 1, 0, 1, 5, 0, 0, 1}
                    }
                },
                {
                    17677952493224740166ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {1799168355831033313ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11936664559898134054ull, 2}, {14666845749088704071ull, 8}, {18429784838380727208ull, 10}, {17027374437435318793ull, 3}, {2862173265672040777ull, 1}, {16080065667299791243ull, 6}, {14677655266354382828ull, 13}, {12391839889973628461ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4082592020331177586ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {315676340386518075ull, 7}, {18446744073709551615ull, 0}, {10716245805238997245ull, 0}, {9313835404293588830ull, 9}, {17603450378469852574ull, 12}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 1.4013e-44, .Count = 46}, {.Sum = 0, .Count = 8}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 2.8026e-45, .Count = 6}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 4}, {.Sum = 0, .Count = 2}, {.Sum = 7.00649e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {10, 46, 0, 8, 0, 2, 0, 1, 0, 2, 1, 3, 2, 6, 0, 1, 2, 0, 0, 4, 1, 4, 0, 2, 5, 0, 1, 0}
                    }
                },
                {
                    17677952493224740167ull,
                    {
                        .IndexHashViewer = {{7515733889724454912ull, 18}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2160905354121516547ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13659069800549444297ull, 4}, {7791826943727985930ull, 0}, {7884511582485373322ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18022007786552474063ull, 10}, {18446744073709551615ull, 0}, {6068383991325515601ull, 17}, {7524725216182310545ull, 13}, {17609669744399151123ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11681580651965248598ull, 12}, {576145588900686679ull, 19}, {13155646805788779928ull, 8}, {18446744073709551615ull, 0}, {5849831644443487770ull, 14}, {3372332782322797723ull, 2}, {18446744073709551615ull, 0}, {9865453060805390877ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9800431588293194596ull, 15}, {9048109927352371876ull, 24}, {16801589031893337254ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2099530300070748010ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4741992351141480365ull, 20}, {17321493568029573614ull, 21}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2151914027663660914ull, 3}, {9012245698387122739ull, 6}, {3718664820244579636ull, 23}, {2925864759981622644ull, 16}, {15505365976869715893ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 13}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 9}, {.Sum = 1.4013e-44, .Count = 15}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 17}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 2}, {.Sum = 5.60519e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {0, 13, 0, 3, 0, 2, 0, 9, 10, 15, 0, 2, 0, 1, 0, 2, 0, 4, 0, 17, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 0, 2, 4, 0, 0, 1, 1, 0, 1, 0}
                    }
                },
                {
                    17677952493224740170ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4074149013537909256ull, 13}, {18446744073709551615ull, 0}, {12733361023712308234ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10360532617354119439ull, 17}, {18446744073709551615ull, 0}, {12179894358259103633ull, 27}, {18446744073709551615ull, 0}, {3294711086045205011ull, 24}, {12096630803572290451ull, 34}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3646433500654197144ull, 10}, {5941490484899010585ull, 25}, {18446744073709551615ull, 0}, {9780057282422735643ull, 4}, {5597533724707970587ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9010253362714991142ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15143877144483529641ull, 30}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17675697484024171309ull, 16}, {7968584429078161966ull, 1}, {2650227733957515949ull, 14}, {17247737190584974768ull, 32}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10533638073885052473ull, 28}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13332404291138955964ull, 31}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6940720914563173696ull, 18}, {4866852255313333953ull, 7}, {18446744073709551615ull, 0}, {11359972533795927107ull, 21}, {2660929095326822084ull, 15}, {13655029518040740548ull, 29}, {17493596315564465606ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8146030859722583625ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11603958955687655886ull, 12}, {3905715562244114895ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13509850721783377623ull, 22}, {18446744073709551615ull, 0}, {15682123462219891929ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2074292331386068202ull, 23}, {15033235432953438954ull, 33}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9366449614381720434ull, 2}, {18446744073709551615ull, 0}, {15859569892864313588ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17157073225186666874ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {862979833014771711ull, 8}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 4.2039e-45, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 4.2039e-45, .Count = 12}, {.Sum = 1.4013e-45, .Count = 20}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 2.8026e-45, .Count = 10}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 0, .Count = 4}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {3, 2, 0, 2, 0, 1, 0, 3, 3, 12, 1, 20, 0, 2, 0, 2, 1, 1, 0, 3, 2, 10, 0, 1, 0, 2, 2, 1, 0, 4, 1, 0, 0, 1, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 2, 0, 1, 0}
                    }
                },
                {
                    17677952493260528854ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17985374731566054150ull, 22}, {18446744073709551615ull, 0}, {4969880389554839688ull, 30}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1883285504791108373ull, 14}, {14139902777924824981ull, 23}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17540248381108153753ull, 15}, {18446744073709551615ull, 0}, {2120068639763588379ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1277857586923739550ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9915512646490226338ull, 9}, {18446744073709551615ull, 0}, {5780999427119446436ull, 37}, {15493676505554854693ull, 31}, {14453653496344422438ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3622512433858345389ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9415440463389949361ull, 35}, {18446744073709551615ull, 0}, {15689261734764374707ull, 7}, {17838331352489460532ull, 17}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18403625549429831228ull, 24}, {18446744073709551615ull, 0}, {16192880425411659454ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6383791411268594626ull, 3}, {18033916581698980546ull, 33}, {18446744073709551615ull, 0}, {11961955270333222981ull, 13}, {18446744073709551615ull, 0}, {11191788834073534919ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17618698769621849933ull, 28}, {5730630563994427981ull, 32}, {16620451033949360975ull, 19}, {647125264645798733ull, 38}, {7150295984444125389ull, 29}, {18446744073709551615ull, 0}, {12157540499542742995ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1072059942279220057ull, 34}, {10177020748048094298ull, 1}, {18446744073709551615ull, 0}, {9494950831378731228ull, 8}, {18446744073709551615ull, 0}, {518361807174415198ull, 26}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {592499207252901221ull, 18}, {4098784705883188966ull, 36}, {10062654256758136807ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3618574749222493677ull, 21}, {18446744073709551615ull, 0}, {13088729798727729263ull, 27}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2625225542620233849ull, 20}, {6645299512826462586ull, 2}, {5651789874985220091ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 4}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 8.40779e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 7}, {.Sum = 9.80909e-45, .Count = 1}, {.Sum = 0, .Count = 7}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 4, 2, 2, 0, 5, 0, 3, 0, 1, 1, 1, 6, 1, 2, 4, 0, 1, 0, 2, 0, 5, 0, 1, 0, 2, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 2, 7, 7, 1, 0, 7, 0, 1, 0, 1, 0, 1, 0, 3, 0, 2, 0, 4, 0, 1, 0, 1, 0, 2, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1}
                    }
                },
                {
                    17677952493261641996ull,
                    {
                        .IndexHashViewer = {{16259707375369223360ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13847085545544291780ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7654620248348398600ull, 0}, {18446744073709551615ull, 0}, {9243796653651753418ull, 8}, {18446744073709551615ull, 0}, {1681026541770505292ull, 15}, {1292491219513334285ull, 20}, {13677090684479491854ull, 14}, {6494991755595340494ull, 17}, {7494438315637327440ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18410271455579776277ull, 3}, {6336919059871405781ull, 22}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9974519673449003035ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5899717636280359390ull, 18}, {18446744073709551615ull, 0}, {15904544917366469984ull, 24}, {18446744073709551615ull, 0}, {862592111642406882ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18161161563956788133ull, 6}, {18446744073709551615ull, 0}, {3340544229935902247ull, 21}, {18446744073709551615ull, 0}, {14827488318775688873ull, 1}, {15675535932091499306ull, 5}, {18446744073709551615ull, 0}, {15230422751883885548ull, 26}, {18446744073709551615ull, 0}, {1662085889209686126ull, 27}, {18446744073709551615ull, 0}, {1062699037197581552ull, 2}, {14072903496117963889ull, 23}, {18446744073709551615ull, 0}, {15434641073738489523ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14277121817972567864ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18160464660109825851ull, 11}, {16406258951888748923ull, 25}, {17480885798804750972ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 3}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 11}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 5.60519e-45, .Count = 7}, {.Sum = 9.80909e-45, .Count = 0}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 8}, {.Sum = 0, .Count = 12}, {.Sum = 5.60519e-45, .Count = 7}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 4.2039e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {0, 3, 1, 1, 0, 11, 0, 3, 0, 1, 4, 7, 7, 0, 0, 5, 0, 8, 0, 12, 4, 7, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 2, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 2, 0, 3, 1, 0, 1, 1, 0}
                    }
                },
                {
                    17677952493261641999ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1955048961233476230ull, 26}, {7848505467359907975ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5362879835494619148ull, 9}, {16975894248446493197ull, 23}, {2989452228607938574ull, 35}, {1555714507101927823ull, 22}, {14000534895061105936ull, 8}, {9017974062374515601ull, 46}, {18446744073709551615ull, 0}, {1131954680416182675ull, 13}, {11251729997004693012ull, 30}, {18446744073709551615ull, 0}, {8075887589815939734ull, 28}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14992766016416223513ull, 16}, {5940650566388999194ull, 18}, {13889299363682761882ull, 15}, {3813165541490991516ull, 44}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1327539909625702689ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11143541822018851883ull, 48}, {18446744073709551615ull, 0}, {17750248367578526765ull, 45}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {456542851774570932ull, 12}, {18446744073709551615ull, 0}, {5819437278606934070ull, 49}, {18446744073709551615ull, 0}, {7440970810988363064ull, 17}, {11397601137671562425ull, 25}, {18446744073709551615ull, 0}, {11235318233015004987ull, 33}, {17945833596788046779ull, 38}, {2258729208810688957ull, 4}, {10240081881905640766ull, 19}, {8486513334958592063ull, 50}, {18446744073709551615ull, 0}, {17956668384463827649ull, 31}, {5591888820769589826ull, 43}, {17054344636960189377ull, 32}, {18446744073709551615ull, 0}, {4354375926224631749ull, 10}, {18446744073709551615ull, 0}, {13413995383303163463ull, 1}, {11593186366881082439ull, 11}, {14262042996618973896ull, 40}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9406867846467797461ull, 20}, {18446744073709551615ull, 0}, {430278822263481431ull, 36}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4720265623279577179ull, 3}, {18446744073709551615ull, 0}, {6093602855291868765ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {3054220616560833121ull, 7}, {9195873423056402274ull, 24}, {12433592610071766370ull, 47}, {14731257914858355553ull, 34}, {12524375666066068707ull, 21}, {1896701360794911462ull, 41}, {6710247791191113447ull, 42}, {11366856410300147048ull, 39}, {9736812123556099689ull, 5}, {8888764510240289256ull, 29}, {18446744073709551615ull, 0}, {8994591768235559660ull, 51}, {18446744073709551615ull, 0}, {6080945380164802030ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2058262929422952306ull, 37}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {15668612263842743423ull, 27}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 2.8026e-45, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 5}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 0, .Count = 4}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 4}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 5.60519e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {2, 3, 0, 1, 1, 3, 0, 1, 0, 5, 1, 0, 1, 2, 0, 4, 0, 2, 0, 2, 0, 1, 0, 5, 0, 1, 1, 1, 2, 5, 0, 1, 0, 2, 0, 3, 0, 1, 0, 2, 0, 1, 1, 4, 0, 2, 2, 2, 4, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1}
                    }
                },
                {
                    17677952493262230690ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {9846335135722999169ull, 14}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13922663151185026821ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9507969301682408328ull, 3}, {18033043251005298120ull, 8}, {15936568783484859849ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8184788035734210702ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1653896762115451860ull, 10}, {18446744073709551615ull, 0}, {16297997274344547222ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1927581216097023258ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8105558900736999913ull, 15}, {14677355849505593194ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {9587198436679619117ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17700407675289955637ull, 2}, {6823283309938636726ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1733125897112662649ull, 5}, {3329991617042431673ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5343897249342264701ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 16}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 10}, {.Sum = 2.52234e-44, .Count = 15}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 5}, {.Sum = 2.8026e-45, .Count = 2}, {.Sum = 0, .Count = 18}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 16, 0, 1, 0, 10, 18, 15, 0, 1, 0, 5, 2, 2, 0, 18, 0, 1, 0, 1, 0, 2, 1, 0, 0, 3, 0, 1, 0, 2, 1, 0, 0, 1}
                    }
                },
                {
                    17677952493263343771ull,
                    {
                        .IndexHashViewer = {{15330345801530070271ull, 8}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13871343560304450565ull, 12}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17989766274143549768ull, 21}, {18334501489220455433ull, 18}, {17271881404906880906ull, 16}, {1327065643761606346ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5745149923951351887ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18147836298725285973ull, 24}, {11919737177904201494ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {220392991226246300ull, 13}, {11009125960592947549ull, 22}, {16732756202475478686ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1799168355831033313ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17299587346058626214ull, 11}, {945432601379406567ull, 6}, {18446744073709551615ull, 0}, {227547732142737705ull, 7}, {8878683662908522218ull, 14}, {18371399316525749547ull, 23}, {18446744073709551615ull, 0}, {12391839889973628461ull, 20}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4342739523005943472ull, 17}, {18446744073709551615ull, 0}, {10362267276645262642ull, 3}, {6966500923373419635ull, 4}, {9445514806491669746ull, 9}, {10820219266285332853ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {17559172457516014783ull, 15}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 8.40779e-45, .Count = 8}, {.Sum = 0, .Count = 2}, {.Sum = 2.8026e-45, .Count = 5}, {.Sum = 0, .Count = 6}, {.Sum = 2.8026e-45, .Count = 11}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 2.8026e-45, .Count = 8}, {.Sum = 0, .Count = 8}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 3}, {.Sum = 0, .Count = 2}, {.Sum = 9.80909e-45, .Count = 6}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {6, 8, 0, 2, 2, 5, 0, 6, 2, 11, 1, 3, 2, 8, 0, 8, 0, 1, 0, 1, 1, 3, 0, 2, 7, 6, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 1, 0, 2, 0, 2, 0, 1, 1, 0, 0, 1, 0, 1}
                    }
                },
                {
                    17677952493297533872ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {228832412018222341ull, 29}, {18446744073709551615ull, 0}, {11579036573410064263ull, 37}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {2142920538933900555ull, 49}, {18446744073709551615ull, 0}, {11420714090427158285ull, 12}, {18446744073709551615ull, 0}, {17720405802426315535ull, 24}, {3215834049561110672ull, 51}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {346575239343974036ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13139983920087306647ull, 41}, {14860408764928037144ull, 6}, {286844492446271769ull, 47}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10925792178412610972ull, 3}, {12726869934920056605ull, 22}, {11945848411936959644ull, 39}, {18446744073709551615ull, 0}, {11343638620497380128ull, 44}, {9857611124702919969ull, 20}, {15541558334966787106ull, 36}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10990677728635501222ull, 8}, {4919457811166910375ull, 13}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4237122415554814250ull, 52}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {339035928827901487ull, 42}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8200830002684883256ull, 14}, {6893797804197340345ull, 28}, {1058988547593232698ull, 43}, {11714417785040418747ull, 23}, {18446744073709551615ull, 0}, {6067291172676902717ull, 35}, {16636473811085647678ull, 18}, {18446744073709551615ull, 0}, {483329372556896832ull, 26}, {3198032362459766081ull, 31}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {12661894127993305031ull, 19}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {4340360739111205579ull, 21}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1471101928894068943ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {464994231589622356ull, 25}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5278641733246315480ull, 17}, {14915048362378503384ull, 30}, {1537907742216832473ull, 48}, {5054839022797264859ull, 16}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {6888411174261376229ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {16939687026671270763ull, 50}, {14120581721888279787ull, 46}, {18080292852670312173ull, 5}, {7952734526884932333ull, 27}, {15038402360561546607ull, 32}, {9875412811804264560ull, 1}, {8723830392309106799ull, 38}, {16771855022716002162ull, 34}, {8813616260402002415ull, 11}, {7006154001587127924ull, 40}, {5933240490959917807ull, 33}, {18446744073709551615ull, 0}, {5540766610232480247ull, 15}, {18446744073709551615ull, 0}, {16586264761736307193ull, 4}, {18446744073709551615ull, 0}, {6712598941894663547ull, 45}, {17585370940655764860ull, 9}, {9392162505557741693ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}, {.Sum = 7.00649e-45, .Count = 1}, {.Sum = 2.8026e-45, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 9.80909e-45, .Count = 1}, {.Sum = 0, .Count = 5}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 3}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}, {.Sum = 1.4013e-45, .Count = 0}},
                        .CtrTotal = {0, 3, 0, 1, 0, 3, 0, 3, 0, 1, 1, 1, 5, 1, 2, 3, 0, 1, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 1, 0, 3, 0, 1, 0, 1, 0, 1, 2, 5, 0, 1, 0, 2, 7, 1, 0, 5, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0}
                    }
                },
                {
                    18092064124022228906ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {3581428127016485793ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13125720576600207402ull, 3}, {5967870314491345259ull, 6}, {9724886183021484844ull, 5}, {2436149079269713547ull, 9}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1236773280081879954ull, 1}, {16151796118569799858ull, 2}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18336378346035991543ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8312525161425951098ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13605281311626526238ull, 8}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 4}, {.Sum = 2.52234e-44, .Count = 19}, {.Sum = 0, .Count = 36}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 9}, {.Sum = 0, .Count = 4}, {.Sum = 5.60519e-45, .Count = 1}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 1}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 4, 18, 19, 0, 36, 0, 2, 0, 9, 0, 4, 4, 1, 0, 2, 0, 1, 0, 1}
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
