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
        {33.5, 35.5, 37.5, 48, 51.5, 52.5, 59.5},
        {51773, 66768, 73183.5, 124942, 208500.5, 213107.5, 222939, 271875, 313025.5, 338713, 350449},
        {8, 10.5, 13.5, 14.5, 15.5},
        {1087, 3280, 5842, 11356},
        {1881.5, 1944.5, 2189.5},
        {44.5, 46.5, 49, 55}
    };
    std::vector<unsigned int> TreeDepth = {4, 3, 6, 6, 1, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
    std::vector<unsigned int> TreeSplits = {55, 19, 1, 24, 24, 29, 69, 56, 25, 35, 68, 70, 24, 38, 66, 20, 64, 6, 36, 31, 23, 32, 56, 40, 42, 20, 34, 67, 24, 2, 52, 16, 31, 62, 41, 37, 51, 41, 19, 71, 14, 54, 22, 46, 25, 26, 52, 56, 24, 4, 40, 56, 31, 13, 53, 3, 48, 19, 44, 62, 7, 22, 43, 33, 11, 20, 28, 49, 47, 33, 60, 48, 7, 33, 28, 27, 61, 47, 18, 56, 17, 15, 0, 12, 58, 53, 8, 65, 72, 56, 65, 59, 45, 5, 10, 30, 39, 50, 24, 52, 55, 9, 7, 57, 63, 22, 33, 21};
    std::vector<unsigned char> TreeSplitIdxs = {2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 1, 2, 1, 1, 3, 1, 7, 3, 2, 1, 3, 3, 3, 5, 3, 1, 1, 2, 3, 4, 10, 2, 3, 4, 1, 3, 4, 2, 2, 8, 1, 5, 1, 3, 4, 4, 3, 2, 5, 3, 3, 2, 7, 5, 4, 3, 2, 2, 3, 1, 5, 1, 4, 5, 3, 2, 1, 2, 4, 1, 3, 1, 4, 2, 1, 2, 2, 1, 3, 11, 9, 1, 6, 2, 5, 2, 2, 3, 3, 2, 3, 3, 6, 4, 1, 2, 2, 2, 4, 2, 3, 1, 1, 4, 5, 4, 4};
    std::vector<unsigned short> TreeSplitFeatureIndex = {12, 2, 0, 3, 3, 4, 17, 12, 3, 6, 17, 18, 3, 8, 16, 2, 15, 0, 6, 5, 3, 5, 12, 8, 8, 2, 6, 17, 3, 0, 11, 1, 5, 14, 8, 7, 11, 8, 2, 18, 1, 12, 2, 10, 3, 3, 11, 12, 3, 0, 8, 12, 5, 1, 11, 0, 10, 2, 9, 14, 1, 2, 9, 5, 1, 2, 4, 11, 10, 5, 14, 10, 1, 5, 4, 4, 14, 10, 2, 12, 1, 1, 0, 1, 13, 11, 1, 15, 18, 12, 15, 13, 9, 0, 1, 5, 8, 11, 3, 11, 12, 1, 1, 13, 14, 2, 5, 2};
    std::vector<unsigned char> TreeSplitXorMask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<unsigned int> CatFeaturesIndex = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<unsigned int> OneHotCatFeatureIndex = {};
    std::vector<std::vector<int>> OneHotHashValues = {
    };
    std::vector<std::vector<float>> CtrFeatureBorders = {
        {6.99999905f, 8.99999905f, 11.999999f},
        {11.999999f},
        {7.99999905f, 8.99999905f, 10.999999f, 12.999999f, 14.999999f},
        {4.99999905f, 6.99999905f, 7.99999905f},
        {4.99999905f, 5.99999905f, 6.99999905f},
        {6.99999905f, 7.99999905f, 8.99999905f, 10.999999f, 13.999999f},
        {1.99999905f, 12.999999f, 13.999999f},
        {0.999998987f, 4.99999905f, 5.99999905f},
        {4.99999905f, 5.99999905f, 10.999999f, 12.999999f},
        {7.99999905f, 12.999999f},
        {10.999999f},
        {1.99999905f, 4.99999905f, 13.999999f},
        {0.999998987f, 1.99999905f, 8.99999905f}
    };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[1066] = {
        0.0976442448024092, 0.0762376219034195, 0.08849009685218334, -0.08062234946659633, 0.08712871074676513, -0.06618030447708934, 0.07920791886069557, -0.251650167008241, 0, 0, 0, -0.09777227789163589, 0, -0.1564356446266174, 0, -0.1955445557832718,
        0.04731530500375747, 0, 0, 0, 0.01040230047035399, -0.2053217835724354, -0.1246525974780844, 0,
        0.009107778382593273, 0, 0, 0, 0.01733149280134318, -0.09360889733378104, 0, 0, 0.01503634570622422, 0, 0, 0, 0.01503634570622422, 0.01639773181164242, 0, 0, 0.02363687873542761, 0, 0, 0, 0.02864354419926268, 0.06037519783418676, 0, 0, 0.04031006819269875, -0.1447999067305404, 0, 0, 0.05310654291135954, -0.03858987882021467, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05988552020862699, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.07626237675547601, 0, -0.0525525993667543, 0, -0.08017326787114144,
        0, 0, 0, 0.01760459342167998, 0, 0, 0, 0, -0.03175111385129831, 0, -0.08771005813725134, 0.01013610143763505, -0.04951630623308248, 0, 0, -0.04293471928329256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.009388126828062976, 0, 0.01144750992980903, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02177344650505227, 0, 0.02223378619841136, 0, 0, 0, -0.1315802946260677, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03902370746173286, 0, -0.06413861429691316, 0, 0, 0, 0.007831876100399648,
        0.01095652039471122, -0.08682125213790973,
        0.0284169019741752, 0, 0, 0, -0.09601639460279633, 0, 0, -0.01631477459334287, 0.01250189981303198, 0, 0.04094519248214902, 0, 0.03112553734223509, -0.08534474502139142, -0.01281380360621756, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01618861494277337, 0.004657715352078109, 0, 0, -0.045401382274715, -0.05350219551647277, 0, 0, 0.01498496102896824, 0, 0, 0, 0, 0, -0.05261673699371252, 0, 0.005483326538510788, 0, 0.03668832017356015, 0, -0.04771458794295363, 0, -0.08523705088122241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0271695498047843, -0.03413391351666534, -0.0296912010506681, 0,
        0, -0.04220819835280077, 0.01042289175156823, 0.01433087780489609, 0, 0, 0, -0.06827579601711314, 0.00286399318848346, 0.003534356724370274, -0.08489184515480262, 0.01631068127076676, 0, 0, -0.03531408167700574, -0.04280175641317822,
        -0.05128221718429064, 0.01203870115213178, 0, 0.006604044946113261, -0.02898333747982898, 0.01090398963078069, 0.06144849507998924, 0, -0.08315903026978098, 0, 0, 0.02981036260834213, -0.009861167559549293, 0, 0, 0, 0, 0.007587770087910585, 0, 0, 0, 0, 0, 0, -0.05904996334983001, -0.0232172151594347, 0, 0.0740459397435463, 0, 0, 0, 0, 0.05498999434137532, 0.001822403216782779, 0.02925040970587103, 0.0006476668826891352, -0.08352736872755136, 0.03006344499301929, 0, 0, 0, 0.0007249690527456887, 0, -0.001975550168695883, -0.02687657017506906, 0, 0, 0, -0.002288791636446731, 0.008866434906846542, 0, 0.00252915915240258, 0, 0.01411143166207128, 0, 0, -0.01748086077184429, -0.004208776513088556, 0, 0.004462139123968145, -0.1035672841095364, 0.01484166911463299, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0005895573814605204, 0, 0.006668085867304774, 0, 0, 0, 0.007348298285487165, 0, 0, 0, 0.004387834751617359, 0, 0, 0, 0.002776417239773663, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004860816533881124, 0.006703143522984831, 0, 0, 0, -0.02640060021004609, -0.02835464303258891, 0.001310515208019651, -0.04585399580221331, 0.008608149946472689, -0.02814369630908069, 0.01599606387030541, -0.0391938568657806, -0.01294375875864308, 0, 0.02608406728229937, 0, 0, -0.07276415148605836, 0, 0, 0, 0.06766048444506202, 0.00307776096839546, 0, -0.02909063915072817, -0.01675490654531372, 0.006066926241298592, -0.04160399201244973, 0.03980453954509394,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007765329011693456, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.02228734603393393, 0, 0, 0, -0.03975056398254966, 0, 0, 0, 0, 0, 0, 0, -0.01062802959839381, 0, 0, 0, -0.002481956846988394, 0.05981462983246627, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.03019623738242279, 0,
        0.003567075365733652, 0, -0.0008692566240047495, 0, -0.002118490728056386, 0, 0.008860313804673245, 0, -0.03089737593412794, 0, 0, 0, 0.005652613495449843, -0.04285152018363539, 0.06433727357352663, -0.02071719272300616, 0, 0, 0, 0, 0.01219113442259298, 0, 0.02333534025864202, 0, -0.005358741190288618, 0, 0, -0.009299525898594582, -0.06345546104343168, 0, -0.06368415897927379, 0, 0.004337894334674589, 0, 0.02185289274555026, 0, -0.004330538662774856, 0, -0.0003743448208952116, 0, 0, 0, 0, 0, 0.03915033812971544, 0, -0.04160433457406398, 0, 0, 0, 0, 0, 0, 0, 0.002503133642458244, 0, 0, 0, -0.008946456029464195, 0, -0.01721311861030859, 0, 0.04809579089359957, 0,
        -0.03709848003530593, 0, 0.03391840483419127, 0, 0.03135033781062688, 0.01304653727643271, 0.01392294786366035, 0, 0, 0, 0, 0, -0.00468889854150254, -0.005456931469273868, 0, -0.001439219858797525, 0, 0, 0, 0, 0.00380756471830521, -0.003712970377482308, 0.04208381703189962, -0.001201987334840293, 0, 0, 0, 0, 0, 0.00516290418066614, -0.007157164823571356, 0.01382611295023392, 0, 0, 0, 0, 0, 0, 0, -0.0003656015720425176, -0.06577650656227374, 0, 0, 0, -0.01506147878402002, 0, 0, 0, 0, 0, 0, 0, -0.003582950802758086, 0.002311671168877672, -0.05653848037628974, 0.01551301729944608, 0, 0, 0, 0, 0.01440257790100259, 0.003415390385763874, -0.047156048593813, -0.003622097687153635,
        0, 0.006549067842000884, 0, 0, -0.04115531410730021, 0.003878010475070771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.004774815035614638, 0, 0, 0, -0.005097037143740719, 0, 0, 0, 0, 0, 0, 0, 0.03682333990291217, 0, 0, 0, 0, 0, 0, 0, -0.01449984543773262, 0, 0, 0, 0, 0, 0, 0, -0.004537227072612612, 0, 0, 0, 0, 0, 0, 0, -0.02752893764635896, 0, 0.04564368659956823, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.04378698954397369, 0, 0.02225168969500559, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.001088315401693574, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0007651313230195216, 0.0245642793095954, 0, 0, 0, 0, 0, 0, 0.03222042241504815, -0.003970073688536035,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0007773681440668304, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.03277122502123626, 0.0003675320173598606, 0, 0.02819286961316712, 0, 0, 0, 0, 0, -0.01933677857048679, 0, 0, 0, 0.01965142344767632, 0, -0.00347381447746903,
        -0.0007583444325187441, 0.001467590896176788, 0, 0.0257986196378194, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00234740439011126, 0.008918975830663407, 0.02223602179572808, -0.03982817004805301, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3.176972271751177e-05, 0.0002269345000389767, 0, 0.01555389978063851, 0, 0, 0, 0, 0, 0, 0, -0.03479067618308838, 0, 0.007835661919350024, 0, 0, 0, -0.002226179924517291, 0.02348284572985268, 0.01434628863798808, 0, 0, 0, 0, 0, -0.0001652916841294131, 0, -0.04105879847977123, 0.003807877274115717, 0.004496516491001246, 0.02573054576323927, 0.04726401907318758,
        0, 0, 0, 0, 0, 0, 0.0004847189588641637, 0.01799635708070568, 0, 0, 0, 0, 0, 0, 0.003708989202219876, 0.003331892614851256, 0, 0, 0, 0.00612082995113562, 0, 0, -0.01017836485400028, 0.001633129655467408, 0, 0, 0, 0.004025007944257031, 0, 0, 0, 0.001148248571021064, 0, 0, 0, 0, 0, -0.02067156234413778, 0, 0, 0, 0, 0, 0, -0.04714104688530915, 0, 0, 0, 0, -0.03434134899737722, 0, 0, -0.02368859128350309, 0.01123559048072924, 0, 0.01426346862480657, 0, 0, 0, 0, 0, 0.02341845658349919, 0, -0.005452093516428694,
        0.02764328514466814, 0, 0.01228988376329744, 0, -0.005167201264095327, 0, -0.009314201677604644, 0, 0, 0, 0, -0.002216004888915518, 0, 0, 0, 0.003521881951224909, -0.04546833291242548, -0.03066640798932052, 0.01921183839183777, 0, -0.004770582177078091, 0.004076142799933069, 0.001119067908020411, 0.01019383339042147, -0.01274725288256139, 0.01288570585042297, -0.003657508281645015, -0.004770581826875107, 0.000424129089006145, -0.005217892902688148, -0.009346994889167119, 0, 0, 0, -0.009609713590588546, 0, 0, 0, 0, 0, 0, 0, 0.02363037695864759, 0, 0, 0, -0.03004868037270507, 0, 0, 0, -0.03337687031449574, 0, 0.02676216099286354, 0, 0.01108889203280436, 0, 0, 0, 0.02161557762428626, 0, 0, 0, 0.01551741462345909, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007857921387467208, 0, 0, 0, 0, 0, 0, 0, -0.02629259532611693, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157523850234196, -0.0006086190082724625, 0, 0, 0, 0, -0.01188699833928054, 0.003710522382902411, -0.02276277909930262, 0.01153632117548773, 0, -0.02033184015409009, -0.001043234865236938, -0.02455174701738091, 0, 0, 0, 0.01956559580391726, 0, 0, 0, -0.00998402757419484,
        0, -0.03440574032194189, -0.01355652106541172, 0.0005487677645788972, 0, 0, 0, -0.003776899523871877, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01419054721638306, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.009177418215405691, 0, 0, 0, 0, 0, 0, 0, 0.02567336485665508, 0, 0, 0, 0, 0, 0, 0, -0.002293666872906442, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
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
                    {.BaseHash = 14216163332699387099ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 15},
                    {.BaseHash = 14216163332699387099ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 15},
                    {.BaseHash = 14216163332699387099ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 14216163332699387101ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 15},
                    {.BaseHash = 14216163332699387101ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 15},
                    {.BaseHash = 14216163332699387101ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 15},
                    {.BaseHash = 16890222057671696978ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Counter, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {6},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 16890222057671696977ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Counter, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 14216163332699387072ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 15},
                    {.BaseHash = 14216163332699387072ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 15},
                    {.BaseHash = 14216163332699387072ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 15},
                    {.BaseHash = 16890222057671696975ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Counter, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {10},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 16890222057671696973ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Counter, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 15}
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
                    16890222057671696973ull,
                    {
                        .IndexHashViewer = {{2136296385601851904ull, 0}, {7428730412605434673ull, 1}, {9959754109938180626ull, 3}, {14256903225472974739ull, 5}, {8056048104805248435ull, 2}, {18446744073709551615ull, 0}, {12130603730978457510ull, 6}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10789443546307262781ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 93,
                        .CtrMeanHistory = {{.Sum = 1.30321e-43, .Count = 2}, {.Sum = 2.8026e-45, .Count = 1}, {.Sum = 1.4013e-45, .Count = 1}},
                        .CtrTotal = {93, 2, 2, 1, 1, 1, 1}
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
                    16890222057671696977ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {14452488454682494753ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1388452262538353895ull, 5}, {8940247467966214344ull, 9}, {4415016594903340137ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {41084306841859596ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8678739366408346384ull, 4}, {18446744073709551615ull, 0}, {4544226147037566482ull, 12}, {14256903225472974739ull, 6}, {16748601451484174196ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5913522704362245435ull, 0}, {1466902651052050075ull, 3}, {2942073219785550491ull, 8}, {15383677753867481021ull, 2}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 16,
                        .CtrMeanHistory = {{.Sum = 1.54143e-44, .Count = 11}, {.Sum = 2.24208e-44, .Count = 2}, {.Sum = 8.40779e-45, .Count = 13}, {.Sum = 8.40779e-45, .Count = 16}, {.Sum = 1.4013e-45, .Count = 10}, {.Sum = 4.2039e-45, .Count = 5}},
                        .CtrTotal = {11, 11, 16, 2, 6, 13, 6, 16, 1, 10, 3, 5, 1}
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
