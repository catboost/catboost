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
        {18.5, 28.5, 34.5, 35.5, 36.5, 41.5, 45.5},
        {116831.5, 117562, 188654.5, 202819.5, 237801},
        {9.5, 11.5, 12.5, 13.5, 15.5},
        {1087, 3280, 5842, 7493},
        {808.5, 1738, 1881.5, 2189.5},
        {11.5, 17, 27, 31.5, 42, 44.5, 46.5, 70}
    };
    std::vector<unsigned int> TreeDepth = {6, 6, 5, 6, 6, 5, 6, 6, 6, 6, 6, 4, 6, 6, 4, 4, 6, 6, 4, 6};
    std::vector<unsigned int> TreeSplits = {31, 18, 24, 9, 57, 66, 65, 15, 20, 58, 25, 10, 64, 42, 18, 31, 2, 62, 13, 29, 38, 23, 39, 69, 45, 64, 43, 32, 14, 30, 5, 70, 41, 34, 55, 26, 8, 0, 42, 58, 40, 61, 18, 33, 2, 3, 18, 68, 37, 22, 27, 57, 18, 21, 11, 13, 37, 46, 69, 13, 11, 16, 4, 73, 54, 4, 60, 72, 55, 46, 24, 43, 19, 63, 52, 36, 48, 1, 7, 13, 67, 40, 44, 56, 45, 68, 71, 12, 31, 68, 50, 17, 35, 51, 52, 30, 36, 13, 18, 47, 56, 4, 49, 28, 38, 53, 10, 6, 59, 14};
    std::vector<unsigned char> TreeSplitIdxs = {7, 2, 4, 3, 1, 4, 3, 4, 4, 2, 1, 4, 2, 3, 2, 7, 3, 6, 2, 5, 5, 3, 6, 3, 1, 2, 4, 8, 3, 6, 6, 4, 2, 1, 1, 2, 2, 1, 3, 2, 1, 5, 2, 255, 3, 4, 2, 2, 4, 2, 3, 1, 2, 1, 5, 2, 4, 2, 3, 2, 5, 5, 5, 3, 1, 5, 4, 2, 1, 2, 4, 4, 3, 1, 1, 3, 2, 2, 1, 2, 1, 1, 5, 2, 1, 2, 1, 1, 7, 2, 4, 1, 2, 1, 1, 6, 3, 2, 2, 1, 2, 5, 3, 4, 5, 2, 4, 7, 3, 3};
    std::vector<unsigned short> TreeSplitFeatureIndex = {5, 3, 4, 1, 15, 16, 16, 2, 3, 15, 5, 1, 16, 8, 3, 5, 0, 15, 2, 5, 7, 4, 7, 17, 9, 16, 8, 5, 2, 5, 0, 17, 8, 7, 14, 5, 1, 0, 8, 15, 8, 15, 3, 6, 0, 0, 3, 17, 7, 4, 5, 15, 3, 4, 1, 2, 7, 9, 17, 2, 1, 2, 0, 18, 13, 0, 15, 18, 14, 9, 4, 8, 3, 16, 12, 7, 10, 0, 1, 2, 17, 8, 8, 14, 9, 17, 18, 2, 5, 17, 10, 3, 7, 11, 12, 5, 7, 2, 3, 10, 14, 0, 10, 5, 7, 12, 1, 0, 15, 2};
    std::vector<unsigned char> TreeSplitXorMask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<unsigned int> CatFeaturesIndex = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<unsigned int> OneHotCatFeatureIndex = {9};
    std::vector<std::vector<int>> OneHotHashValues = {
        {-2114564283}
    };
    std::vector<std::vector<float>> CtrFeatureBorders = {
        {2.99999905f, 3.99999905f, 4.99999905f, 5.99999905f, 6.99999905f, 11.999999f},
        {4.99999905f, 5.99999905f, 8.99999905f, 9.99999905f, 11.999999f},
        {7.99999905f, 12.999999f},
        {2.99999905f, 6.99999905f, 11.999999f, 13.999999f},
        {3.99999905f},
        {11.999999f, 12.999999f},
        {10.999999f},
        {12.999999f, 13.999999f},
        {1.99999905f, 3.99999905f, 4.99999905f, 6.99999905f, 8.99999905f, 9.99999905f},
        {4.99999905f, 8.99999905f, 11.999999f, 12.999999f},
        {11.999999f, 12.999999f, 13.999999f, 14.999999f},
        {11.999999f, 12.999999f, 13.999999f}
    };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[1024] = {
        0.783752480412105, 0.7821782231330872, 0.7763118865907118, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7848971871538577, 0.7687694538933721, 0.7727920846652867, 0.7821782231330872, 0.7763118865907118, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7820650688264799, 0.7712623816933298, 0.7727920846652867, 0.7763118865907118, 0.7821782231330872, 0.7763118865907118, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7877793544130413, 0.7847920843970657, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7880127348830394, 0.7859123106530566, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872, 0.7821782231330872,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004054640901644743, 0, -0.003405029688836914, 0, 0, 0, -0.005822339019290917, 0, -0.004037269580811403, 0.005670354432955627, -0.01321122934015241, 0.001591654806325933, -0.009273504808704451, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00158990445284007, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.001896341675047456, 0.005695476096085938, -0.009028520802652342, 0.00160565763421283, -0.005822339019290917, 0, 0, 0,
        0.001591446554575274, 0.003110080112758972, 0.002368091668484588, 0.005651903583681797, 0, 0, 0, -0.005726389220929591, 0, 0, -0.005666686557488831, 0.003582362793519962, 0, 0, 0, 0, -0.009367244000403662, -0.002366885184592214, 0.00090647973464968, 0.00541042483807386, -0.005778671477622281, 0, -0.009237903141965056, -0.005726389220929591, -0.01143579037170106, 0, -0.01056556436943594, -0.001719289658315743, 0, -0.005778671477622281, 0, 0,
        -0.003365347373424429, -0.001802296819179406, -0.006570920429154065, 0.003063428912110323, -0.005684476745291401, 0, -0.01129815665143798, 0, 0.002572497423290302, 0.004382415247161267, -0.003436272114006239, 0.003073878069885205, 0.002706145807726441, 0, 0, 0.001536262256565371, -0.00577430785962397, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0056440130704549, 0.001661647666395019, 0, 0, 0, 0, 0, 0, 0, 0, 0.001865512086927872, 0.005329675157077261, -0.00332137935321463, 0.001810669945791889, -0.004239661484484254, 0.002464590237914892, -0.008979965539512276, 0.003151340042845451, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.005624186409257617, 0,
        -0.009122238390705126, 0, 0.001788700299095135, 0.001604750810419269, -0.005731000551644785, 0.001578457247403578, -0.005819996458520395, 0.00317660708798352, 0, 0, -0.001821549206201169, 0.002486225529255614, 0, 0, 0.0003588297885850461, 0.005447859980496118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.005496181958626362, 0, -0.005546249230038205, -0.009135295247560363, -0.005650595269517056, 0.001500887066422601, 0, 0.003066496067856784, 0, 0, -0.008527921177224302, 0.001599457907737618, 0, 0, -0.003238268814277495, 0.003141061864412446, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00560168297337264, 0, 0, 0, 0, 0,
        0, 0, -0.009104444863835111, -0.01109106852336971, 0, 0, 0.001566618818312661, 0, 0, 0, -0.005776346486057145, 0, 0.002475326836762977, 0, -0.004594576546920599, 0, 0, 0, 0, 0, 0.001489630413676037, 0, 0, 0, 0.0009233138829555449, -0.008428419037197958, -0.003303877848244923, -0.005616600939513023, 0.005385659546080508, 0.003406116750434492, 0.002710097850271649, 0.002470368067577192,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.005630095482901776, 0, 0, 0.001574150228749378, -0.005583075504234761, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.003197777899113388, 0, 0, 0, 0, 0.002519985122473655, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004310089914467049, -0.01553063796110536, 0.001385513915681356, 0, 0, 0, 0, 0, 0.002768442810089207, 0, 0.001505918570563352, 0, 0.003914418615756548, -0.003114334083659907, 0.001458060460340963, 0, 0.005243510962932889, -0.005082264890699077,
        0, -0.0006018988969443422, 0, 0.004723221032183618, 0, 0, 0, -0.00564154217959738, 0, 0.00405139011696792, 0, 0.004670853336404836, 0, 0, 0, 0, 0, 0.001652854913147926, 0, 0.001434360927340604, 0, 0, 0, 0, 0, 0, 0, 0.001434360927340604, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.01075789087002571, -0.00415322039236337, 0, 0.002362626319681295, 0, -0.01280283139291871, -0.005408553011772181, 0, 0, -0.00558786976772383, -0.003184569142087147, 0.004225229403547064, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0.001440984241564426, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00321120165464247, -0.005445181205588288, 0.002694204472818779, 0, 0.00240605234173029, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002669362373341012, 0, 0.004075305285749032, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.01342197324521837, -0.005367988865090568, 0.003809179067948127, 0, -0.001326473036771662, -0.0110710953180816, 0.005050254931965011, -0.005568276435890602, -0.005479370663379108, 0, 0, 0, -0.008264515831086996, 0, 0.001336376167341697, 0,
        -0.005326425937457565, 0, -0.005438275384322316, 0, -0.002939644547047404, 0, 0, 0, -0.005073948823876418, -0.005327728949502269, 0, 0, -0.00306536148289711, 0, 0, 0, 0.002569900261537024, 0, 0, 0, 0.004230242915665973, 0, 0, 0, 0.00317180678094762, 0, 0, 0, -0.004824298796217501, 0, 0, 0, 0.002442191506978328, 0, 0, 0, -0.005533599006187728, 0, 0, 0, -0.001628967136187236, -0.005404342347459196, 0, 0, 0, 0, 0, 0, 0.003448572968234284, -0.008673076749493528, 0.001326353346310661, 0, 0.003660518910082021, 0, 0.001595379192353279, 0, 0.004058110494381068, -0.00883416941177707, -0.01067529681468698, 0, 0.0006880983951071479, 0, 0.002516435298739455, 0,
        0, 0.003445377997206461, 0, 0.001276670151477802, 0, 0.001988437045943963, -0.003282066922041323, 0.001380565571634928, 0, 0, 0, 0, 0, 0, 0, 0, 0.001664419304646416, 0.002558186306724251, 0.001710767777612482, 0.001703690782262836, 0.001619144788916292, 0, 0, 0.001317792290890214, 0, 0, 0, 0, 0, 0, 0, 0, 0.004170169068052887, 0.004334020954904173, -0.006119433443905164, 0.003774536319484191, 0.002612466439389354, 0.003138994932916815, 0, 0.002802532741276463, 0, 0, 0, 0, 0, 0, 0, 0, -0.007923582334752316, 0.003352632338841635, -0.01189228806624302, -0.006763153157928292, -0.004011748294349467, 0.002838743375122805, -0.01077119681834703, 0.002746437582991726, 0, 0, 0, 0, 0, 0, 0.001779104722700356, 0,
        -0.003242682119857146, 0.002312286518068629, 0, 0.001697937019567178, 0, 0.003585688953144833, 0.00160700120327085, 0.003746879280099171, 0.004029512930561485, 0.002283488271947024, -0.007402768462335289, -0.008617617572752511, -0.002845067962249733, 0.004582103528082507, -0.01484261880817947, 0.003132609147027965,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001685202492205063, 0.001571377494666611, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00338616198514367, -0.009070152470989535, 0.003384909247267386, -0.008595994373473453, 0, -0.005226741815096054, 0, 0, 0.004062687561440687, -0.002010497901354946, 0.004611606776741504, -0.002267922719845288, 0, 0, 0, -0.005105719319087209, 0, -0.005087259186197751, 0, -0.005248454002857522, 0, 0, 0, 0, 0, 0, 0, -0.01072699773586021, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0.002000389823352013, 0.002770886284705806, 0, 0, 0, 0, 0, 0, -0.0002542502184714814, 0.002388150039313043, 0, 0, 0, 0.002650169028920713, 0.001469156951391694, 0, 0.003526516145461285, 0.003506214146455734, -0.009566921895692592, 0.001469724905836752, -0.001318607288863869, 0.002736120283714187, 0, 0, 0.0008189152581051007, 0.003100587536277855, 0, 0, 0, 0, 0, 0, 0, 0.001198850557069416, 0, 0, 0, 0, 0, 0, -0.005200668945903532, 0.002457328364848023, 0, 0.00132474149608083, -0.005683740839099332, 0.002448324046278131, 0, 0, 0.001559592163720034, 0.002500790157636819, -0.009386879772387788, 0.00131110813724118, -0.00549284286280213, 0.002154836003623513, 0, 0, -0.009333544101952607, 0.003277076605983365,
        0, 0.001458701969289358, 0, 0.004347314960510535, 0, 0, 0.001654133511471611, 0.00445878221110099, -0.01122993727499108, -0.00511578933974512, -0.007838329159670136, -0.005731765694152497, 0, 0, -0.003453801487929143, -0.004309610181454544,
        0, 0, 0, 0.003351055138504741, -0.007975319661850376, 0.001825210067769636, 0, 0.004036394067681778, 0, 0.0005793746255166118, 0, 0.003523066463246379, -0.009615842309171055, -0.009191696913392547, 0.002183970222697176, 0.001481907552281097,
        0, 0, 0.001648798426282287, 0, 0, 0, 0.001400400500050468, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002598575412266936, 0.001148820981899794, 0, 0, 0, 0, 0, 0, 0, 0, -0.01001259252557645, -0.007508037770870456, 7.021444834945563e-05, 0, 0, 0, 0, 0, 0, -0.004822359803048105, -0.005106815066645023, 0, 0, 0, 0, 0, -0.002939417639076111, -0.005024956222267081, 0.003700212477972542, 0.001184386708042355, 0.003545409747692671, -0.01278509978245201, 0.004088230325902904, 0.002801044610042925, -0.005089576944244081, 0, 0, 0, -0.008251604669170439, 0, -0.00314020427007904, 0,
        -0.007646230767772927, 0.00143085532010926, -0.004732326725864893, 0, 0.001122264554812672, 0.003116118915937864, 0.002193834351463087, 0, -0.002873321907151558, 0.002017858991331487, -0.004540099923677386, 0, -0.005564624994169313, 0.00271342150269137, -0.005122508642310016, 0.001175503807930586, 0, 0, 0, 0, -0.005051405118015457, 0, 0, 0, -0.005068513954501282, 0, -0.004786192105333656, 0, 0, 0, 0, 0, 0.00138989749653485, 0, 0, 0, 0.003806651062366121, 0.003723495731952642, -0.0008202046257291956, 0.001100216655144491, 0, 0, 0, 0, -0.0003637262204940326, 0.003596475132668466, -0.009777770560139966, 0.002478386717338391, 0, 0, 0, 0, 0, 0, -0.00512162529118055, 0, 0, 0, 0, 0, -0.008245537705456985, 0, 0, 0,
        0.001641483803279961, 0, 0, 0, 0.003489540240027374, 0.002628333283906627, 0.001123244436668467, 0.001572233996379132, 0.003249376579038061, -0.005522890307645885, 0.00379418336283504, -0.01230410622668977, 0.003723264186610606, 0.0008406593859389201, 0.003590066883320619, -0.006313454830314836,
        0.001353301714328614, 0.002026059471033714, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001921679039948161, 0, 0.003251534380687749, 0, -0.001120261425492579, 0, 0.003254637005859013, -0.003616547372481574, -0.002160855004741608, 0, 0.001701222556486485, -0.007639360414803879, 0.004761359143409818, 0.001391667530863093, 0.001685826958124537, 0, 0.002333867378408914, 0, 0, 0, 0, 0, 0, -0.008041978735916947, 0, 0, 0.00155689256439009, 0, 0, 0, 0, 0, -0.007688448451639366, 0.001217524784317293, 0.002575837260911119, 0, -0.002835622792484323, 0.001849310881830333, 0.00255814496592469, -0.004413768380373362, -0.007913300241136826, 0, 0, -0.007401456589373651, -0.002568064577543774, 0, 0.002432539402310483
    };
    struct TCatboostCPPExportModelCtrs modelCtrs = {
        .UsedModelCtrsCount = 12,
        .CompressedModelCtrs = {
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 14216163332699387099ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387099ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = -0, .Scale = 15},
                    {.BaseHash = 14216163332699387099ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = -0, .Scale = 15},
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
            }
        },
        .CtrData = {
            .LearnCtrs = {
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
