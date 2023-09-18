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
    CatboostModel() = default;
    unsigned int FloatFeatureCount = 6;
    unsigned int CatFeatureCount = 11;
    unsigned int BinaryFeatureCount = 19;
    unsigned int TreeCount = 20;
    std::vector<std::vector<float>> FloatFeatureBorders = {
        {32, 33.5, 34.5, 35.5, 50, 53.5, 56, 58.5},
        {51773, 119180.5, 128053.5, 173910, 198094.5, 200721, 215992, 216825, 292939, 318173.5, 325462, 337225.5},
        {4.5, 9.5, 10.5, 11.5, 12.5, 13.5, 15.5},
        {1087, 3280, 7493, 11356, 17537.5},
        {808.5, 1622.5, 1862, 2189.5, 2396},
        {17, 27, 36.5, 44.5, 46.5}
    };
    unsigned int TreeDepth[20] = {6, 5, 6, 6, 4, 6, 6, 6, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
    unsigned int TreeSplits[115] = {68, 28, 22, 3, 13, 32, 69, 15, 22, 9, 34, 69, 1, 23, 27, 39, 34, 75, 55, 21, 72, 41, 83, 28, 45, 38, 30, 46, 28, 42, 36, 76, 51, 35, 28, 66, 2, 59, 48, 62, 24, 47, 16, 40, 78, 41, 74, 63, 67, 56, 73, 62, 54, 82, 17, 8, 50, 57, 69, 19, 58, 8, 29, 37, 28, 31, 26, 28, 20, 7, 49, 6, 79, 52, 64, 33, 28, 14, 83, 8, 0, 72, 60, 65, 18, 52, 64, 33, 10, 81, 65, 25, 44, 61, 53, 77, 40, 25, 28, 61, 44, 53, 12, 70, 22, 11, 5, 4, 43, 71, 64, 80, 22, 41, 73};
    unsigned char TreeSplitIdxs[115] = {1, 2, 3, 4, 6, 1, 2, 8, 3, 2, 3, 2, 2, 4, 1, 3, 3, 4, 2, 2, 1, 5, 3, 2, 1, 2, 4, 2, 2, 1, 5, 5, 2, 4, 2, 3, 3, 3, 4, 3, 5, 3, 9, 4, 1, 5, 3, 4, 4, 3, 2, 3, 1, 2, 10, 1, 1, 1, 2, 12, 2, 1, 3, 1, 2, 5, 7, 2, 1, 8, 5, 7, 2, 3, 1, 2, 2, 7, 3, 1, 1, 1, 1, 2, 11, 3, 1, 2, 3, 1, 2, 6, 3, 2, 4, 1, 4, 6, 2, 2, 3, 4, 5, 1, 3, 4, 6, 5, 2, 2, 1, 3, 3, 5, 2};
    unsigned short TreeSplitFeatureIndex[115] = {13, 3, 2, 0, 1, 4, 13, 1, 2, 1, 4, 13, 0, 2, 3, 5, 4, 15, 9, 2, 15, 5, 18, 3, 7, 5, 3, 7, 3, 6, 4, 15, 8, 4, 3, 12, 0, 10, 7, 11, 2, 7, 1, 5, 17, 5, 15, 11, 12, 9, 15, 11, 9, 18, 1, 1, 8, 10, 13, 1, 10, 1, 3, 5, 3, 3, 2, 3, 2, 0, 7, 0, 17, 8, 12, 4, 3, 1, 18, 1, 0, 15, 11, 12, 1, 8, 12, 4, 1, 18, 12, 2, 6, 11, 8, 16, 5, 2, 3, 11, 6, 8, 1, 14, 2, 1, 0, 0, 6, 14, 12, 17, 2, 5, 15};
    unsigned char TreeSplitXorMask[115] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    unsigned int CatFeaturesIndex[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<unsigned int> OneHotCatFeatureIndex = {};
    std::vector<std::vector<int>> OneHotHashValues = {
    };
    std::vector<std::vector<float>> CtrFeatureBorders = {
        {7.99999905f, 10.999999f, 11.999999f},
        {4.99999905f, 6.99999905f, 8.99999905f, 10.999999f, 11.999999f},
        {4.99999905f, 10.999999f, 12.999999f, 14.999999f},
        {1.99999905f, 6.99999905f, 13.999999f},
        {5.99999905f, 8.99999905f, 11.999999f},
        {6.99999905f, 8.99999905f, 9.99999905f, 10.999999f},
        {7.99999905f, 8.99999905f, 10.999999f, 14.999999f},
        {12.999999f, 13.999999f},
        {4.99999905f, 12.999999f},
        {2.99999905f, 8.99999905f, 9.99999905f, 10.999999f, 11.999999f},
        {11.999999f},
        {7.99999905f, 8.99999905f, 10.999999f},
        {10.999999f, 13.999999f, 14.999999f}
    };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[1136][1] = {
        {0.08969131988637588}, {0.06223479339054653}, {0}, {0}, {0.0762376219034195}, {0.02722772210836411}, {0}, {-0.09777227789163589}, {0.07260725895563762}, {-0.05693069472908974}, {0}, {-0.09777227789163589}, {0.05445544421672821}, {-0.1564356446266174}, {0}, {-0.1564356446266174}, {0.08712871074676513}, {0.04356435537338257}, {0}, {0}, {0.07260725895563762}, {-0.05643564462661743}, {0}, {0}, {0.06806930527091026}, {-0.03160700889734121}, {0}, {-0.09777227789163589}, {0.06223479339054653}, {-0.2234794923237392}, {0}, {-0.09777227789163589}, {0}, {0}, {0}, {0}, {0}, {-0.09777227789163589}, {0}, {0}, {0.02722772210836411}, {-0.09777227789163589}, {0}, {0}, {0}, {-0.1564356446266174}, {0}, {0}, {0}, {0.02722772210836411}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.02722772210836411}, {0.02722772210836411}, {0}, {0},
        {0.04814598989571692}, {-0.05403553864120372}, {0}, {0}, {0.0420084855386189}, {-0.07821782231330872}, {0}, {0}, {0.05696066158914977}, {-0.05809299374158262}, {0.05481848051150641}, {0.04315340051379724}, {0.05426685393372655}, {-0.1840594074875116}, {0.05764281644766767}, {-0.1492377822361295}, {0}, {0}, {0}, {0}, {0}, {-0.08555074315518141}, {0}, {0}, {0.02382425684481859}, {-0.08555074315518141}, {0}, {0}, {0}, {-0.07821782231330872}, {0}, {0.02382425684481859},
        {0.03137505026093929}, {0.01638800262246663}, {0.01831878474066216}, {0.04125507705009137}, {0.01751434029272152}, {0.05293690046620747}, {0.01661890478886013}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.05521039637736977}, {0.02999070495693095}, {0.05722044698612203}, {0.03512631390206653}, {-0.0414609668345468}, {0.03136821194925489}, {-0.02018513340242799}, {0.03791053492085137}, {-0.1477374413305962}, {0}, {0}, {0}, {-0.1252625902999737}, {0.0124469586781093}, {-0.06254331721924246}, {0}, {-0.09420297087728978}, {0}, {0}, {0.02084622473921627}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.07485690026078373}, {0}, {0}, {0}, {-0.08163418003047505}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0},
        {-0.06549978772818577}, {0}, {0.01955744127897836}, {0}, {-0.07472675150042332}, {0}, {0.04935482445606104}, {0}, {0.0142396712098349}, {-0.1382014670843243}, {0.02358570690774493}, {0.04579733696938204}, {0.02122725324386309}, {-0.02526463832556897}, {0.0221972921029005}, {-0.01427810143625929}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.00546771244675838}, {0}, {0.006993636135063985}, {-0.08599816720133265}, {0.01575510044604313}, {-0.1471036584734151}, {0}, {0}, {0}, {0}, {0}, {0}, {0.04540056154541666}, {0}, {0.01550343738321164}, {0}, {0.01467774946844021}, {0}, {0.01271380328761186}, {-0.06953995785848903}, {0.01966783094400374}, {0.02209382114019543}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0},
        {0}, {0}, {0.02485353239785594}, {0}, {-0.08758856393918904}, {0}, {0.008503756560655237}, {-0.1205132116785034}, {0}, {0}, {0}, {0}, {0}, {-0.04001817495941466}, {0}, {-0.06813200444230288},
        {-0.017034550909499}, {0.00713063762148638}, {0}, {0}, {0}, {0.005898955476371634}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.05989443675422698}, {-0.0230286370752431}, {-0.03501590308948782}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.006353840160328333}, {0}, {-0.02557768052486777}, {0}, {0.01795276556315803}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.002680859579710611}, {0}, {0}, {0}, {-0.002991506273515525}, {0}, {-0.1017895354604294}, {0}, {0}, {0}, {0}, {0}, {-0.04854952069025106}, {0}, {0},
        {-0.06153649441207036}, {0}, {0}, {0}, {0.002870884907710092}, {0}, {0}, {0}, {-0.05512354699005387}, {-0.04423442490607654}, {-0.03063891520330184}, {0}, {-0.01961758891819}, {0}, {-0.0223804704592593}, {0}, {0}, {0}, {0}, {0}, {0.008477028389807517}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.01823529362428816}, {0}, {0}, {0}, {0.02481038162103719}, {0}, {-0.02515279649493814}, {0}, {0.008449552785771723}, {0}, {0}, {0}, {0.002666119939088231}, {-0.04248083060396968}, {-0.05780279267657373}, {0}, {0.03471670150635539}, {0}, {-0.02983063198488144}, {0}, {0}, {0}, {0}, {0}, {-0.00170719236995574}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.006844769977370683}, {0}, {0}, {0},
        {0.00960390073798285}, {0}, {0}, {0}, {0}, {0.0008728027915176351}, {0}, {0.004069350611647038}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.05985839825029885}, {0.0068864721943388}, {0}, {-0.02005248000937357}, {0.03632123074220588}, {-0.002586762307199462}, {0.01181472919830097}, {-0.007139962789741505}, {-0.07059519135269535}, {0.01562103064537079}, {0}, {0}, {0.001712187178407122}, {-0.0007225195192925415}, {-0.02868036640640071}, {0.004117601697830442}, {-0.009924408356733642}, {0}, {-0.02198134212266723}, {0}, {-0.0407668494831109}, {0.001690582171681387}, {-0.06968746215392529}, {0.00280525151927267}, {0}, {0}, {-0.01325961406708098}, {0}, {0.05767958141848391}, {0}, {0}, {0.02441062573430942},
        {-0.0561611500110486}, {0}, {-0.07368760119967385}, {-0.01335898100805229}, {0}, {0}, {0}, {0}, {0.008292364035984379}, {0}, {0}, {0}, {0.01012894707868855}, {0.001402081364069302}, {0}, {0}, {0.05538608732483016}, {0}, {0.02372690254668692}, {-0.08691126691204111}, {0}, {0}, {0}, {0}, {0.002736348657113882}, {0.003910595570242384}, {0.01051033440956427}, {0}, {0.0002876826474241352}, {0.0008879501277842856}, {-0.02520930763805357}, {0},
        {-0.02420267826014854}, {-0.0560550863892121}, {-0.005377827301768664}, {0}, {0.02482237360855009}, {-0.01433784153177243}, {-0.0007976069136870404}, {0}, {0.007255818531486324}, {0}, {0.0008406615668154413}, {0}, {0.0009694931198592216}, {0.03335065419679564}, {0.003371834089740268}, {-0.02205814418329687}, {0}, {0.04233932111062674}, {0.006569638527007186}, {0.03530074915597982}, {0}, {-0.008234875700857758}, {0.003519708006153572}, {0}, {0}, {-0.04484750585790322}, {0}, {0}, {0.01952069173506177}, {0}, {-0.00299558972359342}, {0},
        {0}, {0}, {0}, {0.004100976183663983}, {0}, {0}, {0}, {0.0006374990037856719}, {0}, {0}, {0}, {0.03468420304281686}, {0}, {-0.008336435310466756}, {-0.04360122209426692}, {-0.01290216892790015}, {0}, {0}, {0}, {-0.003094570252148876}, {0}, {0}, {0}, {-0.001576608665089685}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.05448781454476914}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.0006751666394972933}, {0.002453744434088276}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.06161545524222856}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.003950910133208789}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.03088815551148234},
        {0}, {-0.0008104090689676768}, {0}, {0}, {-0.03099347211363495}, {0.00266471785150467}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.0311818758241913}, {0}, {-0.01754798606224186}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.01768810504439724}, {0}, {0}, {0}, {0}, {0}, {0.02702713607254704}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0},
        {0}, {0}, {-8.313562479854708e-05}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.002463434407346476}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.005503129460654441}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.004241684982447175}, {0}, {-0.0002966850410983842}, {0.0005114506764807704}, {0}, {0}, {0}, {0}, {0.02649347831166644}, {0}, {-0.00392533003669421}, {-0.03093902495067946}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.02896983395342578}, {-0.00548919444983886}, {-0.001365947007928747}, {0}, {0.0003469334362535964}, {0}, {0}, {0}, {0.03995120460871124}, {0}, {-0.004631303402152342}, {0}, {0.02917917740715937}, {-0.01819914167708286},
        {0.0317146704147269}, {-0.028362280757402}, {-0.004359222716451595}, {0.002471331149468683}, {-0.02043129316512242}, {0.003759863783706727}, {0}, {-0.03041648514274837}, {0}, {-0.02003168399657681}, {-0.004803045143609002}, {-0.00835907446872672}, {0}, {0}, {0}, {0}, {0.03305596989101085}, {-0.01317306683331073}, {0.005547577551451927}, {-0.009946997595045394}, {0}, {0}, {0}, {0.01720393652859675}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.008135190329616182}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.01320219920421223}, {0}, {0}, {0}, {0}, {-0.03792480061097859}, {0}, {0.000257665141464955}, {0.00648728239043565}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0},
        {0}, {0}, {0}, {0}, {0}, {0.001627490649391439}, {0}, {0.005482651492801749}, {0}, {0}, {0}, {-0.01787738151948211}, {0}, {0.0004839787289733499}, {0}, {-0.01758595118883557}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.02673564090993956}, {0}, {-0.009543429624283586}, {0.003465163065547422}, {-3.555438991205856e-05}, {-0.03409513545737599}, {0.0003439604678531014}, {0}, {0}, {0}, {0}, {0}, {-0.03318420053460627}, {0}, {0.01128415732682085}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.0258079288160251}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.01515095139345876}, {0}, {0}, {0}, {-0.003813170301687574}, {0}, {0.00672817952438575},
        {0.0206257027195489}, {0.05196523184802451}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.01888459503301396}, {0.0004234813878516847}, {0}, {0}, {0}, {0}, {0}, {0}, {0.03189963102759652}, {0}, {-0.008357826537310859}, {0}, {0}, {0}, {0}, {-0.02091727593101802}, {0.01697559445261429}, {-0.05556206906621928}, {-0.0001038401646781214}, {0}, {-0.01564270882954685}, {0.005488124709347833}, {0}, {0}, {0}, {0}, {-0.002264695496177527}, {0.003897699342679885}, {0}, {0}, {0}, {0}, {0}, {0}, {0.0006935223106049881}, {-0.001650411558129977}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.01405675334124485}, {0}, {0}, {0}, {0}, {0}, {0}, {0.002306226752351664}, {0.004811689183151976}, {0}, {0}, {0}, {0.002959052750774173},
        {-0.01710295559869078}, {0}, {0.009270608066576173}, {-0.0177439460338205}, {0.007239228996485425}, {0.003535077653689947}, {0.004213809822122315}, {0}, {0.002098237313651311}, {0}, {-0.000829296889079649}, {0}, {-0.01200308225392508}, {-0.01820314490927568}, {-0.002549675567256502}, {0}, {0}, {0}, {0.01265562132138241}, {0}, {0}, {0}, {0.003603600783342772}, {-0.001325842807518712}, {0}, {0}, {0}, {0}, {-0.002338655899955641}, {0.004442426521202568}, {-0.01219638051006133}, {0}, {0.01775718286080228}, {0.005654149411824561}, {0.002051247885930354}, {-0.00651237824925728}, {-0.002381233326569663}, {0}, {-0.01365432070092539}, {-0.002641374606914298}, {0}, {-0.01462350395492341}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.02032443359314638}, {0}, {0},
        {-0.01337978363706224}, {0.006451778232395988}, {0}, {0}, {0.002826972837403324}, {0.003093192946978698}, {0}, {0}, {0.03886265292225098}, {-0.009497930205317676}, {0}, {0}, {0.00516352426082919}, {-0.001160112456578877}, {-0.01097522119615851}, {0}, {0.001573677985238479}, {0}, {0}, {0}, {0.003397973911184632}, {-0.01592775179561622}, {0}, {0.003887123206052247}, {0}, {0}, {0.003870607334513622}, {0}, {0.0008741213354439697}, {0}, {-0.01067183294630367}, {0}, {0.007487553090723776}, {-0.0007976288205714264}, {0}, {0}, {0.004689169136988233}, {0}, {-0.004236283737362264}, {0}, {-0.01418387907066754}, {-0.008783972442688231}, {-0.00865160547713772}, {0}, {-0.002604301128306802}, {-0.002311202781050015}, {0}, {0}, {0}, {-0.01279556596055798}, {0}, {0}, {-0.01767043618030537}, {0.01778387939400308}, {0}, {0}, {-0.00469990422359326}, {0}, {0}, {0}, {-0.002994005078886031}, {0}, {0}, {0},
        {0}, {-0.006489709017711008}, {-6.343314996296034e-05}, {0.008649921817256587}, {-0.002869739685386397}, {-0.0009503885302120929}, {-0.01151624399288695}, {0.001457342233881273}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.01393678282116419}, {0}, {0.01730439641350441}, {0}, {-0.0006979252179999981}, {0}, {0.003134389804782889}, {0}, {0}, {0}, {-0.01390643943880496}, {0.002694893868133749}, {-0.004757643764517387}, {-0.006728981282537239}, {0.005789434689974279}, {0}, {-0.0007615867501832235}, {-0.07413875796983813}, {0.001532492183200656}, {-0.00338057610798756}, {0.01136347630055098}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.001102862133762234}, {-0.02701479980253636}, {0}, {0}, {0}, {0.02731897863214645}, {0}, {-0.01442795622425691}, {0}, {0.02760734403210737}, {0}, {-0.008282361782017793}, {0}, {0.01987058790130948}, {-0.004962994201333892}, {-0.005365135275098964},
        {0}, {0}, {-0.003215503514911955}, {-0.004104710017634727}, {-0.009065814194170489}, {0}, {-0.001062735404497217}, {0.001268475874677219}, {0}, {0}, {0.003163262406789857}, {0.003909863516379048}, {0}, {0}, {0.002569257840883517}, {0.006894368077603033}, {0}, {0}, {0}, {-0.006779921257024007}, {0}, {0}, {0.002253657930262082}, {0}, {0}, {0}, {0}, {-2.571959771949794e-05}, {0}, {0}, {0.0004005129987732087}, {-0.004013474751067558}, {-0.02189584174153003}, {0.04357186434716865}, {0}, {0}, {0.008148028526043025}, {0.02631602487545219}, {-0.01532629738157314}, {0.02774827335357394}, {0}, {0}, {0}, {0}, {-0.03291290452638026}, {0}, {0.01700042496886357}, {-0.01424854304650343}, {-0.04545322678067419}, {0}, {0}, {0}, {0.01514134686181635}, {0}, {-0.02363794982721932}, {0}, {0.00299173549709681}, {0}, {-0.0006106845657499983}, {0}, {0}, {0.007685209907145727}, {-0.016206129539046}, {-0.01159226463548056}
    };
    double Scale = 1;
    double Biases[1] = {0.7821782231};
    unsigned int Dimension = 1;
    struct TCatboostCPPExportModelCtrs modelCtrs = {
        .UsedModelCtrsCount = 13,
        .CompressedModelCtrs = {
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {3},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 2967152236118276030ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 15},
                    {.BaseHash = 2967152236118276030ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 15},
                    {.BaseHash = 2967152236118276030ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 15},
                    {.BaseHash = 4017420253906208356ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Counter, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {5},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 2967152236118276024ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 15},
                    {.BaseHash = 2967152236118276024ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 15},
                    {.BaseHash = 2967152236118276024ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 15},
                    {.BaseHash = 4017420253906208354ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Counter, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {6},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 4017420253906208353ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Counter, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {7},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 4017420253906208352ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Counter, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 15}
                }
            },
            {
                .Projection = {
                    .transposedCatFeatureIndexes = {8},
                    .binarizedIndexes = {},
                },
                .ModelCtrs = {
                    {.BaseHash = 2967152236118276005ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0, .PriorDenom = 1, .Shift = 0, .Scale = 15},
                    {.BaseHash = 2967152236118276005ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 0.5, .PriorDenom = 1, .Shift = 0, .Scale = 15},
                    {.BaseHash = 2967152236118276005ull, .BaseCtrType = ECatboostCPPExportModelCtrType::Borders, .TargetBorderIdx = 0, .PriorNum = 1, .PriorDenom = 1, .Shift = 0, .Scale = 15}
                }
            }
        },
        .CtrData = {
            .LearnCtrs = {
                {
                    2967152236118276005ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8473802870189803490ull, 2}, {7071392469244395075ull, 1}, {18446744073709551615ull, 0}, {8806438445905145973ull, 3}, {619730330622847022ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 12}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 2.94273e-44, .Count = 61}, {.Sum = 0, .Count = 1}},
                        .CtrTotal = {0, 12, 1, 5, 21, 61, 0, 1}
                    }
                },
                {
                    2967152236118276024ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13987540656699198946ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18089724839685297862ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10353740403438739754ull, 2}, {3922001124998993866ull, 0}, {13686716744772876732ull, 1}, {18293943161539901837ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 0, .Count = 37}, {.Sum = 0, .Count = 4}, {.Sum = 3.08286e-44, .Count = 20}, {.Sum = 0, .Count = 13}, {.Sum = 0, .Count = 2}, {.Sum = 0, .Count = 3}},
                        .CtrTotal = {0, 37, 0, 4, 22, 20, 0, 13, 0, 2, 0, 3}
                    }
                },
                {
                    2967152236118276030ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {15379737126276794113ull, 5}, {18446744073709551615ull, 0}, {14256903225472974739ull, 2}, {18048946643763804916ull, 4}, {2051959227349154549ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {7024059537692152076ull, 6}, {18446744073709551615ull, 0}, {15472181234288693070ull, 1}, {8864790892067322495ull, 0}},
                        .TargetClassesCount = 2,
                        .CounterDenominator = 0,
                        .CtrMeanHistory = {{.Sum = 1.4013e-44, .Count = 58}, {.Sum = 1.4013e-45, .Count = 6}, {.Sum = 1.4013e-45, .Count = 5}, {.Sum = 4.2039e-45, .Count = 6}, {.Sum = 0, .Count = 4}, {.Sum = 2.8026e-45, .Count = 0}, {.Sum = 7.00649e-45, .Count = 0}},
                        .CtrTotal = {10, 58, 1, 6, 1, 5, 3, 6, 0, 4, 2, 0, 5, 0}
                    }
                },
                {
                    4017420253906208352ull,
                    {
                        .IndexHashViewer = {{3607388709394294015ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18356215166324018775ull, 0}, {18365206492781874408ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {14559146096844143499ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {11416626865500250542ull, 3}, {5549384008678792175ull, 2}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 36,
                        .CtrMeanHistory = {{.Sum = 1.96182e-44, .Count = 22}, {.Sum = 3.08286e-44, .Count = 36}, {.Sum = 7.00649e-45, .Count = 2}},
                        .CtrTotal = {14, 22, 22, 36, 5, 2}
                    }
                },
                {
                    4017420253906208353ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {14452488454682494753ull, 1}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {1388452262538353895ull, 5}, {8940247467966214344ull, 9}, {4415016594903340137ull, 11}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {41084306841859596ull, 7}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {8678739366408346384ull, 4}, {18446744073709551615ull, 0}, {4544226147037566482ull, 12}, {14256903225472974739ull, 6}, {16748601451484174196ull, 10}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {5913522704362245435ull, 0}, {1466902651052050075ull, 3}, {2942073219785550491ull, 8}, {15383677753867481021ull, 2}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 16,
                        .CtrMeanHistory = {{.Sum = 1.54143e-44, .Count = 11}, {.Sum = 2.24208e-44, .Count = 2}, {.Sum = 8.40779e-45, .Count = 13}, {.Sum = 8.40779e-45, .Count = 16}, {.Sum = 1.4013e-45, .Count = 10}, {.Sum = 4.2039e-45, .Count = 5}},
                        .CtrTotal = {11, 11, 16, 2, 6, 13, 6, 16, 1, 10, 3, 5, 1}
                    }
                },
                {
                    4017420253906208354ull,
                    {
                        .IndexHashViewer = {{18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {13987540656699198946ull, 4}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18089724839685297862ull, 5}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}, {10353740403438739754ull, 2}, {3922001124998993866ull, 0}, {13686716744772876732ull, 1}, {18293943161539901837ull, 3}, {18446744073709551615ull, 0}, {18446744073709551615ull, 0}},
                        .TargetClassesCount = 0,
                        .CounterDenominator = 42,
                        .CtrMeanHistory = {{.Sum = 5.1848e-44, .Count = 4}, {.Sum = 5.88545e-44, .Count = 13}, {.Sum = 2.8026e-45, .Count = 3}},
                        .CtrTotal = {37, 4, 42, 13, 2, 3}
                    }
                },
                {
                    4017420253906208356ull,
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
std::vector<double> ApplyCatboostModelMulti(
    const std::vector<float>& floatFeatures,
    const std::vector<std::string>& catFeatures
) {
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
    std::vector<double> results(model.Dimension, 0.0);
    const unsigned int* treeSplitsPtr = model.TreeSplits;
    const auto* leafValuesPtr = model.LeafValues;
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

        for (unsigned int resultIndex = 0; resultIndex < model.Dimension; resultIndex++) {
            results[resultIndex] += leafValuesPtr[index][resultIndex];
        }

        treeSplitsPtr += currentTreeDepth;
        leafValuesPtr += 1 << currentTreeDepth;
        treePtr += currentTreeDepth;
    }

    std::vector<double> finalResults(model.Dimension);
    for (unsigned int resultId = 0; resultId < model.Dimension; resultId++) {
        finalResults[resultId] = model.Scale * results[resultId] + model.Biases[resultId];
    }
    return finalResults;
}


double ApplyCatboostModel(
    const std::vector<float>& floatFeatures,
    const std::vector<std::string>& catFeatures
) {
    return ApplyCatboostModelMulti(floatFeatures, catFeatures)[0];
}
