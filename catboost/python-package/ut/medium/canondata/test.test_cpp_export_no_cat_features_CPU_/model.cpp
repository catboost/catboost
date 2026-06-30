#include <string>
#include <vector>

/* Model data */
static const struct CatboostModel {
    CatboostModel() = default;
    unsigned int FloatFeatureCount = 50;
    unsigned int CatFeatureCount = 0;
    unsigned int BinaryFeatureCount = 12;
    unsigned int TreeCount = 2;
    std::vector<std::vector<float>> FloatFeatureBorders = {
        {},
        {0.156479999},
        {0.937812984},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {0.398038983},
        {0.748417497},
        {0.0504557006, 0.564610481},
        {},
        {},
        {},
        {0.398038983},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {0.134183004},
        {},
        {0.185759991},
        {},
        {0.318073004, 0.423586994},
        {},
        {},
        {},
        {0.005463365},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}
    };
    unsigned int TreeDepth[2] = {6, 6};
    unsigned int TreeSplits[12] = {7, 0, 11, 3, 5, 10, 1, 2, 6, 9, 4, 8};
    unsigned int BorderCounts[50] = {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float Borders[12] = {0.156479999f, 0.937812984f, 0.398038983f, 0.748417497f, 0.0504557006f, 0.564610481f, 0.398038983f, 0.134183004f, 0.185759991f, 0.318073004f, 0.423586994f, 0.005463365f, };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[128][1] = {
        {-0.01476383881461352}, {-0.001527162562859686}, {-0.006224901143771906}, {-0.008500670393308004}, {-0.005548771821939555}, {0.03321244932711125}, {-0.0007866994090765327}, {0.02974564745090902}, {-0.01629468016139227}, {0.01505178487614581}, {0.01007350760379008}, {0.009937248658388853}, {0.01932053081691265}, {0.02920025381548651}, {0.02495628258062376}, {0.03457912147045136}, {0.02889959737658501}, {0}, {0}, {0}, {-0.02016733710964521}, {0}, {0.02151997769251466}, {0}, {-0.007286288908549717}, {0}, {0.07163563167506998}, {0}, {0.03654116361091534}, {0}, {0.008546827094895499}, {0.00702062388882041}, {-0.010533983425406}, {0.06907871986428897}, {-0.01257979418886335}, {0}, {-0.006533451605497337}, {0.02355278163616146}, {0.00832707500306978}, {0.0532265424051068}, {-0.005099951297420414}, {0.0023744972422719}, {0.006521995941346342}, {0}, {0.01014725991559249}, {0.03873599536324802}, {0.002457431196395693}, {0.0023744972422719}, {0}, {0}, {-0.007562751416116953}, {0}, {0}, {0}, {-0.005775703676044941}, {-0.007562751416116953}, {-0.007562751416116953}, {0}, {0.07546812086366117}, {0}, {-0.005100402235984802}, {0}, {0.008899597823619843}, {0},
        {-0.005185098429306783}, {0}, {-0.001709653076446105}, {0}, {-0.006138443525332251}, {0}, {-0.01215968660400086}, {0}, {-0.01519701123714509}, {0}, {-0.003332020375386257}, {0}, {-0.007397120564829815}, {0}, {-0.005090657595602796}, {0}, {0.001331868529870192}, {0.001844042242737487}, {0.0006912374866847305}, {-0.002530957310227677}, {0.0003234795604749375}, {0}, {-0.003460162245464885}, {0}, {0.006208794650302458}, {0}, {0.01026425088720871}, {0}, {-0.009062705858377137}, {0}, {-0.003170142447814982}, {0.00509524923798285}, {-0.01921837785302248}, {0}, {0.01186028514135008}, {-0.007371856095759492}, {0}, {0}, {-0.008735171039180272}, {0.008055775548870627}, {0.02558814778354052}, {0}, {0.01813888668502663}, {0}, {0.00976561973448551}, {0.05585603478054205}, {0.01800648072426705}, {0}, {0.005122852583468298}, {0}, {0.01407679499998793}, {0.01682312171287297}, {0}, {0}, {0.0002649408239403998}, {0.06453569420613349}, {0.005155574898665143}, {0.1034671237030998}, {0.01886628208918604}, {0.006703344076586683}, {-0.008603635791500676}, {0}, {-0.003121383204795378}, {0.04060238017912277}
    };
    double Scale = 1;
    double Biases[1] = {0.06050201133};
    unsigned int Dimension = 1;
} CatboostModelStatic;

/* Model applicator */
std::vector<double> ApplyCatboostModelMulti(
    const std::vector<float>& floatFeatures
) {
    const struct CatboostModel& model = CatboostModelStatic;

    /* Binarize features */
    std::vector<unsigned char> binaryFeatures(model.BinaryFeatureCount);
    unsigned int binFeatureIndex = 0;
    for (unsigned int i = 0; i < model.FloatFeatureCount; ++i) {
        for(unsigned int j = 0; j < model.BorderCounts[i]; ++j) {
            binaryFeatures[binFeatureIndex] = (unsigned char)(floatFeatures[i] > model.Borders[binFeatureIndex]);
            ++binFeatureIndex;
        }
    }

    /* Extract and sum values from trees */
    std::vector<double> results(model.Dimension, 0.0);
    const unsigned int* treeSplitsPtr = model.TreeSplits;
    const auto* leafValuesForCurrentTreePtr = model.LeafValues;
    for (unsigned int treeId = 0; treeId < model.TreeCount; ++treeId) {
        const unsigned int currentTreeDepth = model.TreeDepth[treeId];
        unsigned int index = 0;
        for (unsigned int depth = 0; depth < currentTreeDepth; ++depth) {
            index |= (binaryFeatures[treeSplitsPtr[depth]] << depth);
        }

        for (unsigned int resultIndex = 0; resultIndex < model.Dimension; resultIndex++) {
            results[resultIndex] += leafValuesForCurrentTreePtr[index][resultIndex];
        }

        treeSplitsPtr += currentTreeDepth;
        leafValuesForCurrentTreePtr += 1 << currentTreeDepth;
    }

    std::vector<double> finalResults(model.Dimension);
    for (unsigned int resultId = 0; resultId < model.Dimension; resultId++) {
        finalResults[resultId] = model.Scale * results[resultId] + model.Biases[resultId];
    }
    return finalResults;
}

double ApplyCatboostModel(
    const std::vector<float>& floatFeatures
) {
    return ApplyCatboostModelMulti(floatFeatures)[0];
}

// Also emit the API with catFeatures, for uniformity
std::vector<double> ApplyCatboostModelMulti(
    const std::vector<float>& floatFeatures,
    const std::vector<std::string>&
) {
    return ApplyCatboostModelMulti(floatFeatures);
}

double ApplyCatboostModel(
    const std::vector<float>& floatFeatures,
    const std::vector<std::string>&
) {
    return ApplyCatboostModelMulti(floatFeatures)[0];
}
