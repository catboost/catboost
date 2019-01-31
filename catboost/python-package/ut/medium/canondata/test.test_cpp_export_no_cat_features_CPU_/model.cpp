#include <string>
#include <vector>

/* Model data */
static const struct CatboostModel {
    unsigned int FloatFeatureCount = 50;
    unsigned int BinaryFeatureCount = 12;
    unsigned int TreeCount = 2;
    unsigned int TreeDepth[2] = {6, 6};
    unsigned int TreeSplits[12] = {2, 6, 0, 9, 10, 8, 7, 5, 3, 11, 1, 4};
    unsigned int BorderCounts[50] = {0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1};
    float Borders[12] = {0.00132575491f, 0.130032003f, 0.521369994f, 0.5f, 0.5f, 0.5f, 0.134183004f, 0.5f, 0.5f, 2.37056011e-05f, 0.250777483f, 0.5f, };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[128] = {
        0.0005996913577188494, 0, 0, 0, 0, 0, 0, 0, 0.000221052627579162, 0, 0, 0, 0.0009768016478353648, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001960402433831547, 0, 0, 0, 0.0004199999924004077, 0, 0, 0, 0, 0, 0, 0, 0.0006999999873340129, 0, 0, 0, 0.001368055558297782, 0.0001866665999591365, 0, 0.001679999969601631, 0.001604166730772702, 0.001049999981001019, 0.001499999972858599, 0.003954763324079735, 0, 0, 0, 0, 0, 0, 0.001049999981001019, 0, 0.001779299344200712, 0.001465909057127481, 0.002944927643757796, 0.003975728267858036,
        0.0008915419558863947, 0, 0.0003512460733605982, 0, 0.0004428744834216426, 0.0005497278883047848, 0.0007437548665033633, 0, 0, 0, 0, 0, 0.001210233305897622, 0.003847465523299554, 0.001161668204743658, 0.002286348013900796, 0, 0, 0, 0, 0.001845609629668785, 0.00280269449486017, 0.001692741537757669, 0, 0, 0, 0, 0, 0.002007853056220058, 0.003279802623184852, 0.002235919726910874, 0.003849602074058009, -1.028042304539491e-05, 0, -4.497685082360273e-06, 0, 0.001735347943945, 0, -1.657894669786881e-06, 0, 0, 0, 0, 0, 0.002337718806557816, 0.001027913024166518, 0.0008234633792182209, 0, 0, 0, 0, 0, 0, 0, 0.0008223767607047237, 0, 0, 0, 0, 0, 0.002439206489894105, 0.001638476035060068, 0.003277561698191198, 0
    };
} CatboostModelStatic;

/* Model applicator */
double ApplyCatboostModel(
    const std::vector<float>& features
) {
    const struct CatboostModel& model = CatboostModelStatic;

    /* Binarise features */
    std::vector<unsigned char> binaryFeatures(model.BinaryFeatureCount);
    unsigned int binFeatureIndex = 0;
    for (unsigned int i = 0; i < model.FloatFeatureCount; ++i) {
        for(unsigned int j = 0; j < model.BorderCounts[i]; ++j) {
            binaryFeatures[binFeatureIndex] = (unsigned char)(features[i] > model.Borders[binFeatureIndex]);
            ++binFeatureIndex;
        }
    }

    /* Extract and sum values from trees */
    double result = 0.0;
    const unsigned int* treeSplitsPtr = model.TreeSplits;
    const double* leafValuesForCurrentTreePtr = model.LeafValues;
    for (unsigned int treeId = 0; treeId < model.TreeCount; ++treeId) {
        const unsigned int currentTreeDepth = model.TreeDepth[treeId];
        unsigned int index = 0;
        for (unsigned int depth = 0; depth < currentTreeDepth; ++depth) {
            index |= (binaryFeatures[treeSplitsPtr[depth]] << depth);
        }
        result += leafValuesForCurrentTreePtr[index];
        treeSplitsPtr += currentTreeDepth;
        leafValuesForCurrentTreePtr += (1 << currentTreeDepth);
    }
    return result;
}

double ApplyCatboostModel(
    const std::vector<float>& floatFeatures,
    const std::vector<std::string>&
) {
    return ApplyCatboostModel(floatFeatures);
}
