#include <string>
#include <vector>

/* Model data */
static const struct CatboostModel {
    unsigned int FloatFeatureCount = 50;
    unsigned int BinaryFeatureCount = 12;
    unsigned int TreeCount = 2;
    unsigned int TreeDepth[2] = {6, 6};
    unsigned int TreeSplits[12] = {2, 1, 6, 5, 8, 11, 7, 3, 10, 4, 9, 0};
    unsigned int BorderCounts[50] = {1, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0};
    float Borders[12] = {0.323544502f, 0.0031594648f, 0.521369994f, 0.5f, 0.550980508f, 0.979140043f, 0.0950040966f, 0.5f, 0.239182502f, 0.287964523f, 0.238694996f, 0.447151482f, };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[128] = {
        0.001259011842534326, 0, 0.001474009916287763, 0.00064166661547497, 0, 0, 0.002863636311820962, 0.0009333333164453506, 0.0002386363593184135, 0, 0.002116666806116696, 0, 0, 0, 0, 0.001679999969601631, 0.0005249999905005097, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002088370528948082, 0, 0.001851857674803305, 0.001553571389056742, 0, 0, 0.002822325645650539, 0.004138979825677001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001799999967430319, 0, 0, 0, 0, 0, 0, 0, 0,
        0.0003675624400133544, 0, 0.001059092784733128, 0, 0, 0, 0.002066946665087761, 0.001192459419076518, 0.001103175683534166, 0, 0.0004773354283567961, 0, 0, 0, 0.001827927523090045, 0, 0, 0, 0.001934663401212227, 0, 0, 0, 0.002079562911984692, 0.002016393416732761, 0, 0, 0.00153485204725672, 0, 0, 0, 0.001034478467845754, 0, 0.0006763936415426848, 0, 0.0005583088061050648, 0, 0, 0, 0.00178538384126724, 0.003265464248219695, 0.0004448876734695067, 0, 0.0005178430285843198, 0, 0, 0, 0.00121787682371597, 0.003195419440159142, 0, 0, 0.002190701743713123, 0, 0, 0, 0.00230819225408413, 0.003807514519097499, 0, 0, 0.00167323412960814, 0, 0, 0, 0.001315321241806467, 0.003950911460742678
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
