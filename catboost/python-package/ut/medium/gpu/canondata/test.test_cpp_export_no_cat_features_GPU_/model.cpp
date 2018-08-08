#include <vector>

/* Model data */
static const struct CatboostModel {
    unsigned int FloatFeatureCount = 9;
    unsigned int BinaryFeatureCount = 11;
    unsigned int TreeCount = 2;
    unsigned int TreeDepth[2] = {6, 5};
    unsigned int TreeSplits[11] = {10, 5, 0, 4, 1, 9, 3, 6, 8, 7, 2};
    unsigned int BorderCounts[9] = {1, 1, 1, 2, 1, 1, 1, 1, 2};
    float Borders[11] = {0.8374055f, 0.60392153f, 0.387779f, 0.58333349f, 1.5f, 0.93881702f, 0.061012201f, 0.5f, 0.97901797f, 0.27336848f, 0.66261351f};

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[96] = {
        0.0005814876058138907, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001133998390287161, 0, 0.0009975000284612179, 0, 0, 0, 0.0008555554668419063, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001372431288473308, 0.002215142827481031, 0.001875349087640643, 0.002333333250135183, 0, 0, 0.001049999962560833, 0.006780804600566626, 0, 0, 0, 0, 0, 0, 0, 0, 0.002136547351256013, 0.002390978159382939, 0.002491590334102511, 0.003389361780136824, 0, 0, 0.002099999925121665, 0.003947368357330561, 0, 0, 0, 0.002099999925121665, 0, 0, 0, 0,
        0.001135568716563284, 0.001330881263129413, 0, 0, 0.0002743172226473689, 0, 0, 0, 0.001436591031961143, 0.001021189265884459, 0.003254958894103765, 0.003336918773129582, 0.000900790560990572, 0, 0.001032500062137842, 0, 0.002713314490392804, -2.869173840736039e-05, 0, 0, 0, 0, 0, 0, 0.00300825503654778, 0, 0.002350773196667433, 0.000385663821361959, 0, 0, 0, 0
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
