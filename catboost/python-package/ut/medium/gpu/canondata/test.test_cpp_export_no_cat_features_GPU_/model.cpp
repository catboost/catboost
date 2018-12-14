#include <string>
#include <vector>

/* Model data */
static const struct CatboostModel {
    unsigned int FloatFeatureCount = 50;
    unsigned int BinaryFeatureCount = 10;
    unsigned int TreeCount = 2;
    unsigned int TreeDepth[2] = {6, 5};
    unsigned int TreeSplits[11] = {9, 0, 1, 3, 5, 8, 3, 6, 2, 4, 7};
    unsigned int BorderCounts[50] = {1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0};
    float Borders[10] = {0.0012634799f, 0.44707751f, 0.49607849f, 1.5f, 0.5f, 0.00025269552f, 0.0017220699f, 0.67529798f, 0.2598795f, 0.66261351f, };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[96] = {
        0.0002299999759998173, 0, 0.001010674517601728, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001014084555208683, 0, 0, 0, 0.001166666625067592, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001468627364374697, 0.001049999962560833, 0.001727674854919314, 0.002749624894931912, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002085107145830989, 0.00265559321269393, 0, 0, 0.002889422932639718, 0.004042857326567173, 0, 0, 0, 0.001799999969080091, 0, 0, 0, 0.001049999962560833,
        0.0008321835193783045, 0, 0.001828600885346532, 0, 0.0005662433104589581, 0, 0.001081391936168075, 0, 0.001461817068047822, 0, 0.003061093855649233, 0.001769142807461321, 0.0009195390157401562, 0, 0.00115906388964504, 0.001042125048115849, 0, 0, 0.002436290495097637, 0, -1.295756101171719e-05, 0, 0.001525443862192333, 0, -1.295756101171719e-05, 0, 0.001712905708700418, 0, 0, 0, 0.002524094888940454, 0
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
