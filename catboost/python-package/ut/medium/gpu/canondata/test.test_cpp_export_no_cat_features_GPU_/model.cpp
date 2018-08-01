#include <vector>

/* Model data */
static const struct CatboostModel {
    unsigned int FloatFeatureCount = 9;
    unsigned int BinaryFeatureCount = 12;
    unsigned int TreeCount = 2;
    unsigned int TreeDepth[2] = {6, 6};
    unsigned int TreeSplits[12] = {11, 2, 4, 7, 1, 8, 6, 3, 10, 9, 0, 5};
    unsigned int BorderCounts[9] = {1, 2, 1, 1, 3, 1, 1, 1, 1};
    float Borders[12] = {0.011715351f, 0.34408599f, 0.76047051f, 0.63000298f, 0.43333352f, 0.41666651f, 0.58333349f, 1.5f, 0.94102502f, 0.5f, 0.50285947f, 0.3318105f};

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[128] = {
        0.0001500000071246177, 0.00201694923453033, 0, 0, 0.0005999999702908099, 0.002136326860636473, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001308776205405593, 0.002161809476092458, 0.001329330960288644, 0.00378138036467135, 0.0004965625121258199, 0.00143311102874577, 0.0009444663301110268, 0.002024568850174546, 0, 0.00139999995008111, 0, 0.001049999962560833, 0, 0, 0, 0.001049999962560833, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0.0004868714313488454, 0, 0, 0, 0, 0, 0, 0, 0.0007201300468295813, 0, 0, 0, 0.003045676508918405, 0, 0, 0, 0.001428714254871011, 0, 0.00415117060765624, 0, 0, 0, 0, 0, 0.001657033222727478, 0, 0.003496180754154921, 0, 0.003044427838176489, 0, -1.621357114345301e-05, 0, -7.083497166604502e-06, 0.0008241712930612266, 0, 0, 0, 0, 0, 0, -1.791286740626674e-05, 0, 0, 0, 0.0005087864119559526, 0.001163561129942536, 0, 0, 0.0006044265464879572, 0.001137889921665192, 0.0005150138749741018, -3.483569525997154e-05, 0, 0, 0, 0, 0.001425641938112676, 0.0004013623401988298, 0, 0, 0.00198999117128551, 0.003376733278855681, 0.0006148157408460975, 0
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
