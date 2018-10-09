#include <vector>

/* Model data */
static const struct CatboostModel {
    unsigned int FloatFeatureCount = 50;
    unsigned int BinaryFeatureCount = 11;
    unsigned int TreeCount = 2;
    unsigned int TreeDepth[2] = {6, 5};
    unsigned int TreeSplits[11] = {8, 0, 5, 9, 4, 6, 10, 7, 3, 2, 1};
    unsigned int BorderCounts[50] = {0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
    float Borders[11] = {0.0622565f, 0.5f, 0.5f, 0.5f, 0.5f, 0.98272598f, 0.5f, 0.097222149f, 0.5f, 0.0010412449f, 0.60571146f, };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[96] = {
        0.0007909629375735687, 0.0005249999905005097, 0.00191470584442072, 0, 0.0009313725844463869, 0.001049999981001019, 0, 0, 0.001480637814883255, 0.002774999973736703, 0.001819217596641749, 0.002135000514574336, 0, 0, 0, 0, 0.001424999964237213, 0, 0.001049999981001019, 0, 0, 0, 0, 0, 0.001224999977834523, 0, 0.00305039463412899, 0.002099999962002039, 0, 0, 0, 0, 0.0005876865565304213, 0.00295422133567028, 0, 0, 0, 0, 0, 0, 0.001826086923480034, 0.003670588189538786, 0.002372058927585532, 0.003879166769288062, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0005249999905005097, 0, 0, 0, 0, 0,
        0.0006921719453553643, 0, 0.0004117280956206751, 0, 0.0002214690529509848, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001420420624297372, 0.003184800622098175, 0.0004887949433514606, 0.002349379335729056, 0.0008775089953256384, 0.0005138952071370965, 0, 0, 0.001868792742743027, 0.003363426897674278, 0.001574119166780254, 0.003566977137765126, 0.0006661870385846651, 0.001839427350751397, 0, 0.0006163558457548979
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
