#include <vector>

/* Model data */
static const struct CatboostModel {
    unsigned int TreeCount = 2;
    unsigned int FloatFeatureCount = 8;
    unsigned int BinaryFeatureCount = 8;
    unsigned int BorderCounts[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    float Borders[8] = {0.5,0.5,0.5,0.0511068,0.5,0.5,0.5,0.259879};
    unsigned int TreeDepth[2] = {6, 4};
    unsigned int TreeSplits[10] = {6, 4, 3, 5, 2, 7, 6, 0, 1, 4};

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[80] = {
        0.0008211111193150276, 0, 0.0002249999959287899, 0, 0.001252702682193469, 0, 0.0003937499928753823, 0, 0.000989974279526389, 0.001679999969601631, 0.000647499988283962, 0, 0.0007328258582250274, 0, 0.001145454524728385, 0, 0.0008999999837151595, 0, 0, 0, 0, 0, 0, 0, 0.001374999961815775, 0, 0, 0, 0.001399999974668026, 0, 0.0005249999905005097, 0, 0.001039998001456305, 0, 0.002099999962002039, 0.001259999977201223, 0.001890090150245136, 0.002099999962002039, 0.002390476385184691, 0.00347205903071033, 0.001214583574654528, 0.0004199999924004077, 0.001259999977201223, 0.003790092530154747, 0.001781553489653492, 0.002363889468771702, 0.002231249959627166, 0.003923684268934943, 0, 0, 0, 0, 0.002604545391452584, 0, 0, 0, 0, 0, 0, 0, 0.00275191166696851, 0.002099999962002039, 0, 0,
        0.0007538577474999658, 0, 0.001673253488408079, 0.002738652127683553, 0.0002184838828884162, 0, 0.0008825505085869142, 0, 0.0003120346099853925, 0, 0.00158349975932882, 0.00373994872438916, 0, 0, 0, 0
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
