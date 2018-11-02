#include <string>
#include <vector>

/* Model data */
static const struct CatboostModel {
    unsigned int FloatFeatureCount = 50;
    unsigned int BinaryFeatureCount = 12;
    unsigned int TreeCount = 2;
    unsigned int TreeDepth[2] = {6, 6};
    unsigned int TreeSplits[12] = {8, 0, 5, 9, 11, 6, 10, 2, 7, 4, 3, 1};
    unsigned int BorderCounts[50] = {0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0};
    float Borders[12] = {0.18692601f, 0.5f, 0.5f, 0.5f, 0.5f, 0.95771205f, 0.5f, 0.097222149f, 0.5f, 0.00032979899f, 0.75300753f, 0.26409501f, };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[128] = {
        0.0006955065930426542, 0, 0, 0, 0.0001810344794829344, 0, 0, 0, 0.001098490266995831, 0, 0.001589791657724417, 0, 0.001828124975902028, 0.001679999969601631, 0, 0, 0.001329545421779833, 0, 0, 0, 0.0005833335090428549, 0, 0, 0, 0.001529906971799882, 0.00269666736505924, 0.002342030875375432, 0.001938461503386497, 0.001049999981001019, 0.001049999981001019, 0.0008765624896273946, 0, 0.0002999999945717198, 0, 0, 0, 0, 0, 0, 0, 0.0003187499942324523, 0, 0.001949999964716179, 0, 0.0005249999905005097, 0, 0, 0, 0.001469999973401427, 0.002677381649373881, 0, 0, 0, 0, 0, 0, 0.001873529377864564, 0.003867032899167183, 0.003096296610434842, 0.003725676117001744, 0, 0.001679999969601631, 0, 0,
        0.0006982795337251036, 0, 0, 0, 0.0004140269530082293, 0, 0, 0, 0.000228045619107548, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0009179783236801756, 0.004755924048153645, 0.001566909289921064, 0.002035708918338523, 0.0007676849182978572, 0.002487475692542658, 0.0005208372840066653, 0.0024006384178372, 0.0005134157523437949, 0, 0.0009316294913245664, 0.0005135256884684811, 0, 0, 0, 0, 0, 0, 0.00191797439944338, 0.003727659281110782, 0, 0, 0.002112067698191821, 0.003495110258087365, 0, 0, 0.000791394002761522, 0.001031641075398032, 0, 0, 0.0006124347462520387, 0
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
