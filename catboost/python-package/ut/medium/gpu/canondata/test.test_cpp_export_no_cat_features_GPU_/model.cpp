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
        {},
        {0.937671542},
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
        {},
        {0.215384498},
        {0.630002975, 0.795647502},
        {},
        {},
        {},
        {0.441176474, 0.492156982},
        {},
        {},
        {0.0416666493},
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
        {},
        {0.5},
        {0.026869949},
        {},
        {0.00181464502},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {0.753007531},
        {0.211401001},
        {}
    };
    unsigned int TreeDepth[2] = {6, 6};
    unsigned int TreeSplits[12] = {10, 9, 5, 6, 3, 11, 1, 0, 8, 4, 7, 2};
    unsigned int BorderCounts[50] = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0};
    float Borders[12] = {0.937671542f, 0.215384498f, 0.630002975f, 0.795647502f, 0.441176474f, 0.492156982f, 0.0416666493f, 0.5f, 0.026869949f, 0.00181464502f, 0.753007531f, 0.211401001f, };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[128][1] = {
        {-0.0145869292318821}, {0}, {-0.002365272026509047}, {0}, {-0.02357292920351028}, {0.00993724912405014}, {-0.01199380867183208}, {0}, {-0.009273458272218704}, {0}, {0.04368724673986435}, {0}, {0.00189959816634655}, {0}, {-0.007562751416116953}, {0}, {0}, {0}, {-0.007562751416116953}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.008393868803977966}, {0.04390034452080727}, {0.005161481443792582}, {0.03577119112014771}, {-0.009541401639580727}, {-0.007562751416116953}, {-0.01070352923125029}, {0.02063612826168537}, {-0.01358906365931034}, {0.006478413008153439}, {0.01261449605226517}, {0.03521017357707024}, {-0.01512550283223391}, {0.00993724912405014}, {0.003547068685293198}, {0.02271371148526669}, {0}, {0}, {0.08332269638776779}, {0}, {0}, {0}, {0.01057085301727057}, {0}, {0}, {0}, {-0.009767072275280952}, {0}, {0}, {0}, {-0.0008338363841176033}, {0},
        {-0.01313269883394241}, {-0.005075628403574228}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.006224810145795345}, {-0.008623102679848671}, {0}, {0}, {0}, {0}, {0}, {0}, {-0.01049762405455112}, {0.005245530512183905}, {0.06421585381031036}, {0.07963623106479645}, {0}, {0.01120027247816324}, {0}, {0.007361581083387136}, {0.02234085090458393}, {-0.003822155995294452}, {0}, {0.008695092052221298}, {0}, {0.01350340619683266}, {0.05900333076715469}, {0.06428597867488861}, {0}, {0.03240378201007843}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.00427056523039937}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.02046489156782627}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0.009019204415380955}, {0}, {0}, {0}, {0}, {0}, {0}
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
