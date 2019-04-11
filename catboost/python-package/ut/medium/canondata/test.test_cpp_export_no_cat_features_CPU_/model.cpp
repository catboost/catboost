#include <string>
#include <vector>

/* Model data */
static const struct CatboostModel {
    unsigned int FloatFeatureCount = 50;
    unsigned int BinaryFeatureCount = 12;
    unsigned int TreeCount = 2;
    unsigned int TreeDepth[2] = {6, 6};
    unsigned int TreeSplits[12] = {2, 1, 6, 5, 8, 11, 7, 3, 10, 4, 9, 0};
    unsigned int BorderCounts[50] = {1, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0};
    float Borders[12] = {0.322089016f, 0.0031594648f, 0.521369994f, 0.5f, 0.550980508f, 0.996324539f, 0.656862974f, 0.5f, 0.0069912402f, 0.285950482f, 0.236585006f, 0.445143998f, };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[128] = {
        0.001294099685395057, 0, 0.001663731629059374, 0.0005249999905005097, 0.0008576818214875049, 0, 0.0005905404957665791, 0.001049999981001019, 0, 0, 0, 0, 0, 0, 0, 0, 0.0005414062402036506, 0, 0.002567500461246807, 0.0007333332748285372, 0.0005249999905005097, 0, 0, 0.001679999969601631, 0, 0, 0, 0, 0, 0, 0, 0, 0.002037932443256619, 0, 0.002110218347738532, 0.003083823474551387, 0.002161956464572121, 0, 0.001152380927474726, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0005249999905005097, 0, 0.003047058768395115, 0.004090148077996925, 0, 0, 0.0005249999905005097, 0.002062499998603015, 0, 0, 0, 0, 0, 0, 0, 0,
        0.0003231377723979838, 0, 0.00105791920933408, 0, 0, 0, 0.002063173418219615, 0.001217081715965126, 0.001102714260714291, 0, 0.0004444306049201454, 0, 0, 0, 0.001594016627768246, 0, 0, 0, 0.001935656303693709, 0, 0, 0, 0.002064051449280681, 0.002037610333493785, 0, 0, 0.001519155501243254, 0, 0, 0, 0.001067039113331727, 0, 0.0008372005929878915, 0, 0.0005312171758752161, 0, 0, 0, 0.001773409163652216, 0.003262653168409861, 0.0004447483693683492, 0, 0.0005997946154227071, 0, 0, 0, 0.001232067526134975, 0.003190115195858002, 0, 0, 0.002193397001291078, 0, 0, 0, 0.002352321022124672, 0.003810887531325806, 0, 0, 0.001677124391007667, 0, 0, 0, 0.001325081612224604, 0.003956065626275658
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
