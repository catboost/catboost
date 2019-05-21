#include <string>
#include <vector>

/* Model data */
static const struct CatboostModel {
    unsigned int FloatFeatureCount = 50;
    unsigned int BinaryFeatureCount = 12;
    unsigned int TreeCount = 2;
    unsigned int TreeDepth[2] = {6, 6};
    unsigned int TreeSplits[12] = {7, 5, 0, 4, 9, 3, 11, 6, 1, 10, 2, 8};
    unsigned int BorderCounts[50] = {1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0};
    float Borders[12] = {0.168183506f, 0.136649996f, 0.5f, 0.645990014f, 0.272549003f, 0.5f, 0.5f, 0.00311657996f, 0.392024517f, 0.163893014f, 0.387494028f, 0.767975509f, };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[128] = {
        0, 0, 0.001049999981001019, 0, 0.0006999999873340129, 0.0004199999924004077, 0, 0.0003499999936670065, 0.0006999999873340129, 0, 0, 0, 0.0008999999837151595, 0.0008399999848008155, 0, 0.005499108720590722, 0, 0, 0.001310204828003209, 0, 0, 0, 0.002099799992893635, 0.003281666943095616, 0, 0, 0.001081739094670376, 0, 0, 0, 0.001261285375551461, 0.003276219466932844, 0.0001235294095295317, 0, 0.001143749976810068, 0, 0, 0, 0.002145652130229966, 0.001699999969239746, 0.0005409089896827957, 0, 0.0004052632035001316, 0, 0.001076041724695822, 0, 0.001088611397153381, 0.001412068939966888, 0, 0, 0.001978133278083802, 0.001049999981001019, 0, 0, 0.002257627220401316, 0.002847457878670443, 0, 0, 0.002380742281139534, 0, 0, 0, 0.00181075114546265, 0.003175870739941051,
        0.0008166451595296908, 0, 0.001435124236388357, 0, 0.002230458559199401, 0, 0.002666874626702434, 0, 0.001534999525103938, 0.001002557986130936, 0.001371602072510968, 0.001025428335548242, 0.002692151343928724, 0.002324272621936243, 0.001188379847259753, 0.002102531171547649, 0.0002273645927514973, 0, 0.002259198170953273, 0, 0.0005932082092034878, 0, 0.001936697595469175, 0, 0.002274218690520189, 0.00329724810179383, 0.003553552019915716, -2.457164545277724e-05, 0.001361408072189387, 0.002058595183732444, 0.002040357509679229, 0.006411161288685438, 0.001238019183794491, 0, 0.0007759661277461006, 0, 0.002378015135461136, 0, 0.0005649959300306291, 0, 0.001080913918043391, 0.005514189264833609, 0.001790046575156435, 0.001103960055365507, 0.002146005492857177, 0.002352553092417847, 0.001165392567571486, 0, 0.0003389142165572722, 0, 0.000740077287450055, 0, 0.0006711111753359577, 0, 0.001173741518772528, 0, 0.0008677387848554152, 0.003458858412712405, 0.001394273280892068, 0.002416533521423629, 0.003145683167659726, 0.002795911932059462, 0.001047576512888525, 0.003163031305315038
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
