#include <string>
#include <vector>

/* Model data */
static const struct CatboostModel {
    unsigned int FloatFeatureCount = 50;
    unsigned int BinaryFeatureCount = 12;
    unsigned int TreeCount = 2;
    unsigned int TreeDepth[2] = {6, 6};
    unsigned int TreeSplits[12] = {5, 1, 10, 3, 8, 7, 2, 9, 11, 0, 4, 6};
    unsigned int BorderCounts[50] = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
    float Borders[12] = {0.834502459f, 0.156479999f, 0.937812984f, 0.748417497f, 0.594812512f, 0.0867389515f, 0.185759991f, 0.423586994f, 0.00513814017f, 0.00177894998f, 0.005463365f, 0.464960515f, };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[128] = {
        -0.01499078282534252, 0.00332429613918066, -0.005713707430565611, -0.01512550283223391, -0.008831512074296673, 0.002103412213424841, 0.0001156789504668929, 0.03964748047292233, -0.01583432078382558, 0.008207830600440502, 0.0298728235630375, 0, 0.02366911508142948, 0.02338480926118791, 0.02104378228143948, 0.01871371109570776, -0.007375754183158278, 0.002013050951063633, -0.007562751416116953, 0.0023744972422719, 0.008899597823619843, 0.0366896146110126, 0.03626341403772434, 0.02045945468403044, -0.01853587350773591, 0.01367971766740084, 0.0140411639586091, 0.009937248658388853, 0.01987449731677771, 0.02625836556156476, 0.01609312160871923, 0.03387760596039394, -0.01026646057143807, -0.001406878465786576, -0.01346335476264358, 0, -0.006807229557150119, -0.01228628892983709, 0.007414435004730793, 0.01585341275980075, -0.004237469251549572, 0.007713711155312402, 0.0422179326415062, 0, 0.006258371010146759, 0.03761044061846203, 0.003625589655712247, 0.008207830600440502, -0.01015687850303948, 0.07274493551813066, 0, 0, 0, 0.02855278165744884, 0, 0.05904116512586673, -0.01539658755064011, -0.007562751416116953, 0, 0, 0, 0.03179919570684433, 0.00118724862113595, -0.005100402235984802,
        -0.009776548494163753, 0, -0.0004823473147140704, 0, 0.0006686191379420487, 0, 0.00197923692820046, 0.001307706499383562, -0.007401229853826732, 0, 0.02096871966459878, 0.005962349195033311, 0, 0, 0.007140203175090606, 0, 0.008357482100824973, 0, 0.0744871213682976, 0, 0, 0, 0.01239996451039868, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004691376493660588, 0.0003303209319710732, 0.009402653884111427, 0.1217359174825972, -0.003059979277895764, 0, 0.004428125522045438, 0.009124076661343378, 0.05959413343225606, 0.05539775782381184, -0.00358151692808384, 0.005702547913339611, 0, 0, 0.01204003252658165, 0.01557480075163767, 0, 0, 0, 0, 0, 0, -0.009735384877496088, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    double Scale = 1;
    double Bias = 0.06050201133;
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
    return model.Scale * result + model.Bias;
}

double ApplyCatboostModel(
    const std::vector<float>& floatFeatures,
    const std::vector<std::string>&
) {
    return ApplyCatboostModel(floatFeatures);
}
