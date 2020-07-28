#include <string>
#include <vector>

/* Model data */
static const struct CatboostModel {
    unsigned int FloatFeatureCount = 50;
    unsigned int BinaryFeatureCount = 12;
    unsigned int TreeCount = 2;
    unsigned int TreeDepth[2] = {6, 6};
    unsigned int TreeSplits[12] = {11, 10, 4, 5, 3, 0, 7, 6, 1, 9, 8, 2};
    unsigned int BorderCounts[50] = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    float Borders[12] = {0.5f, 0.215384498f, 0.387778997f, 0.795647502f, 0.405882478f, 0.0416666493f, 0.416666508f, 0.0950040966f, 0.5f, 0.682412982f, 0.00160859502f, 0.662613511f, };

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[128] = {
        -0.0159872081130743, 0, 0.001827788073569536, -0.00241227587684989, -0.01710828207433224, 0, -0.003780113765969872, 0.02732429653406143, -0.01001864671707153, 0.00189959816634655, -0.0001757035061018541, -0.005100402049720287, -0.01767570339143276, 0, 0.005213711876422167, 0.04889960214495659, 0, 0, 0.09147840738296509, 0.02057085372507572, 0, 0, 0.02487449534237385, 0.0115411626175046, 0, 0, 0, -0.007562751416116953, 0, 0, 0, 0, 0.01425643637776375, -0.007286288775503635, 0.01101872883737087, 0.02800958603620529, -0.008713617920875549, 0.1065425053238869, -0.01192301977425814, 0.02110324613749981, -0.007286288775503635, 0.002374497475102544, 0.01222138572484255, 0.02749908715486526, 0.00189959816634655, 0.00993724912405014, 0.03425620496273041, 0.02792106010019779, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.004646088927984238, 0, 0, 0, -0.0008338363841176033,
        -0.01520246081054211, 0, 0, 0, -0.006262023467570543, 0, -0.003924420569092035, 0, 0, 0, 0, 0, -0.01667694933712482, 0, -0.003839465323835611, 0, -0.01459948904812336, 0.05867299810051918, 0, -0.01128849759697914, -0.0004827358352486044, 0.01916735246777534, 0.0005391894374042749, 0.01980392262339592, 0, 0.06027792394161224, 0, 0, -0.01042931713163853, 0.02095017768442631, 0.001649297773838043, 0.01597516052424908, 0, 0, 0, 0, 0.01961536332964897, 0, -0.01132655702531338, 0, 0, 0, 0, 0, -0.02513201162219048, 0, -0.01091300323605537, 0, 0, 0, 0, 0, 0.02840910479426384, 0.002173124812543392, 0, -0.00408032163977623, 0, 0, 0, 0, -0.0216811764985323, -0.01041869167238474, 0, 0.003041477873921394
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
