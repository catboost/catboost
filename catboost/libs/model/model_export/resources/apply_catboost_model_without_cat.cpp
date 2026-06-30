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
