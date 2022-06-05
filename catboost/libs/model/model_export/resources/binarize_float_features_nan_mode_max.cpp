static void BinarizeFloatFeatures(
    const std::vector<float>& floatFeatures,
    const struct CatboostModel& model,
    std::vector<unsigned char>& binaryFeatures,
    unsigned int& binFeatureIndex) {
    for (size_t i = 0; i < model.FloatFeatureBorders.size(); ++i) {
        if (!model.FloatFeatureBorders[i].empty()) {
            for (const float border : model.FloatFeatureBorders[i]) {
                binaryFeatures[binFeatureIndex] += (unsigned char) ((floatFeatures[i] > border) || std::isnan(floatFeatures[i]));
            }
            ++binFeatureIndex;
        }
    }
}
