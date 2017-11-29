#include <iostream>
#include <random>
#include "evaluator.h"


int main(int argc, char** argv) {
    NCatboostStandalone::TOwningEvaluator evaluator("../model.bin");
    auto modelFloatFeatureCount = (size_t)evaluator.GetFloatFeatureCount();
    std::cout << "Model uses: " << modelFloatFeatureCount << " float features" << std::endl;
    std::vector<float> features(modelFloatFeatureCount);
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (size_t j = 0; j < modelFloatFeatureCount; ++j) {
        features[j] = dis(mt);
    }
    for (size_t i = 0; i < 100000; ++i) {
        evaluator.Apply(features, NCatboostStandalone::EPredictionType::RawValue);
    }
    return 0;
}
