#pragma once

#include "model_calcer_wrapper.h"
#include <array>
#include <vector>
#include <functional>
#include <memory>

class ModelCalcerWrapper {
public:
    ModelCalcerWrapper() {
    }

    ModelCalcerWrapper(const std::string& filename) {
        CalcerHolder = CalcerHolderType(ModelCalcerCreate(), ModelCalcerDelete);
        LoadFullModelFromFile(CalcerHolder.get(), filename.c_str());
    }

    double CalcFlat(const std::vector<float>& features) {
        double result;
        const float* ptr = &features[0];
        CalcModelPredictionFlat(CalcerHolder.get(), 1, &ptr, features.size(), &result, 1);
        return result;
    }

    double Calc(const std::vector<float>& floatFeatures, const std::vector<std::string>& catFeatures) {
        double result;
        const float* floatPtr = floatFeatures.data();
        std::vector<const char*> catFeaturesPtrs;
        catFeaturesPtrs.reserve(catFeatures.size());
        for (const auto& str : catFeatures) {
            catFeaturesPtrs.push_back(str.data());
        }
        const char** catFeaturesPtr = catFeaturesPtrs.data();
        CalcModelPrediction(CalcerHolder.get(), 1, &floatPtr, floatFeatures.size(), &catFeaturesPtr, catFeatures.size(), &result, 1);
        return result;
    }

    bool init_from_file(const std::string& filename) {
        return LoadFullModelFromFile(CalcerHolder.get(), filename.c_str());
    }

private:
    using CalcerHolderType = std::unique_ptr<ModelCalcerHandle, std::function<void(ModelCalcerHandle*)>>;
    CalcerHolderType CalcerHolder;
};
