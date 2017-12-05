#pragma once

#include "model_calcer_wrapper.h"

#include <string>
#include <array>
#include <vector>
#include <functional>
#include <memory>

class ModelCalcerWrapper {
public:
    ModelCalcerWrapper() = default;

    explicit ModelCalcerWrapper(const std::string& filename) {
        CalcerHolder = CalcerHolderType(ModelCalcerCreate(), ModelCalcerDelete);

        if (!LoadFullModelFromFile(CalcerHolder.get(), filename.c_str()) ) {
            throw std::runtime_error(GetErrorString());
        }
    }

    double CalcFlat(const std::vector<float>& features) {
        double result;
        const float* ptr = features.data();
        if (!CalcModelPredictionFlat(CalcerHolder.get(), 1, &ptr, features.size(), &result, 1)) {
            throw std::runtime_error(GetErrorString());
        }
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
        if (!CalcModelPrediction(CalcerHolder.get(), 1, &floatPtr, floatFeatures.size(), &catFeaturesPtr, catFeatures.size(), &result, 1)) {
            throw std::runtime_error(GetErrorString());
        }
        return result;
    }

    std::vector<double> CalcFlat(const std::vector<std::vector<float>>& features) {
        std::vector<double> result(features.size());
        std::vector<const float*> ptrsVector;
        size_t flatVecSize = 0;
        for (const auto& flatVec : features) {
            flatVecSize = flatVec.size();
            // TODO: add check that all flatVecSize are equal
            ptrsVector.push_back(flatVec.data());
        }
        if (!CalcModelPredictionFlat(CalcerHolder.get(), features.size(), ptrsVector.data(), flatVecSize, result.data(), result.size())) {
            throw std::runtime_error(GetErrorString());
        }
        return result;
    }

    std::vector<double> CalcHashed(const std::vector<std::vector<float>>& floatFeatures,
                                   const std::vector<std::vector<int>>& catFeatureHashes) {
        std::vector<double> result(floatFeatures.size());
        std::vector<const float*> floatPtrsVector;
        std::vector<const int*> hashPtrsVector;
        size_t floatFeatureCount = 0;

        for (const auto& floatFeatureVec : floatFeatures) {
            floatFeatureCount = floatFeatureVec.size();
            floatPtrsVector.push_back(floatFeatureVec.data());
        }
        size_t catFeatureCount = 0;
        for (const auto& hashVec : catFeatureHashes) {
            catFeatureCount = hashVec.size();
            hashPtrsVector.push_back(hashVec.data());
        }

        if (!CalcModelPredictionWithHashedCatFeatures(
            CalcerHolder.get(),
            result.size(),
            floatPtrsVector.data(), floatFeatureCount,
            hashPtrsVector.data(), catFeatureCount,
            result.data(), result.size())
            ) {
            throw std::runtime_error(GetErrorString());
        }
        return result;
    }

    bool init_from_file(const std::string& filename) {
        return LoadFullModelFromFile(CalcerHolder.get(), filename.c_str());
    }

private:
    using CalcerHolderType = std::unique_ptr<ModelCalcerHandle, std::function<void(ModelCalcerHandle*)>>;
    CalcerHolderType CalcerHolder;
};
