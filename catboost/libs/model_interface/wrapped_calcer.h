#pragma once

#include "model_calcer_wrapper.h"

#include <string>
#include <array>
#include <vector>
#include <functional>
#include <memory>

/**
 * Model C API header-only wrapper class
 * Currently supports only raw-value predictions
 * TODO(kirillovs): add support for probability and class results postprocessing
 */
class ModelCalcerWrapper {
public:
    /**
     * Create empty model
     */
    ModelCalcerWrapper()
        : CalcerHolder(CalcerHolderType(ModelCalcerCreate(), ModelCalcerDelete))
    {}
    /**
     * Load model from file
     * @param[in] filename
     */
    explicit ModelCalcerWrapper(const std::string& filename) {
        CalcerHolder = CalcerHolderType(ModelCalcerCreate(), ModelCalcerDelete);

        if (!LoadFullModelFromFile(CalcerHolder.get(), filename.c_str()) ) {
            throw std::runtime_error(GetErrorString());
        }
    }
    /**
     * Load model from memory buffer
     * @param[in] binaryBuffer
     * @param[in] binaryBufferSize
     */
    explicit ModelCalcerWrapper(const void* binaryBuffer, size_t binaryBufferSize) {
        CalcerHolder = CalcerHolderType(ModelCalcerCreate(), ModelCalcerDelete);

        if (!LoadFullModelFromBuffer(CalcerHolder.get(), binaryBuffer, binaryBufferSize) ) {
            throw std::runtime_error(GetErrorString());
        }
    }
    /**
     * Evaluate model on single object flat features vector.
     * Flat here means that float features and categorical feature are in the same float array.
     * Don't work on multiclass models (models with ApproxDimension > 1)
     * @param[in] features
     * @return double raw model prediction
     */
    double CalcFlat(const std::vector<float>& features) const {
        double result;
        const float* ptr = features.data();
        if (!CalcModelPredictionFlat(CalcerHolder.get(), 1, &ptr, features.size(), &result, 1)) {
            throw std::runtime_error(GetErrorString());
        }
        return result;
    }

    /**
     * Evaluate model on single object float features vector and vector of categorical features strings.
     * Don't work on multiclass models (models with ApproxDimension > 1)
     * @param[in] features
     * @return double raw model prediction
     */
    double Calc(const std::vector<float>& floatFeatures, const std::vector<std::string>& catFeatures) const {
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

    /**
     * Evaluate model on flat feature vectors for multiple objects.
     * Flat here means that float features and categorical feature are in the same float array.
     * **WARNING** currently supports only singleclass models.
     * TODO(kirillovs): implement multiclass models support here.
     * @param features
     * @return vector of raw prediction values
     */
    std::vector<double> CalcFlat(const std::vector<std::vector<float>>& features) const {
        std::vector<double> result(features.size());
        std::vector<const float*> ptrsVector;
        size_t flatVecSize = 0;
        for (const auto& flatVec : features) {
            flatVecSize = flatVec.size();
            // TODO(kirillovs): add check that all flatVecSize are equal
            ptrsVector.push_back(flatVec.data());
        }
        if (!CalcModelPredictionFlat(CalcerHolder.get(), features.size(), ptrsVector.data(), flatVecSize, result.data(), result.size())) {
            throw std::runtime_error(GetErrorString());
        }
        return result;
    }
    /**
     * Evaluate model on float features vector and vector of hashed categorical feature values.
     * **WARNING** categorical features string values should not contain zero bytes in the middle of the string (latter this could be changed).
     * If so, use GetStringCatFeatureHash from model_calcer_wrapper.h and use CalcHashed method.
     * **WARNING** currently supports only singleclass models (no multiclassification support).
     * TODO(kirillovs): implement multiclass models support here.
     * @param floatFeatures
     * @param catFeatureHashes
     * @return vector of raw prediction values
     */
    std::vector<double> Calc(const std::vector<std::vector<float>>& floatFeatures,
                             const std::vector<std::vector<std::string>>& catFeatures) const {
        std::vector<double> result(floatFeatures.size());
        std::vector<const float*> floatPtrsVector;
        std::vector<const char*> catFeaturesPtrsVector;
        std::vector<const char**> charPtrPtrsVector;
        size_t floatFeatureCount = 0;

        for (const auto& floatFeatureVec : floatFeatures) {
            floatFeatureCount = floatFeatureVec.size();
            floatPtrsVector.push_back(floatFeatureVec.data());
        }

        size_t catFeatureCount = 0;
        size_t currentOffset = 0;
        for (const auto& stringVec : catFeatures) {
            catFeatureCount = stringVec.size();
            if (catFeatureCount == 0) {
                break;
            }
            for (const auto& string : stringVec) {
                catFeaturesPtrsVector.push_back(string.data());
            }
            charPtrPtrsVector.push_back(catFeaturesPtrsVector.data() + currentOffset);
            currentOffset += catFeatureCount;
        }

        if (!CalcModelPrediction(
            CalcerHolder.get(),
            result.size(),
            floatPtrsVector.data(), floatFeatureCount,
            charPtrPtrsVector.data(), catFeatureCount,
            result.data(), result.size())
            ) {
            throw std::runtime_error(GetErrorString());
        }
        return result;
    }

    /**
     * Evaluate model on float features vector and vector of hashed categorical feature values.
     * **WARNING** currently supports only singleclass models (no multiclassification support).
     * TODO(kirillovs): implement multiclass models support here.
     * @param floatFeatures
     * @param catFeatureHashes
     * @return vector of raw prediction values
     */
    std::vector<double> CalcHashed(const std::vector<std::vector<float>>& floatFeatures,
                                   const std::vector<std::vector<int>>& catFeatureHashes) const {
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
