#pragma once

#include "c_api.h"

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
     * Switch evaluation backend to CUDA device
     * @param[in] deviceId - CUDA device id, formula evaluation will be done on that device
     */
    void EnableGPUEvaluation(int deviceId = 0) {
        if (!::EnableGPUEvaluation(CalcerHolder.get(), deviceId)) {
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
        FromStringToCharVector(catFeatures, &catFeaturesPtrs);
        const char** catFeaturesPtr = catFeaturesPtrs.data();
        if (!CalcModelPrediction(CalcerHolder.get(), 1, &floatPtr, floatFeatures.size(), &catFeaturesPtr, catFeatures.size(), &result, 1)) {
            throw std::runtime_error(GetErrorString());
        }
        return result;
    }

    /**
     * Evaluate model on single object float features vector, vector of categorical features strings and
     * vector of text features strings.
     * Don't work on multiclass models (models with ApproxDimension > 1)
     * @param[in] features
     * @return double raw model prediction
     */
    double Calc(
        const std::vector<float>& floatFeatures,
        const std::vector<std::string>& catFeatures,
        const std::vector<std::string>& textFeatures
    ) const {
        double result;
        const float* floatPtr = floatFeatures.data();

        std::vector<const char*> catFeaturesPtrs;
        FromStringToCharVector(catFeatures, &catFeaturesPtrs);
        const char** catFeaturesPtr = catFeaturesPtrs.data();

        std::vector<const char*> textFeaturesPtrs;
        FromStringToCharVector(textFeatures, &textFeaturesPtrs);
        const char** textFeaturesPtr = textFeaturesPtrs.data();
        if (!CalcModelPredictionText(
            CalcerHolder.get(), 1,
            &floatPtr, floatFeatures.size(),
            &catFeaturesPtr, catFeatures.size(),
            &textFeaturesPtr, textFeatures.size(),
            &result, 1
        )) {
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
     * Evaluate model on float features vector and vector of categorical feature values.
     * **WARNING** categorical features string values should not contain zero bytes in the middle of the string (latter this could be changed).
     * If so, use GetStringCatFeatureHash from model_calcer_wrapper.h and use CalcHashed method.
     * **WARNING** currently supports only singleclass models (no multiclassification support).
     * TODO(kirillovs): implement multiclass models support here.
     * @param floatFeatures
     * @param catFeature
     * @return vector of raw prediction values
     */
    std::vector<double> Calc(const std::vector<std::vector<float>>& floatFeatures,
                             const std::vector<std::vector<std::string>>& catFeatures) const {
        std::vector<double> result(floatFeatures.size());
        std::vector<const float*> floatPtrsVector;
        size_t floatFeatureCount = 0;

        for (const auto& floatFeatureVec : floatFeatures) {
            if (floatFeatureCount == 0) {
                floatFeatureCount = floatFeatureVec.size();
            }
            floatPtrsVector.push_back(floatFeatureVec.data());
        }

        size_t catFeatureCount = 0;
        std::vector<const char*> catFeaturesPtrsVector;
        std::vector<const char**> charPtrPtrsVector;
        FromStringToCharVectors(catFeatures, &catFeatureCount, &catFeaturesPtrsVector, &charPtrPtrsVector);

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
     * Evaluate model on float features vector and vector of categorical and text feature values.
     * **WARNING** categorical and text features string values should not contain zero bytes in the middle of the string (latter this could be changed).
     * If so, use GetStringCatFeatureHash from model_calcer_wrapper.h and use CalcHashed method.
     * **WARNING** currently supports only singleclass models (no multiclassification support).
     * TODO(kirillovs): implement multiclass models support here.
     * @param floatFeatures
     * @param catFeatures
     * @param textFeatures
     * @return vector of raw prediction values
     */
    std::vector<double> Calc(
        const std::vector<std::vector<float>>& floatFeatures,
        const std::vector<std::vector<std::string>>& catFeatures,
        const std::vector<std::vector<std::string>>& textFeatures
    ) const {
        std::vector<double> result(floatFeatures.size());
        std::vector<const float*> floatPtrsVector;
        size_t floatFeatureCount = 0;

        for (const auto& floatFeatureVec : floatFeatures) {
            if (floatFeatureCount == 0) {
                floatFeatureCount = floatFeatureVec.size();
            }
            floatPtrsVector.push_back(floatFeatureVec.data());
        }

        size_t catFeatureCount = 0;
        std::vector<const char*> catFeaturesPtrsVector;
        std::vector<const char**> charPtrPtrsVector;
        FromStringToCharVectors(catFeatures, &catFeatureCount, &catFeaturesPtrsVector, &charPtrPtrsVector);

        size_t textFeatureCount = 0;
        std::vector<const char*> textFeaturesPtrsVector;
        std::vector<const char**> charTextPtrPtrsVector;
        FromStringToCharVectors(textFeatures, &textFeatureCount, &textFeaturesPtrsVector, &charTextPtrPtrsVector);

        if (!CalcModelPredictionText(
            CalcerHolder.get(),
            result.size(),
            floatPtrsVector.data(), floatFeatureCount,
            charPtrPtrsVector.data(), catFeatureCount,
            charTextPtrPtrsVector.data(), textFeatureCount,
            result.data(), result.size()
        )) {
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


    bool InitFromFile(const std::string& filename) {
        return LoadFullModelFromFile(CalcerHolder.get(), filename.c_str());
    }

    bool InitFromMemory(const void* pointer, size_t size) {
        return LoadFullModelFromBuffer(CalcerHolder.get(), pointer, size);
    }

    bool init_from_file(const std::string& filename) {  // TODO(kirillovs): mark as deprecated
        return InitFromFile(filename);
    }

    size_t GetTreeCount() const {
        return ::GetTreeCount(CalcerHolder.get());
    }

    size_t GetFloatFeaturesCount() const {
        return ::GetFloatFeaturesCount(CalcerHolder.get());
    }

    size_t GetCatFeaturesCount() const {
        return ::GetCatFeaturesCount(CalcerHolder.get());
    }

    bool CheckMetadataHasKey(const std::string& key) const {
        return ::CheckModelMetadataHasKey(CalcerHolder.get(), key.c_str(), key.size());
    }

    std::string GetMetadataKeyValue(const std::string& key) const {
        if (!CheckMetadataHasKey(key)) {
            return "";
        }
        size_t value_size = GetModelInfoValueSize(CalcerHolder.get(), key.c_str(), key.size());
        const char* value_ptr = GetModelInfoValue(CalcerHolder.get(), key.c_str(), key.size());
        return std::string(value_ptr, value_size);
    }

private:
    void FromStringToCharVector(const std::vector<std::string>& stringFeatures, std::vector<const char*>* charFeatures) const {
        charFeatures->clear();
        charFeatures->reserve(stringFeatures.size());
        for (const auto& str : stringFeatures) {
            charFeatures->push_back(str.data());
        }
    }

    void FromStringToCharVectors(
        const std::vector<std::vector<std::string>>& stringFeatures,
        size_t* featureCount,
        std::vector<const char*>* featuresPtrsVector,
        std::vector<const char**>* charPtrPtrsVector
    ) const {
        size_t currentTextOffset = 0;
        for (const auto& stringVec : stringFeatures) {
            if (*featureCount == 0) {
                *featureCount = stringVec.size();
            }
            if (*featureCount != stringVec.size()) {
                throw std::runtime_error("All text feature vectors should be of the same length");
            }
        }
        if (*featureCount != 0) {
            featuresPtrsVector->reserve(stringFeatures.size() * (*featureCount));
            for (const auto& stringVec : stringFeatures) {
                for (const auto& string : stringVec) {
                    featuresPtrsVector->push_back(string.data());
                }
                charPtrPtrsVector->push_back(featuresPtrsVector->data() + currentTextOffset);
                currentTextOffset += *featureCount;
            }
        }
    }

    using CalcerHolderType = std::unique_ptr<ModelCalcerHandle, std::function<void(ModelCalcerHandle*)>>;
    CalcerHolderType CalcerHolder;
};
