#pragma once

#include "c_api.h"

#include <cstdlib>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <stdexcept>

/**
 * Model C API header-only wrapper class
 * Currently supports only raw-value predictions
 * TODO(kirillovs): add support for probability and class results postprocessing
 */
class ModelCalcerWrapper {
public:
    /// TODO(kirillovs): support different prediction types
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
        InitProps();
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
        InitProps();
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
     * Get supported formula evaluator types
     */
    std::vector<ECatBoostApiFormulaEvaluatorType> GetSupportedEvaluatorTypes() {
        enum ECatBoostApiFormulaEvaluatorType* formulaEvaluatorTypes = nullptr;
        size_t formulaEvaluatorTypesCount = 0;
        if (!::GetSupportedEvaluatorTypes(CalcerHolder.get(), &formulaEvaluatorTypes, &formulaEvaluatorTypesCount)) {
            throw std::runtime_error(GetErrorString());
        }
        std::vector<ECatBoostApiFormulaEvaluatorType> result;
        try {
            for (size_t i = 0; i < formulaEvaluatorTypesCount; ++i) {
                result.push_back(formulaEvaluatorTypes[i]);
            }
        } catch (...) {
            free(formulaEvaluatorTypes);
            throw;
        }
        free(formulaEvaluatorTypes);
        return result;
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
     * Evaluate model on single object flat features vector.
     * Flat here means that float features and categorical feature are in the same float array.
     * Work for models with any dimension count
     * @param[in] features
     * @return double raw model prediction
     */
    std::vector<double> CalcFlatMulti(const std::vector<float>& features) const {
        std::vector<double> result(DimensionsCount, 0.0);
        const float* ptr = features.data();
        if (!CalcModelPredictionFlat(CalcerHolder.get(), 1, &ptr, features.size(), result.data(), DimensionsCount)) {
            throw std::runtime_error(GetErrorString());
        }
        return result;
    }

    /**
     * Evaluate model on single object float features vector, vector of categorical features strings,
     * vector of text features strings and vector of embedding features vectors.
     * Don't work on multiclass models (models with ApproxDimension > 1)
     * @param[in] features
     * @return double raw model prediction
     */
    double Calc(
        const std::vector<float>& floatFeatures,
        const std::vector<std::string>& catFeatures = {},
        const std::vector<std::string>& textFeatures = {},
        const std::vector<std::vector<float>>& embeddingFeatures = {}
    ) const {
        double result;
        const float* floatPtr = floatFeatures.data();

        std::vector<const char*> catFeaturesPtrs;
        FromStringToCharVector(catFeatures, &catFeaturesPtrs);
        const char** catFeaturesPtr = catFeaturesPtrs.data();

        std::vector<const char*> textFeaturesPtrs;
        FromStringToCharVector(textFeatures, &textFeaturesPtrs);
        const char** textFeaturesPtr = textFeaturesPtrs.data();

        std::vector<const float*> embeddingFeaturesPtrs;
        std::vector<size_t> embeddingFeatureSizes;
        for (const auto& embeddingFeatureData : embeddingFeatures) {
            embeddingFeaturesPtrs.push_back(embeddingFeatureData.data());
            embeddingFeatureSizes.push_back(embeddingFeatureData.size());
        }
        const float** embeddingFeaturesPtr = embeddingFeaturesPtrs.data();

        if (!CalcModelPredictionTextAndEmbeddings(
            CalcerHolder.get(), 1,
            &floatPtr, floatFeatures.size(),
            &catFeaturesPtr, catFeatures.size(),
            &textFeaturesPtr, textFeatures.size(),
            &embeddingFeaturesPtr, embeddingFeatureSizes.data(), embeddingFeatures.size(),
            &result, 1
        )) {
            throw std::runtime_error(GetErrorString());
        }
        return result;
    }

    /**
     * Evaluate model on single object float features vector, vector of categorical features strings,
     * vector of text features strings, and vector of embedding features vectors
     * Work for models with any dimension count
     * @param[in] features
     * @return double raw model prediction
     */
    std::vector<double> CalcMulti(
        const std::vector<float>& floatFeatures,
        const std::vector<std::string>& catFeatures = {},
        const std::vector<std::string>& textFeatures = {},
        const std::vector<std::vector<float>>& embeddingFeatures = {}
    ) const {
        std::vector<double> result(DimensionsCount);
        const float* floatPtr = floatFeatures.data();

        std::vector<const char*> catFeaturesPtrs;
        FromStringToCharVector(catFeatures, &catFeaturesPtrs);
        const char** catFeaturesPtr = catFeaturesPtrs.data();

        std::vector<const char*> textFeaturesPtrs;
        FromStringToCharVector(textFeatures, &textFeaturesPtrs);
        const char** textFeaturesPtr = textFeaturesPtrs.data();

        std::vector<const float*> embeddingFeaturesPtrs;
        std::vector<size_t> embeddingFeatureSizes;
        for (const auto& embeddingFeatureData : embeddingFeatures) {
            embeddingFeaturesPtrs.push_back(embeddingFeatureData.data());
            embeddingFeatureSizes.push_back(embeddingFeatureData.size());
        }
        const float** embeddingFeaturesPtr = embeddingFeaturesPtrs.data();

        if (!CalcModelPredictionTextAndEmbeddings(
            CalcerHolder.get(), 1,
            &floatPtr, floatFeatures.size(),
            &catFeaturesPtr, catFeatures.size(),
            &textFeaturesPtr, textFeatures.size(),
            &embeddingFeaturesPtr, embeddingFeatureSizes.data(), embeddingFeatures.size(),
            result.data(), DimensionsCount
        )) {
            throw std::runtime_error(GetErrorString());
        }
        return result;
    }

    /**
     * Evaluate model on flat feature vectors for multiple objects.
     * Flat here means that float features and categorical feature are in the same float array.
     * **WARNING** currently supports only singleclass models.
     * @param features
     * @return vector of raw prediction values
     */
    std::vector<double> CalcFlat(const std::vector<std::vector<float>>& features) const {
        std::vector<double> result(features.size() * DimensionsCount);
        std::vector<const float*> ptrsVector;
        ptrsVector.reserve(features.size());
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
     * Evaluate model on transposed dataset layout.
     * **WARNING** currently supports only singleclass models.
     * @param transposedFeatures
     * @return vector of raw prediction values
     */
    std::vector<double> CalcFlatTransposed(const std::vector<std::vector<float>>& transposedFeatures) const {
        std::vector<const float*> ptrsVector;
        ptrsVector.reserve(transposedFeatures.size());
        size_t docCount = 0;
        for (const auto& feature : transposedFeatures) {
            docCount = feature.size();
            // TODO(kirillovs): add check that all docCount are equal
            ptrsVector.push_back(feature.data());
        }
        std::vector<double> result(docCount * DimensionsCount);
        if (!CalcModelPredictionFlatTransposed(CalcerHolder.get(), docCount, ptrsVector.data(), transposedFeatures.size(), result.data(), result.size())) {
            throw std::runtime_error(GetErrorString());
        }
        return result;
    }

    /**
     * Evaluate model on vectors of float, categorical, text and embedding feature values.
     * **WARNING** categorical and text features string values should not contain zero bytes in the middle of the string (latter this could be changed).
     * If so, use GetStringCatFeatureHash from model_calcer_wrapper.h and use CalcHashed method.
     * @param floatFeatures
     * @param catFeatures
     * @param textFeatures
     * @param embeddingFeatures
     * @return vector of raw prediction values
     */
    std::vector<double> Calc(
        const std::vector<std::vector<float>>& floatFeatures,
        const std::vector<std::vector<std::string>>& catFeatures = {},
        const std::vector<std::vector<std::string>>& textFeatures = {},
        const std::vector<std::vector<std::vector<float>>>& embeddingFeatures = {}
    ) const {
        std::vector<double> result(floatFeatures.size() * DimensionsCount);
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
        FromStringToCharVectors(
            "categorical",
            catFeatures,
            &catFeatureCount,
            &catFeaturesPtrsVector,
            &charPtrPtrsVector
        );

        size_t textFeatureCount = 0;
        std::vector<const char*> textFeaturesPtrsVector;
        std::vector<const char**> charTextPtrPtrsVector;
        FromStringToCharVectors(
            "text",
            textFeatures,
            &textFeatureCount,
            &textFeaturesPtrsVector,
            &charTextPtrPtrsVector
        );

        size_t embeddingFeatureCount = 0;
        std::vector<const float*> embeddingFeaturesPtrs;
        std::vector<const float**> embeddingFeaturesPerSamplePtrs;
        std::vector<size_t> embeddingFeatureSizes;
        EmbeddingFeaturesVectorsToPtrs(
            embeddingFeatures,
            &embeddingFeatureCount,
            &embeddingFeaturesPtrs,
            &embeddingFeaturesPerSamplePtrs,
            &embeddingFeatureSizes
        );

        if (!CalcModelPredictionTextAndEmbeddings(
            CalcerHolder.get(),
            floatFeatures.size(),
            floatPtrsVector.data(), floatFeatureCount,
            charPtrPtrsVector.data(), catFeatureCount,
            charTextPtrPtrsVector.data(), textFeatureCount,
            embeddingFeaturesPerSamplePtrs.data(), embeddingFeatureSizes.data(), embeddingFeatureCount,
            result.data(), result.size()
        )) {
            throw std::runtime_error(GetErrorString());
        }
        return result;
    }

    /**
     * Evaluate model on vectors of float, hashed categorical, text and embedding feature values.
     * @param floatFeatures
     * @param catFeatureHashes
     * @param textFeatures
     * @param embeddingFeatures
     * @return vector of raw prediction values
     */
    std::vector<double> CalcHashed(const std::vector<std::vector<float>>& floatFeatures,
                                   const std::vector<std::vector<int>>& catFeatureHashes,
                                   const std::vector<std::vector<std::string>>& textFeatures = {},
                                   const std::vector<std::vector<std::vector<float>>>& embeddingFeatures = {}
                                   ) const {
        std::vector<double> result(floatFeatures.size() * DimensionsCount);
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

        size_t textFeatureCount = 0;
        std::vector<const char*> textFeaturesPtrsVector;
        std::vector<const char**> charTextPtrPtrsVector;
        FromStringToCharVectors(
            "text",
            textFeatures,
            &textFeatureCount,
            &textFeaturesPtrsVector,
            &charTextPtrPtrsVector
        );

        size_t embeddingFeatureCount = 0;
        std::vector<const float*> embeddingFeaturesPtrs;
        std::vector<const float**> embeddingFeaturesPerSamplePtrs;
        std::vector<size_t> embeddingFeatureSizes;
        EmbeddingFeaturesVectorsToPtrs(
            embeddingFeatures,
            &embeddingFeatureCount,
            &embeddingFeaturesPtrs,
            &embeddingFeaturesPerSamplePtrs,
            &embeddingFeatureSizes
        );

        if (!CalcModelPredictionWithHashedCatFeaturesAndTextAndEmbeddingFeatures(
            CalcerHolder.get(),
            floatFeatures.size(),
            floatPtrsVector.data(), floatFeatureCount,
            hashPtrsVector.data(), catFeatureCount,
            charTextPtrPtrsVector.data(), textFeatureCount,
            embeddingFeaturesPerSamplePtrs.data(), embeddingFeatureSizes.data(), embeddingFeatureCount,
            result.data(), result.size()
        )) {
            throw std::runtime_error(GetErrorString());
        }
        return result;
    }


    bool InitFromFile(const std::string& filename) {
        if (!LoadFullModelFromFile(CalcerHolder.get(), filename.c_str())) {
            return false;
        }
        InitProps();
        return true;
    }

    bool InitFromMemory(const void* pointer, size_t size) {
        if (!LoadFullModelFromBuffer(CalcerHolder.get(), pointer, size)) {
            return false;
        }
        InitProps();
        return true;
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

    std::vector<size_t> GetFloatFeatureIndices() const {
        return GetFeaturesIndicesImpl(::GetFloatFeatureIndices);
    }

    size_t GetCatFeaturesCount() const {
        return ::GetCatFeaturesCount(CalcerHolder.get());
    }

    std::vector<size_t> GetCatFeatureIndices() const {
        return GetFeaturesIndicesImpl(::GetCatFeatureIndices);
    }

    size_t GetTextFeaturesCount() const {
        return ::GetTextFeaturesCount(CalcerHolder.get());
    }

    std::vector<size_t> GetTextFeatureIndices() const {
        return GetFeaturesIndicesImpl(::GetTextFeatureIndices);
    }

    size_t GetEmbeddingFeaturesCount() const {
        return ::GetEmbeddingFeaturesCount(CalcerHolder.get());
    }

    std::vector<size_t> GetEmbeddingFeatureIndices() const {
        return GetFeaturesIndicesImpl(::GetEmbeddingFeatureIndices);
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

    std::vector<std::string> GetUsedFeaturesNames() const {
        char** featureNames = nullptr;
        size_t featureCount = 0;
        if (!GetModelUsedFeaturesNames(CalcerHolder.get(), &featureNames, &featureCount)) {
            throw std::runtime_error(GetErrorString());
        }
        std::vector<std::string> result;
        try {
            result.reserve(featureCount);
            for (size_t i = 0; i < featureCount; ++i) {
                result.push_back(std::string(featureNames[i]));
            }
        } catch (...) {
            for (size_t i = 0; i < featureCount; ++i) {
                std::free(featureNames[i]);
            }
            std::free(featureNames);
            throw;
        }

        {
            for (size_t i = 0; i < featureCount; ++i) {
                std::free(featureNames[i]);
            }
            std::free(featureNames);
        }

        return result;
    }

private:
    void InitProps() {
        DimensionsCount = GetDimensionsCount(CalcerHolder.get());
    }

    void FromStringToCharVector(const std::vector<std::string>& stringFeatures, std::vector<const char*>* charFeatures) const {
        charFeatures->clear();
        charFeatures->reserve(stringFeatures.size());
        for (const auto& str : stringFeatures) {
            charFeatures->push_back(str.data());
        }
    }

    void FromStringToCharVectors(
        const char* featuresType,
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
                throw std::runtime_error(
                    std::string("All ") + featuresType + " feature vectors should be of the same length"
                );
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

    static void EmbeddingFeaturesVectorsToPtrs(
        const std::vector<std::vector<std::vector<float>>>& embeddingFeatures,
        size_t* embeddingFeatureCount,
        std::vector<const float*>* embeddingFeaturesPtrs,
        std::vector<const float**>* embeddingFeaturesPerSamplePtrs,
        std::vector<size_t>* embeddingFeatureSizes
    ) {
        const size_t embeddingFeatureCountLocal = embeddingFeatures.empty() ? 0 : embeddingFeatures.begin()->size();
        *embeddingFeatureCount = embeddingFeatureCountLocal;
        embeddingFeaturesPtrs->resize(embeddingFeatures.size() * embeddingFeatureCountLocal);
        embeddingFeaturesPerSamplePtrs->resize(embeddingFeatures.size());

        for (size_t sampleIdx = 0; sampleIdx < embeddingFeatures.size(); ++sampleIdx) {
            if (embeddingFeatureSizes->empty()) {
                for (const auto& embeddingFeatureData : embeddingFeatures[sampleIdx]) {
                    embeddingFeatureSizes->push_back(embeddingFeatureData.size());
                }
            }
            for (size_t embeddingFeatureIdx = 0; embeddingFeatureIdx < embeddingFeatureCountLocal; ++embeddingFeatureIdx) {
                (*embeddingFeaturesPtrs)[sampleIdx * embeddingFeatureCountLocal + embeddingFeatureIdx]
                    = embeddingFeatures[sampleIdx][embeddingFeatureIdx].data();
            }
            (*embeddingFeaturesPerSamplePtrs)[sampleIdx] = embeddingFeaturesPtrs->data() + sampleIdx * embeddingFeatureCountLocal;
        }
    }

    std::vector<size_t> GetFeaturesIndicesImpl(
        std::function<bool (ModelCalcerHandle*, size_t**, size_t*)>&& cApiCall
    ) const {
        size_t* featureIndices = nullptr;
        size_t featureCount = 0;
        if (!cApiCall(CalcerHolder.get(), &featureIndices, &featureCount)) {
            throw std::runtime_error(GetErrorString());
        }

        std::vector<size_t> result;
        try {
            result.assign(featureIndices, featureIndices + featureCount);
        } catch (...) {
            std::free(featureIndices);
            throw;
        }
        std::free(featureIndices);

        return result;
    }

    using CalcerHolderType = std::unique_ptr<ModelCalcerHandle, std::function<void(ModelCalcerHandle*)>>;
    CalcerHolderType CalcerHolder;
    size_t DimensionsCount = 0;
};
