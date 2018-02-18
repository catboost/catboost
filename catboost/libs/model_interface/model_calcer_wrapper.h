#pragma once

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

#ifdef _WIN32
#ifdef _WINDLL
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
#endif
#else
#define EXPORT
#endif

typedef void ModelCalcerHandle;

/**
 * Create empty model handle
 * @return
 */
EXPORT ModelCalcerHandle* ModelCalcerCreate();

/**
 * Delete model handle
 * @param calcer
 */
EXPORT void ModelCalcerDelete(ModelCalcerHandle* calcer);

/**
 * If error occured will return stored exception message.
 * If no error occured, will return invalid pointer
 * @return
 */
EXPORT const char* GetErrorString();

/**
 * Load model from file into given model handle
 * @param calcer
 * @param filename
 * @return false if error occured
 */
EXPORT bool LoadFullModelFromFile(
    ModelCalcerHandle* calcer,
    const char* filename);

/**
 * Load model from memory buffer into given model handle
 * @param calcer
 * @param binaryBuffer pointer to a memory buffer where model file is mapped
 * @param binaryBufferSize size of the buffer in bytes
 * @return false if error occured
 */
EXPORT bool LoadFullModelFromBuffer(
    ModelCalcerHandle* calcer,
    const void* binaryBuffer,
    size_t binaryBufferSize);

/**
 * **Use this method only if you really understand what you want.**
 * Calculate raw model predictions on flat feature vectors
 * Flat here means that float features and categorical feature are in the same float array.
 * @param calcer model handle
 * @param docCount number of objects
 * @param floatFeatures array of array of float (first dimension is object index, second if feature index)
 * @param floatFeaturesSize float values array size
 * @param result pointer to user allocated results vector
 * @param resultSize Result size should be equal to modelApproxDimension * docCount
 * (e.g. for non multiclass models should be equal to docCount)
 * @return false if error occured
 */
EXPORT bool CalcModelPredictionFlat(
    ModelCalcerHandle* calcer,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    double* result, size_t resultSize);

/**
 * Calculate raw model predictions on float features and string categorical feature values
 * @param calcer modle handle
 * @param docCount object count
 * @param floatFeatures array of array of float (first dimension is object index, second if feature index)
 * @param floatFeaturesSize float feature count
 * @param catFeatures array of array of char* categorical value pointers.
 * String pointer should point to zero terminated string.
 * @param catFeaturesSize categorical feature count
 * @param result pointer to user allocated results vector
 * @param resultSize result size should be equal to modelApproxDimension * docCount
 * (e.g. for non multiclass models should be equal to docCount)
 * @return false if error occured
 */
EXPORT bool CalcModelPrediction(
    ModelCalcerHandle* calcer,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    const char*** catFeatures, size_t catFeaturesSize,
    double* result, size_t resultSize);

/**
 * Calculate raw model predictions on float features and hashed categorical feature values
 * @param calcer modle handle
 * @param docCount object count
 * @param floatFeatures array of array of float (first dimension is object index, second if feature index)
 * @param floatFeaturesSize float feature count
 * @param catFeatures array of array of integers - hashed categorical feature values.
 * @param catFeaturesSize categorical feature count
 * @param result pointer to user allocated results vector
 * @param resultSize result size should be equal to modelApproxDimension * docCount
 * (e.g. for non multiclass models should be equal to docCount)
 * @return
 */
EXPORT bool CalcModelPredictionWithHashedCatFeatures(
    ModelCalcerHandle* calcer,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    const int** catFeatures, size_t catFeaturesSize,
    double* result, size_t resultSize);

/**
 * Get hash for given string value
 * @param data we don't expect data to be zero terminated, so pass correct size
 * @param size string length
 * @return hash value
 */
EXPORT int GetStringCatFeatureHash(const char* data, size_t size);

/**
 * Special case for hash calculation - integer hash.
 * Internally we cast value to string and then calulcate string hash function.
 * Used in ClickHouse for catboost model evaluation on integer cat features.
 * @param val integer cat feature value
 * @return hash value
 */
EXPORT int GetIntegerCatFeatureHash(long long val);

/**
 * Get expected float feature count for model
 * @param calcer model handle
 */
EXPORT size_t GetFloatFeaturesCount(ModelCalcerHandle* calcer);

/**
 * Get expected categorical feature count for model
 * @param calcer model handle
 */
EXPORT size_t GetCatFeaturesCount(ModelCalcerHandle* calcer);

#if defined(__cplusplus)
}
#endif
