#pragma once

#include <stdbool.h>
#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif


#if defined(_WIN32) && !defined(CATBOOST_API_STATIC_LIB)
#ifdef _WINDLL
#define CATBOOST_API __declspec(dllexport)
#else
#define CATBOOST_API __declspec(dllimport)
#endif
#else
#define CATBOOST_API
#endif

typedef void ModelCalcerHandle;

/**
 * Create empty model handle
 * @return
 */
CATBOOST_API ModelCalcerHandle* ModelCalcerCreate();

/**
 * Delete model handle
 * @param calcer
 */
CATBOOST_API void ModelCalcerDelete(ModelCalcerHandle* modelHandle);

/**
 * If error occured will return stored exception message.
 * If no error occured, will return invalid pointer
 * @return
 */
CATBOOST_API const char* GetErrorString();

/**
 * Load model from file into given model handle
 * @param calcer
 * @param filename
 * @return false if error occured
 */
CATBOOST_API bool LoadFullModelFromFile(
    ModelCalcerHandle* modelHandle,
    const char* filename);

/**
 * Load model from memory buffer into given model handle
 * @param calcer
 * @param binaryBuffer pointer to a memory buffer where model file is mapped
 * @param binaryBufferSize size of the buffer in bytes
 * @return false if error occured
 */
CATBOOST_API bool LoadFullModelFromBuffer(
    ModelCalcerHandle* modelHandle,
    const void* binaryBuffer,
    size_t binaryBufferSize);

/**
 * Use CUDA gpu device for model evaluation
*/
CATBOOST_API bool EnableGPUEvaluation(ModelCalcerHandle* modelHandle, int deviceId);

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
CATBOOST_API bool CalcModelPredictionFlat(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    double* result, size_t resultSize);

/**
 * Calculate raw model predictions on float features and string categorical feature values
 * @param calcer model handle
 * @param docCount object count
 * @param floatFeatures array of array of float (first dimension is object index, second is feature index)
 * @param floatFeaturesSize float feature count
 * @param catFeatures array of array of char* categorical value pointers.
 * String pointer should point to zero terminated string.
 * @param catFeaturesSize categorical feature count
 * @param result pointer to user allocated results vector
 * @param resultSize result size should be equal to modelApproxDimension * docCount
 * (e.g. for non multiclass models should be equal to docCount)
 * @return false if error occured
 */
CATBOOST_API bool CalcModelPrediction(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    const char*** catFeatures, size_t catFeaturesSize,
    double* result, size_t resultSize);

/**
 * Calculate raw model predictions on float features and string categorical feature values
 * @param calcer model handle
 * @param docCount object count
 * @param floatFeatures array of array of float (first dimension is object index, second is feature index)
 * @param floatFeaturesSize float feature count
 * @param catFeatures array of array of char* categorical value pointers.
 * String pointer should point to zero terminated string.
 * @param catFeaturesSize categorical feature count
 * @param textFeatures array of array of char* text value pointers.
 * String pointer should point to zero terminated string.
 * @param textFeaturesSize text feature count
 * @param result pointer to user allocated results vector
 * @param resultSize result size should be equal to modelApproxDimension * docCount
 * (e.g. for non multiclass models should be equal to docCount)
 * @return false if error occured
 */
CATBOOST_API bool CalcModelPredictionText(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    const char*** catFeatures, size_t catFeaturesSize,
    const char*** textFeatures, size_t textFeaturesSize,
    double* result, size_t resultSize);

/**
 * Calculate raw model prediction on float features and string categorical feature values for single object
 * @param calcer model handle
 * @param floatFeatures array of float features
 * @param floatFeaturesSize float feature count
 * @param catFeatures array of char* categorical feature value pointers.
 * Each string pointer should point to zero terminated string.
 * @param catFeaturesSize categorical feature count
 * @param result pointer to user allocated results vector (or single double)
 * @param resultSize result size should be equal to modelApproxDimension
 * (e.g. for non multiclass models should be equal to 1)
 * @return false if error occured
 */
CATBOOST_API bool CalcModelPredictionSingle(
        ModelCalcerHandle* modelHandle,
        const float* floatFeatures, size_t floatFeaturesSize,
        const char** catFeatures, size_t catFeaturesSize,
        double* result, size_t resultSize);


/**
 * Calculate raw model predictions on float features and hashed categorical feature values
 * @param calcer model handle
 * @param docCount object count
 * @param floatFeatures array of array of float (first dimension is object index, second if feature index)
 * @param floatFeaturesSize float feature count
 * @param catFeatures array of array of integers - hashed categorical feature values.
 * @param catFeaturesSize categorical feature count
 * @param result pointer to user allocated results vector
 * @param resultSize result size should be equal to modelApproxDimension * docCount
 * (e.g. for non multiclass models should be equal to docCount)
 * @return false if error occured
 */
CATBOOST_API bool CalcModelPredictionWithHashedCatFeatures(
    ModelCalcerHandle* modelHandle,
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
CATBOOST_API int GetStringCatFeatureHash(const char* data, size_t size);

/**
 * Special case for hash calculation - integer hash.
 * Internally we cast value to string and then calulcate string hash function.
 * Used in ClickHouse for catboost model evaluation on integer cat features.
 * @param val integer cat feature value
 * @return hash value
 */
CATBOOST_API int GetIntegerCatFeatureHash(long long val);

/**
 * Get expected float feature count for model
 * @param calcer model handle
 */
CATBOOST_API size_t GetFloatFeaturesCount(ModelCalcerHandle* modelHandle);

/**
 * Get expected categorical feature count for model
 * @param calcer model handle
 */
CATBOOST_API size_t GetCatFeaturesCount(ModelCalcerHandle* modelHandle);

/**
 * Get number of trees in model
 * @param calcer model handle
 */
CATBOOST_API size_t GetTreeCount(ModelCalcerHandle* modelHandle);

/**
 * Get number of dimensions in model
 * @param calcer model handle
 */
CATBOOST_API size_t GetDimensionsCount(ModelCalcerHandle* modelHandle);

/**
 * Check if model metadata holds some value for provided key
 * @param calcer model handle
 */
CATBOOST_API bool CheckModelMetadataHasKey(ModelCalcerHandle* modelHandle, const char* keyPtr, size_t keySize);

/**
 * Get model metainfo value size for some key. Returns 0 both if key is missing in model metadata and if it is really missing
 * @param calcer model handle
 */
CATBOOST_API size_t GetModelInfoValueSize(ModelCalcerHandle* modelHandle, const char* keyPtr, size_t keySize);

/**
 * Get model metainfo for some key. Returns const char* pointer to inner string. If key is missing in model metainfo storage this method will return nullptr
 * @param calcer model handle
 */
CATBOOST_API const char* GetModelInfoValue(ModelCalcerHandle* modelHandle, const char* keyPtr, size_t keySize);

#if defined(__cplusplus)
}
#endif
