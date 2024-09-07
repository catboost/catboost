#pragma once

#include <stdbool.h>
#include <stddef.h>


#define CATBOOST_APPLIER_MAJOR 1
#define CATBOOST_APPLIER_MINOR 2
#define CATBOOST_APPLIER_FIX 7

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

typedef void DataWrapperHandle;

typedef void DataProviderHandle;

/**
 * Create empty data wrapper
 * @return
 */
CATBOOST_API DataWrapperHandle* DataWrapperCreate(size_t docsCount);

CATBOOST_API void DataWrapperDelete(DataWrapperHandle* dataWrapperHandle);

CATBOOST_API void AddFloatFeatures(DataWrapperHandle* dataWrapperHandle, const float** floatFeatures, size_t floatFeaturesSize);

CATBOOST_API void AddCatFeatures(DataWrapperHandle* dataWrapperHandle, const char*** catFeatures, size_t catFeaturesSize);

CATBOOST_API void AddTextFeatures(DataWrapperHandle* dataWrapperHandle, const char*** textFeatures, size_t textFeaturesSize);

CATBOOST_API void AddEmbeddingFeatures(DataWrapperHandle* dataWrapperHandle, const float*** embeddingFeatures, size_t* embeddingDimensions, size_t embeddingFeaturesSize);

CATBOOST_API DataProviderHandle* BuildDataProvider(DataWrapperHandle* dataWrapperHandle);

typedef void ModelCalcerHandle;

enum EApiPredictionType {
    APT_RAW_FORMULA_VAL = 0,
    APT_EXPONENT = 1,
    APT_RMSE_WITH_UNCERTAINTY = 2,
    APT_PROBABILITY = 3,
    APT_CLASS = 4,
    APT_MULTI_PROBABILITY = 5,
};

enum ECatBoostApiFormulaEvaluatorType {
    CBA_FET_CPU = 0,
    CBA_FET_GPU = 1,
};

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
 * Use CUDA GPU device for model evaluation
*/
CATBOOST_API bool EnableGPUEvaluation(ModelCalcerHandle* modelHandle, int deviceId);

/**
 * Get supported formula evaluator types
 * formulaEvaluatorTypes array must be deallocated using free() after use.
 *
 * @param modelHandle model handle
 * @param formulaEvaluatorTypes address of the pointer to an array that will be initialized with formula evaluator types
 * @param formulaEvaluatorTypesCount address of the variable where the size of formulaEvaluatorTypes array will be stored
 * @return true on success, false on error
 */
CATBOOST_API bool GetSupportedEvaluatorTypes(
    ModelCalcerHandle* modelHandle,
    enum ECatBoostApiFormulaEvaluatorType** formulaEvaluatorTypes,
    size_t* formulaEvaluatorTypesCount);


/**
 * Set prediction type for model evaluation
*/
CATBOOST_API bool SetPredictionType(ModelCalcerHandle* modelHandle, enum EApiPredictionType predictionType);

/**
 * Set prediction type for model evaluation with string constant
*/
CATBOOST_API bool SetPredictionTypeString(ModelCalcerHandle* modelHandle, const char* predictionTypeStr);


/**
 * **Use this method only if you really understand what you want.**
 * Calculate raw model predictions on flat feature vectors
 * Flat here means that float features and categorical feature are in the same float array.
 * @param calcer model handle
 * @param docCount number of objects
 * @param floatFeatures array of array of float (first dimension is object index, second is feature index)
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
 * **Use this method only if you really understand what you want.**
 * Calculate raw model predictions on flat feature vectors
 * taking into consideration only the trees in the range [treeStart; treeEnd)
 * Flat here means that float features and categorical feature are in the same float array.
 * @param calcer model handle
 * @param docCount number of objects
 * @param treeStart the index of the first tree to be used when applying the model (zero-based)
 * @param treeEnd the index of the last tree to be used when applying the model (non-inclusive, zero-based)
 * @param floatFeatures array of array of float (first dimension is object index, second is feature index)
 * @param floatFeaturesSize float values array size
 * @param result pointer to user allocated results vector
 * @param resultSize Result size should be equal to modelApproxDimension * docCount
 * (e.g. for non multiclass models should be equal to docCount)
 * @return false if error occured
 */
CATBOOST_API bool CalcModelPredictionFlatStaged(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    size_t treeStart, size_t treeEnd,
    const float** floatFeatures, size_t floatFeaturesSize,
    double* result, size_t resultSize);


/**
 * **Use this method only if you really understand what you want.**
 * Calculate raw model predictions on transposed dataset layout
 * @param calcer model handle
 * @param docCount number of objects
 * @param floatFeatures array of array of float (first dimension is feature index, second is object index)
 * @param floatFeaturesSize float values array size
 * @param result pointer to user allocated results vector
 * @param resultSize Result size should be equal to modelApproxDimension * docCount
 * (e.g. for non multiclass models should be equal to docCount)
 * @return false if error occured
 */
CATBOOST_API bool CalcModelPredictionFlatTransposed(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    double* result, size_t resultSize);


/**
 * **Use this method only if you really understand what you want.**
 * Calculate raw model predictions on transposed dataset layout
 * taking into consideration only the trees in the range [treeStart; treeEnd)
 * @param calcer model handle
 * @param docCount number of objects
 * @param treeStart the index of the first tree to be used when applying the model (zero-based)
 * @param treeEnd the index of the last tree to be used when applying the model (non-inclusive, zero-based)
 * @param floatFeatures array of array of float (first dimension is feature index, second is object index)
 * @param floatFeaturesSize float values array size
 * @param result pointer to user allocated results vector
 * @param resultSize Result size should be equal to modelApproxDimension * docCount
 * (e.g. for non multiclass models should be equal to docCount)
 * @return false if error occured
 */
CATBOOST_API bool CalcModelPredictionFlatTransposedStaged(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    size_t treeStart, size_t treeEnd,
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
 * taking into consideration only the trees in the range [treeStart; treeEnd)
 * @param calcer model handle
 * @param docCount object count
 * @param treeStart the index of the first tree to be used when applying the model (zero-based)
 * @param treeEnd the index of the last tree to be used when applying the model (non-inclusive, zero-based)
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
CATBOOST_API bool CalcModelPredictionStaged(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    size_t treeStart, size_t treeEnd,
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
 * Calculate raw model predictions on float features and string categorical feature values
 * taking into consideration only the trees in the range [treeStart; treeEnd)
 * @param calcer model handle
 * @param docCount object count
 * @param treeStart the index of the first tree to be used when applying the model (zero-based)
 * @param treeEnd the index of the last tree to be used when applying the model (non-inclusive, zero-based)
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
CATBOOST_API bool CalcModelPredictionTextStaged(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    size_t treeStart, size_t treeEnd,
    const float** floatFeatures, size_t floatFeaturesSize,
    const char*** catFeatures, size_t catFeaturesSize,
    const char*** textFeatures, size_t textFeaturesSize,
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
 * @param embeddingFeatures array of array of array of float (first dimension is object index, second is feature index, third is index in embedding array).
 * String pointer should point to zero terminated string.
 * @param embeddingFeaturesSize embedding feature count
 * @param result pointer to user allocated results vector
 * @param resultSize result size should be equal to modelApproxDimension * docCount
 * (e.g. for non multiclass models should be equal to docCount)
 * @return false if error occured
 */
CATBOOST_API bool CalcModelPredictionTextAndEmbeddings(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    const char*** catFeatures, size_t catFeaturesSize,
    const char*** textFeatures, size_t textFeaturesSize,
    const float*** embeddingFeatures, size_t* embeddingDimensions, size_t embeddingFeaturesSize,
    double* result, size_t resultSize);


/**
 * Calculate raw model predictions on float features and string categorical feature values
 * taking into consideration only the trees in the range [treeStart; treeEnd)
 * @param calcer model handle
 * @param docCount object count
 * @param treeStart the index of the first tree to be used when applying the model (zero-based)
 * @param treeEnd the index of the last tree to be used when applying the model (non-inclusive, zero-based)
 * @param floatFeatures array of array of float (first dimension is object index, second is feature index)
 * @param floatFeaturesSize float feature count
 * @param catFeatures array of array of char* categorical value pointers.
 * String pointer should point to zero terminated string.
 * @param catFeaturesSize categorical feature count
 * @param textFeatures array of array of char* text value pointers.
 * String pointer should point to zero terminated string.
 * @param textFeaturesSize text feature count
 * @param embeddingFeatures array of array of array of float (first dimension is object index, second is feature index, third is index in embedding array).
 * String pointer should point to zero terminated string.
 * @param embeddingFeaturesSize embedding feature count
 * @param result pointer to user allocated results vector
 * @param resultSize result size should be equal to modelApproxDimension * docCount
 * (e.g. for non multiclass models should be equal to docCount)
 * @return false if error occured
 */
CATBOOST_API bool CalcModelPredictionTextAndEmbeddingsStaged(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    size_t treeStart, size_t treeEnd,
    const float** floatFeatures, size_t floatFeaturesSize,
    const char*** catFeatures, size_t catFeaturesSize,
    const char*** textFeatures, size_t textFeaturesSize,
    const float*** embeddingFeatures, size_t* embeddingDimensions, size_t embeddingFeaturesSize,
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
 * Calculate raw model prediction on float features and string categorical feature values for single object
 * taking into consideration only the trees in the range [treeStart; treeEnd)
 * @param calcer model handle
 * @param treeStart the index of the first tree to be used when applying the model (zero-based)
 * @param treeEnd the index of the last tree to be used when applying the model (non-inclusive, zero-based)
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
CATBOOST_API bool CalcModelPredictionSingleStaged(
        ModelCalcerHandle* modelHandle,
        size_t treeStart, size_t treeEnd,
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

CATBOOST_API bool CalcModelPredictionWithHashedCatFeaturesAndTextFeatures(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    const int** catFeatures, size_t catFeaturesSize,
    const char*** textFeatures, size_t textFeaturesSize,
    double* result, size_t resultSize);

CATBOOST_API bool CalcModelPredictionWithHashedCatFeaturesAndTextAndEmbeddingFeatures(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    const int** catFeatures, size_t catFeaturesSize,
    const char*** textFeatures, size_t textFeaturesSize,
    const float*** embeddingFeatures, size_t* embeddingDimensions, size_t embeddingFeaturesSize,
    double* result, size_t resultSize);

/**
 * Methods equivalent to the methods above
 * only returning a prediction for the specific class
 * @param classId number of the class should be in [0, modelApproxDimension - 1]
 * @param resultSize result size should be equal to docCount
*/
CATBOOST_API bool PredictSpecificClassFlat(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    int classId,
    double* result, size_t resultSize);

CATBOOST_API bool PredictSpecificClass(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    const char*** catFeatures, size_t catFeaturesSize,
    int classId,
    double* result, size_t resultSize);

CATBOOST_API bool PredictSpecificClassText(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    const char*** catFeatures, size_t catFeaturesSize,
    const char*** textFeatures, size_t textFeaturesSize,
    int classId,
    double* result, size_t resultSize);

CATBOOST_API bool PredictSpecificClassTextAndEmbeddings(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    const char*** catFeatures, size_t catFeaturesSize,
    const char*** textFeatures, size_t textFeaturesSize,
    const float*** embeddingFeatures, size_t* embeddingDimensions, size_t embeddingFeaturesSize,
    int classId,
    double* result, size_t resultSize);

CATBOOST_API bool PredictSpecificClassSingle(
    ModelCalcerHandle* modelHandle,
    const float* floatFeatures, size_t floatFeaturesSize,
    const char** catFeatures, size_t catFeaturesSize,
    int classId,
    double* result, size_t resultSize);

CATBOOST_API bool PredictSpecificClassWithHashedCatFeatures(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    const int** catFeatures, size_t catFeaturesSize,
    int classId,
    double* result, size_t resultSize);

CATBOOST_API bool PredictSpecificClassWithHashedCatFeaturesAndTextFeatures(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    const int** catFeatures, size_t catFeaturesSize,
    const char*** textFeatures, size_t textFeaturesSize,
    int classId,
    double* result, size_t resultSize);

CATBOOST_API bool PredictSpecificClassWithHashedCatFeaturesAndTextAndEmbeddingFeatures(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    const int** catFeatures, size_t catFeaturesSize,
    const char*** textFeatures, size_t textFeaturesSize,
    const float*** embeddingFeatures, size_t* embeddingDimensions, size_t embeddingFeaturesSize,
    int classId,
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
 * Get expected indices of float features used in the model.
 * indices array must be deallocated using free() after use.
 * @param modelHandle model handle
 * @param indices indices of the features
 * @param count indices size
 * @return true on success, false on error
 */
CATBOOST_API bool GetFloatFeatureIndices(ModelCalcerHandle* modelHandle, size_t** indices, size_t* count);

/**
 * Get expected categorical feature count for model
 * @param calcer model handle
 */
CATBOOST_API size_t GetCatFeaturesCount(ModelCalcerHandle* modelHandle);

/**
 * Get expected indices of category features used in the model.
 * indices array must be deallocated using free() after use.
 * @param modelHandle model handle
 * @param indices indices of the features
 * @param count indices size
 * @return true on success, false on error
 */
CATBOOST_API bool GetCatFeatureIndices(ModelCalcerHandle* modelHandle, size_t** indices, size_t* count);

/**
 * Get expected text feature count for model
 * @param calcer model handle
 */
CATBOOST_API size_t GetTextFeaturesCount(ModelCalcerHandle* modelHandle);

/**
 * Get expected indices of text features used in the model.
 * indices array must be deallocated using free() after use.
 * @param modelHandle model handle
 * @param indices indices of the features
 * @param count indices size
 * @return true on success, false on error
 */
CATBOOST_API bool GetTextFeatureIndices(ModelCalcerHandle* modelHandle, size_t** indices, size_t* count);

/**
 * Get expected embedding feature count for model
 * @param calcer model handle
 */
CATBOOST_API size_t GetEmbeddingFeaturesCount(ModelCalcerHandle* modelHandle);

/**
 * Get expected indices of embedding features used in the model.
 * indices array must be deallocated using free() after use.
 * @param modelHandle model handle
 * @param indices indices of the features
 * @param count indices size
 * @return true on success, false on error
 */
CATBOOST_API bool GetEmbeddingFeatureIndices(ModelCalcerHandle* modelHandle, size_t** indices, size_t* count);

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
 * Get number of dimensions for current prediction
 * For default `APT_RAW_FORMULA_VAL`, `APT_EXPONENT`, `APT_PROBABILITY`, `APT_CLASS` prediction type GetPredictionDimensionsCount == GetDimensionsCount
 * For `APT_RMSE_WITH_UNCERTAINTY` - returns 2 (value prediction and predicted uncertainty)
 * @param calcer model handle
 */
CATBOOST_API size_t GetPredictionDimensionsCount(ModelCalcerHandle* modelHandle);


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


/**
 * Get names of features used in the model.
 * individual strings in featureNames array and featureNames array itself must be deallocated using free() after use.
 *
 * @return true on success, false on error
 */
CATBOOST_API bool GetModelUsedFeaturesNames(ModelCalcerHandle* modelHandle, char*** featureNames, size_t* featureCount);


#if defined(__cplusplus)
}
#endif
