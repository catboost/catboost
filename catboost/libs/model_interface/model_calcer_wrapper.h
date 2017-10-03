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

EXPORT ModelCalcerHandle* ModelCalcerCreate();
EXPORT void ModelCalcerDelete(ModelCalcerHandle* calcer);

EXPORT const char* GetErrorString();

EXPORT bool LoadFullModelFromFile(ModelCalcerHandle* calcer, const char* filename);
EXPORT bool CalcModelPredictionFlat(ModelCalcerHandle* calcer, size_t docCount, const float** floatFeatures, size_t floatFeaturesSize, double* result, size_t resultSize);
EXPORT bool CalcModelPrediction(ModelCalcerHandle* calcer, size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    const char*** catFeatures, size_t catFeaturesSize,
    double* result, size_t resultSize);

EXPORT bool CalcModelPredictionWithHashedCatFeatures(ModelCalcerHandle* calcer, size_t docCount,
                                const float** floatFeatures, size_t floatFeaturesSize,
                                const int** catFeatures, size_t catFeaturesSize,
                                double* result, size_t resultSize);

EXPORT int GetStringCatFeatureHash(const char* data, size_t size);
EXPORT int GetIntegerCatFeatureHash(long long val);
#if defined(__cplusplus)
}
#endif
