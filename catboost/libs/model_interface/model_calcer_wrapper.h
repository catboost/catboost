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

EXPORT bool LoadFullModelFromFile(ModelCalcerHandle* calcer, const char* filename);
EXPORT void CalcModelPredictionFlat(ModelCalcerHandle* calcer, size_t docCount, const float** floatFeatures, size_t floatFeaturesSize, double* result, size_t resultSize);
EXPORT void CalcModelPrediction(ModelCalcerHandle* calcer, size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    const char*** catFeatures, size_t catFeaturesSize,
    double* result, size_t resultSize);

#if defined(__cplusplus)
}
#endif
