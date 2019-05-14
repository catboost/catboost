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

typedef void* ResultHandle;

EXPORT void FreeHandle(ResultHandle* modelHandle);

EXPORT const char* GetErrorString();

EXPORT int TreesCount(ResultHandle handle);
EXPORT int OutputDim(ResultHandle handle);
EXPORT int TreeDepth(ResultHandle handle, int treeIndex);
EXPORT bool CopyTree(ResultHandle handle, int treeIndex, int* features, float* conditions, float* leaves, float* weights);


struct TDataSet {
    const float* Features = nullptr;
    const float* Labels = nullptr;
    const float* Weights = nullptr;
    const float* Baseline = nullptr;
    int BaselineDim = 0;
    int FeaturesCount = 0;
    int SamplesCount = 0;
};

EXPORT bool TrainCatBoost(const struct TDataSet* train,
                          const struct TDataSet* test,
                          const char* params,
                          ResultHandle* handle);


#if defined(__cplusplus)
}
#endif

