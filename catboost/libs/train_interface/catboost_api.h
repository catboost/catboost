#pragma once

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

#ifdef _WIN32
#ifdef _WINDLL
#define CATBOOST_API __declspec(dllexport)
#else
#define CATBOOST_API __declspec(dllimport)
#endif
#else
#define CATBOOST_API
#endif

typedef void* ResultHandle;

CATBOOST_API void FreeHandle(ResultHandle* modelHandle);

CATBOOST_API const char* GetErrorString();

CATBOOST_API int TreesCount(ResultHandle handle);
CATBOOST_API int OutputDim(ResultHandle handle);
CATBOOST_API int TreeDepth(ResultHandle handle, int treeIndex);
CATBOOST_API bool CopyTree(ResultHandle handle, int treeIndex, int* features, float* conditions, float* leaves, float* weights);


struct TDataSet {
    const float* Features = nullptr;
    const float* Labels = nullptr;
    const float* Weights = nullptr;
    const float* Baseline = nullptr;
    int BaselineDim = 0;
    int FeaturesCount = 0;
    int SamplesCount = 0;
};

CATBOOST_API bool TrainCatBoost(const struct TDataSet* train,
                                const struct TDataSet* test,
                                const char* params,
                                ResultHandle* handle);


#if defined(__cplusplus)
}
#endif

