#pragma once

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef void ModelCalcerHandle;

ModelCalcerHandle* ModelCalcerCreate();
ModelCalcerHandle* LoadModelCalcerFromFile(const char* filename);
void ModelCalcerDelete(ModelCalcerHandle* calcer);

float PredictFloatValue(ModelCalcerHandle* calcer, const float* features, int resultId);
double PredictDoubleValue(ModelCalcerHandle* calcer, const float* features, int resultId);

void PredictMultiFloatValue(ModelCalcerHandle* calcer, const float* features, float* results, int resultsSize);
void PredictMultiDoubleValue(ModelCalcerHandle* calcer, const float* features, double* results, int resultsSize);

#if defined(__cplusplus)
}
#endif
