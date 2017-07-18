#include "model_calcer_wrapper.h"

#include <catboost/libs/model/model_calcer.h>

#include <util/stream/file.h>

using namespace NCatBoost;


#if defined(_win_)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#define CALCER(x) ((TFullModelCalcer*)(x))

extern "C" {

EXPORT ModelCalcerHandle* ModelCalcerCreate() {
    try {
        return new TFullModelCalcer;
    } catch (...) {
    }

    return nullptr;
}

EXPORT ModelCalcerHandle* LoadModelCalcerFromFile(const char* filename) {
    try {
        THolder<TFullModelCalcer> modelHolder;
        TIFStream inputf(filename);
        modelHolder->Load(&inputf);
        return modelHolder.Release();
    } catch (...) {
    }

    return nullptr;
}

EXPORT void ModelCalcerDelete(ModelCalcerHandle* calcer) {
    if (calcer != nullptr) {
        delete CALCER(calcer);
    }
}

EXPORT float PredictFloatValue(ModelCalcerHandle* calcer, const float* features, int resultId) {
    return CALCER(calcer)->CalcOneResult<float>(features, resultId);
}
EXPORT double PredictDoubleValue(ModelCalcerHandle* calcer, const float* features, int resultId) {
    return CALCER(calcer)->CalcOneResult<double>(features, resultId);
}

EXPORT void PredictMultiFloatValue(ModelCalcerHandle* calcer, const float* features, float* results, int resultsSize) {
    return CALCER(calcer)->CalcMulti<float>(features, results, resultsSize);
}
EXPORT void PredictMultiDoubleValue(ModelCalcerHandle* calcer, const float* features, double* results, int resultsSize) {
    return CALCER(calcer)->CalcMulti<double>(features, results, resultsSize);
}

}
