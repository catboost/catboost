#include "model_calcer_wrapper.h"

#include <catboost/libs/model/model_calcer.h>

#include <util/stream/file.h>

using namespace NCatBoost;

#define CALCER(x) ((TModelCalcer*)(x))

extern "C" {
EXPORT ModelCalcerHandle* ModelCalcerCreate() {
    try {
        return new TModelCalcer;
    } catch (...) {
    }

    return nullptr;
}

EXPORT void ModelCalcerDelete(ModelCalcerHandle* calcer) {
    if (calcer != nullptr) {
        delete CALCER(calcer);
    }
}

EXPORT bool LoadFullModelFromFile(ModelCalcerHandle* calcer, const char* filename) {
    try {
        TFullModel fullModel;
        TIFStream inputf(filename);
        fullModel.Load(&inputf);
        CALCER(calcer)->InitFromFullModel(std::move(fullModel));
    } catch (...) {
        return false;
    }

    return true;
}

EXPORT void CalcModelPredition(ModelCalcerHandle* calcer, size_t docCount, const float** features, size_t featuresSize, double* result, size_t resultSize) {
    yvector<NArrayRef::TConstArrayRef<float>> featuresVec(docCount);
    for (size_t i = 0; i < docCount; ++i) {
        featuresVec[i] = NArrayRef::TConstArrayRef<float>(features[i], featuresSize);
    }
    CALCER(calcer)->CalcFlat(featuresVec, NArrayRef::TArrayRef<double>(result, resultSize));
}
}
