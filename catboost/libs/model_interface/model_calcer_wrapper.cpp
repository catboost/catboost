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

EXPORT void CalcModelPredictionFlat(ModelCalcerHandle* calcer, size_t docCount, const float** floatFeatures, size_t floatFeaturesSize, double* result, size_t resultSize) {
    yvector<NArrayRef::TConstArrayRef<float>> featuresVec(docCount);
    for (size_t i = 0; i < docCount; ++i) {
        featuresVec[i] = NArrayRef::TConstArrayRef<float>(floatFeatures[i], floatFeaturesSize);
    }
    CALCER(calcer)->CalcFlat(featuresVec, NArrayRef::TArrayRef<double>(result, resultSize));
}

void CalcModelPrediction(ModelCalcerHandle* calcer, size_t docCount, const float** floatFeatures, size_t floatFeaturesSize, const char*** catFeatures, size_t catFeaturesSize, double* result, size_t resultSize) {
    yvector<NArrayRef::TConstArrayRef<float>> floatFeaturesVec(docCount);
    yvector<yvector<TStringBuf>> catFeaturesVec(docCount, yvector<TStringBuf>(catFeaturesSize));
    for (size_t i = 0; i < docCount; ++i) {
        floatFeaturesVec[i] = NArrayRef::TConstArrayRef<float>(floatFeatures[i], floatFeaturesSize);
        for (size_t catFeatureIdx = 0; catFeatureIdx < catFeaturesSize; ++catFeatureIdx) {
            catFeaturesVec[i][catFeatureIdx] = catFeatures[i][catFeatureIdx];
        }
    }
    CALCER(calcer)->Calc(floatFeaturesVec, catFeaturesVec, NArrayRef::TArrayRef<double>(result, resultSize));
}
}
