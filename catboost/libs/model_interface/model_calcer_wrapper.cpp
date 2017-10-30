#include "model_calcer_wrapper.h"

#include <catboost/libs/model/formula_evaluator.h>

#include <util/generic/singleton.h>
#include <util/stream/file.h>
#include <util/string/builder.h>

using namespace NCatBoost;

#define CALCER(x) ((TFormulaEvaluator*)(x))

struct TErrorMessageHolder {
    TString Message;
};

extern "C" {
EXPORT ModelCalcerHandle* ModelCalcerCreate() {
    try {
        return new TFormulaEvaluator;
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
    }

    return nullptr;
}

EXPORT const char* GetErrorString() {
    return Singleton<TErrorMessageHolder>()->Message.data();
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
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }

    return true;
}

EXPORT bool CalcModelPredictionFlat(ModelCalcerHandle* calcer, size_t docCount, const float** floatFeatures, size_t floatFeaturesSize, double* result, size_t resultSize) {
    try {
        yvector<NArrayRef::TConstArrayRef<float>> featuresVec(docCount);
        for (size_t i = 0; i < docCount; ++i) {
            featuresVec[i] = NArrayRef::TConstArrayRef<float>(floatFeatures[i], floatFeaturesSize);
        }
        CALCER(calcer)->CalcFlat(featuresVec, NArrayRef::TArrayRef<double>(result, resultSize));
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

EXPORT bool CalcModelPrediction(ModelCalcerHandle* calcer, size_t docCount, const float** floatFeatures, size_t floatFeaturesSize, const char*** catFeatures, size_t catFeaturesSize, double* result, size_t resultSize) {
    try {
        yvector<NArrayRef::TConstArrayRef<float>> floatFeaturesVec(docCount);
        yvector<yvector<TStringBuf>> catFeaturesVec(docCount, yvector<TStringBuf>(catFeaturesSize));
        for (size_t i = 0; i < docCount; ++i) {
            floatFeaturesVec[i] = NArrayRef::TConstArrayRef<float>(floatFeatures[i], floatFeaturesSize);
            for (size_t catFeatureIdx = 0; catFeatureIdx < catFeaturesSize; ++catFeatureIdx) {
                catFeaturesVec[i][catFeatureIdx] = catFeatures[i][catFeatureIdx];
            }
        }
        CALCER(calcer)->Calc(floatFeaturesVec, catFeaturesVec, NArrayRef::TArrayRef<double>(result, resultSize));
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

EXPORT bool CalcModelPredictionWithHashedCatFeatures(ModelCalcerHandle* calcer, size_t docCount,
                                                     const float** floatFeatures, size_t floatFeaturesSize,
                                                     const int** catFeatures, size_t catFeaturesSize,
                                                     double* result, size_t resultSize) {
    try {
        yvector<NArrayRef::TConstArrayRef<float>> floatFeaturesVec(docCount);
        yvector<NArrayRef::TConstArrayRef<int>> catFeaturesVec(docCount, yvector<int>(catFeaturesSize));
        for (size_t i = 0; i < docCount; ++i) {
            floatFeaturesVec[i] = NArrayRef::TConstArrayRef<float>(floatFeatures[i], floatFeaturesSize);
            catFeaturesVec[i] = NArrayRef::TConstArrayRef<int>(catFeatures[i], catFeaturesSize);
        }
        CALCER(calcer)->Calc(floatFeaturesVec, catFeaturesVec, NArrayRef::TArrayRef<double>(result, resultSize));
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

EXPORT int GetStringCatFeatureHash(const char* data, size_t size) {
    return CalcCatFeatureHash(TStringBuf(data, size));
}

EXPORT int GetIntegerCatFeatureHash(long long val) {
    TStringBuilder valStr;
    valStr << val;
    return CalcCatFeatureHash(valStr);
}

EXPORT size_t GetFloatFeaturesCount(ModelCalcerHandle* calcer) {
    return CALCER(calcer)->GetFloatFeaturesUsed();
}

EXPORT size_t GetCatFeaturesCount(ModelCalcerHandle* calcer) {
    return CALCER(calcer)->GetCatFeaturesUsed();
}

}
