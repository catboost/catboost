#include "model_calcer_wrapper.h"

#include <catboost/libs/model/model.h>

#include <util/generic/singleton.h>
#include <util/stream/file.h>
#include <util/string/builder.h>

#define CALCER(x) ((TFullModel*)(x))

struct TErrorMessageHolder {
    TString Message;
};

extern "C" {
EXPORT ModelCalcerHandle* ModelCalcerCreate() {
    try {
        return new TFullModel;
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
        *CALCER(calcer) = ReadModel(filename);
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }

    return true;
}

EXPORT bool CalcModelPredictionFlat(ModelCalcerHandle* calcer, size_t docCount, const float** floatFeatures, size_t floatFeaturesSize, double* result, size_t resultSize) {
    try {
        TVector<TConstArrayRef<float>> featuresVec(docCount);
        for (size_t i = 0; i < docCount; ++i) {
            featuresVec[i] = TConstArrayRef<float>(floatFeatures[i], floatFeaturesSize);
        }
        CALCER(calcer)->CalcFlat(featuresVec, TArrayRef<double>(result, resultSize));
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

EXPORT bool CalcModelPrediction(ModelCalcerHandle* calcer, size_t docCount, const float** floatFeatures, size_t floatFeaturesSize, const char*** catFeatures, size_t catFeaturesSize, double* result, size_t resultSize) {
    try {
        TVector<TConstArrayRef<float>> floatFeaturesVec(docCount);
        TVector<TVector<TStringBuf>> catFeaturesVec(docCount, TVector<TStringBuf>(catFeaturesSize));
        for (size_t i = 0; i < docCount; ++i) {
            floatFeaturesVec[i] = TConstArrayRef<float>(floatFeatures[i], floatFeaturesSize);
            for (size_t catFeatureIdx = 0; catFeatureIdx < catFeaturesSize; ++catFeatureIdx) {
                catFeaturesVec[i][catFeatureIdx] = catFeatures[i][catFeatureIdx];
            }
        }
        CALCER(calcer)->Calc(floatFeaturesVec, catFeaturesVec, TArrayRef<double>(result, resultSize));
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
        TVector<TConstArrayRef<float>> floatFeaturesVec(docCount);
        TVector<TConstArrayRef<int>> catFeaturesVec(docCount, TVector<int>(catFeaturesSize));
        for (size_t i = 0; i < docCount; ++i) {
            floatFeaturesVec[i] = TConstArrayRef<float>(floatFeatures[i], floatFeaturesSize);
            catFeaturesVec[i] = TConstArrayRef<int>(catFeatures[i], catFeaturesSize);
        }
        CALCER(calcer)->Calc(floatFeaturesVec, catFeaturesVec, TArrayRef<double>(result, resultSize));
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
    return CALCER(calcer)->GetNumFloatFeatures();
}

EXPORT size_t GetCatFeaturesCount(ModelCalcerHandle* calcer) {
    return CALCER(calcer)->GetNumCatFeatures();
}

}
