#include "c_api.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/model/model.h>

#include <util/generic/singleton.h>
#include <util/stream/file.h>
#include <util/string/builder.h>

#define FULL_MODEL_PTR(x) ((TFullModel*)(x))


struct TErrorMessageHolder {
    TString Message;
};

extern "C" {
CATBOOST_API ModelCalcerHandle* ModelCalcerCreate() {
    try {
        return new TFullModel;
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
    }

    return nullptr;
}

CATBOOST_API const char* GetErrorString() {
    return Singleton<TErrorMessageHolder>()->Message.data();
}

CATBOOST_API void ModelCalcerDelete(ModelCalcerHandle* modelHandle) {
    if (modelHandle != nullptr) {
        delete FULL_MODEL_PTR(modelHandle);
    }
}

CATBOOST_API bool LoadFullModelFromFile(ModelCalcerHandle* modelHandle, const char* filename) {
    try {
        *FULL_MODEL_PTR(modelHandle) = ReadModel(filename);
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }

    return true;
}

CATBOOST_API bool LoadFullModelFromBuffer(ModelCalcerHandle* modelHandle, const void* binaryBuffer, size_t binaryBufferSize) {
    try {
        *FULL_MODEL_PTR(modelHandle) = ReadModel(binaryBuffer, binaryBufferSize);
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }

    return true;
}

CATBOOST_API bool EnableGPUEvaluation(ModelCalcerHandle* modelHandle, int deviceId) {
    try {
        //TODO(kirillovs): fix this after adding set evaluator props interface
        CB_ENSURE(deviceId == 0, "FIXME: Only device 0 is supported for now");
        FULL_MODEL_PTR(modelHandle)->SetEvaluatorType(EFormulaEvaluatorType::GPU);
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

CATBOOST_API bool CalcModelPredictionFlat(ModelCalcerHandle* modelHandle, size_t docCount, const float** floatFeatures, size_t floatFeaturesSize, double* result, size_t resultSize) {
    try {
        if (docCount == 1) {
            FULL_MODEL_PTR(modelHandle)->CalcFlatSingle(TConstArrayRef<float>(*floatFeatures, floatFeaturesSize), TArrayRef<double>(result, resultSize));
        } else {
            TVector<TConstArrayRef<float>> featuresVec(docCount);
            for (size_t i = 0; i < docCount; ++i) {
                featuresVec[i] = TConstArrayRef<float>(floatFeatures[i], floatFeaturesSize);
            }
            FULL_MODEL_PTR(modelHandle)->CalcFlat(featuresVec, TArrayRef<double>(result, resultSize));
        }
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

CATBOOST_API bool CalcModelPrediction(
        ModelCalcerHandle* modelHandle,
        size_t docCount,
        const float** floatFeatures, size_t floatFeaturesSize,
        const char*** catFeatures, size_t catFeaturesSize,
        double* result, size_t resultSize) {
    try {
        TVector<TConstArrayRef<float>> floatFeaturesVec(docCount);
        TVector<TVector<TStringBuf>> catFeaturesVec(docCount, TVector<TStringBuf>(catFeaturesSize));
        for (size_t i = 0; i < docCount; ++i) {
            floatFeaturesVec[i] = TConstArrayRef<float>(floatFeatures[i], floatFeaturesSize);
            for (size_t catFeatureIdx = 0; catFeatureIdx < catFeaturesSize; ++catFeatureIdx) {
                catFeaturesVec[i][catFeatureIdx] = catFeatures[i][catFeatureIdx];
            }
        }
        FULL_MODEL_PTR(modelHandle)->Calc(floatFeaturesVec, catFeaturesVec, TArrayRef<double>(result, resultSize));
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

CATBOOST_API bool CalcModelPredictionText(
        ModelCalcerHandle* modelHandle,
        size_t docCount,
        const float** floatFeatures, size_t floatFeaturesSize,
        const char*** catFeatures, size_t catFeaturesSize,
        const char*** textFeatures, size_t textFeaturesSize,
        double* result, size_t resultSize) {
    try {
        TVector<TConstArrayRef<float>> floatFeaturesVec(docCount);
        TVector<TVector<TStringBuf>> catFeaturesVec(docCount, TVector<TStringBuf>(catFeaturesSize));
        TVector<TVector<TStringBuf>> textFeaturesVec(docCount, TVector<TStringBuf>(textFeaturesSize));
        for (size_t i = 0; i < docCount; ++i) {
            floatFeaturesVec[i] = TConstArrayRef<float>(floatFeatures[i], floatFeaturesSize);
            for (size_t catFeatureIdx = 0; catFeatureIdx < catFeaturesSize; ++catFeatureIdx) {
                catFeaturesVec[i][catFeatureIdx] = catFeatures[i][catFeatureIdx];
            }
            for (size_t textFeatureIdx = 0; textFeatureIdx < textFeaturesSize; ++textFeatureIdx) {
                textFeaturesVec[i][textFeatureIdx] = textFeatures[i][textFeatureIdx];
            }
        }
        FULL_MODEL_PTR(modelHandle)->Calc(floatFeaturesVec, catFeaturesVec, textFeaturesVec, TArrayRef<double>(result, resultSize));
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

CATBOOST_API bool CalcModelPredictionSingle(
        ModelCalcerHandle* modelHandle,
        const float* floatFeatures, size_t floatFeaturesSize,
        const char** catFeatures, size_t catFeaturesSize,
        double* result, size_t resultSize) {
    try {
        TVector<TConstArrayRef<float>> floatFeaturesVec(1);
        TVector<TVector<TStringBuf>> catFeaturesVec(1, TVector<TStringBuf>(catFeaturesSize));
        floatFeaturesVec[0] = TConstArrayRef<float>(floatFeatures, floatFeaturesSize);
        for (size_t catFeatureIdx = 0; catFeatureIdx < catFeaturesSize; ++catFeatureIdx) {
            catFeaturesVec[0][catFeatureIdx] = catFeatures[catFeatureIdx];
        }
        FULL_MODEL_PTR(modelHandle)->Calc(floatFeaturesVec, catFeaturesVec, TArrayRef<double>(result, resultSize));
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

CATBOOST_API bool CalcModelPredictionWithHashedCatFeatures(ModelCalcerHandle* modelHandle, size_t docCount,
                                                     const float** floatFeatures, size_t floatFeaturesSize,
                                                     const int** catFeatures, size_t catFeaturesSize,
                                                     double* result, size_t resultSize) {
    try {
        TVector<TConstArrayRef<float>> floatFeaturesVec(docCount);
        TVector<TConstArrayRef<int>> catFeaturesVec(docCount);
        for (size_t i = 0; i < docCount; ++i) {
            floatFeaturesVec[i] = TConstArrayRef<float>(floatFeatures[i], floatFeaturesSize);
            catFeaturesVec[i] = TConstArrayRef<int>(catFeatures[i], catFeaturesSize);
        }
        FULL_MODEL_PTR(modelHandle)->Calc(floatFeaturesVec, catFeaturesVec, TArrayRef<double>(result, resultSize));
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

CATBOOST_API int GetStringCatFeatureHash(const char* data, size_t size) {
    return CalcCatFeatureHash(TStringBuf(data, size));
}

CATBOOST_API int GetIntegerCatFeatureHash(long long val) {
    TStringBuilder valStr;
    valStr << val;
    return CalcCatFeatureHash(valStr);
}

CATBOOST_API size_t GetFloatFeaturesCount(ModelCalcerHandle* modelHandle) {
    return FULL_MODEL_PTR(modelHandle)->GetNumFloatFeatures();
}

CATBOOST_API size_t GetCatFeaturesCount(ModelCalcerHandle* modelHandle) {
    return FULL_MODEL_PTR(modelHandle)->GetNumCatFeatures();
}

CATBOOST_API size_t GetTreeCount(ModelCalcerHandle* modelHandle) {
    return FULL_MODEL_PTR(modelHandle)->GetTreeCount();
}

CATBOOST_API size_t GetDimensionsCount(ModelCalcerHandle* modelHandle) {
    return FULL_MODEL_PTR(modelHandle)->GetDimensionsCount();
}

CATBOOST_API bool CheckModelMetadataHasKey(ModelCalcerHandle* modelHandle, const char* keyPtr, size_t keySize) {
    return FULL_MODEL_PTR(modelHandle)->ModelInfo.contains(TStringBuf(keyPtr, keySize));
}

CATBOOST_API size_t GetModelInfoValueSize(ModelCalcerHandle* modelHandle, const char* keyPtr, size_t keySize) {
    TStringBuf key(keyPtr, keySize);
    if (!FULL_MODEL_PTR(modelHandle)->ModelInfo.contains(key)) {
        return 0;
    }
    return FULL_MODEL_PTR(modelHandle)->ModelInfo.at(key).size();
}

CATBOOST_API const char* GetModelInfoValue(ModelCalcerHandle* modelHandle, const char* keyPtr, size_t keySize) {
    TStringBuf key(keyPtr, keySize);
    if (!FULL_MODEL_PTR(modelHandle)->ModelInfo.contains(key)) {
        return nullptr;
    }
    return FULL_MODEL_PTR(modelHandle)->ModelInfo.at(key).c_str();
}

}
