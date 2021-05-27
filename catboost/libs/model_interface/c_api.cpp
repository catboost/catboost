#include "c_api.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/helpers/polymorphic_type_containers.h>
#include <catboost/libs/model/model.h>

#include <util/generic/singleton.h>
#include <util/stream/file.h>
#include <util/string/builder.h>

struct TModelHandleContent {
    THolder<TFullModel> FullModel;
    NCB::NModelEvaluation::TConstModelEvaluatorPtr Evaluator;
};

#define MODEL_HANDLE_CONTENT_PTR(x) ((TModelHandleContent*)(x))
#define FULL_MODEL_PTR(x) (MODEL_HANDLE_CONTENT_PTR(x)->FullModel)
#define EVALUATOR_PTR(x) (MODEL_HANDLE_CONTENT_PTR(x)->Evaluator)

#define DATA_WRAPPER_PTR(x) ((TFeaturesDataWrapper*)(x))

struct TErrorMessageHolder {
    TString Message;
};

class TFeaturesDataWrapper {
public:
    TFeaturesDataWrapper(size_t docsCount)
        : DocsCount(docsCount)
    {
    }

    // dim : floatFeaturesSize x docsCount
    void AddFloatFeatures(const float** floatFeatures, size_t floatFeaturesSize) {
        FloatFeatures.emplace_back(floatFeatures, floatFeaturesSize);
    }

    void AddCatFeatures(const char*** catFeatures, size_t catFeaturesSize) {
        CatFeatures.emplace_back(catFeatures, catFeaturesSize);
    }

    void AddTextFeatures(const char*** textFeatures, size_t textFeaturesSize) {
        TextFeatures.emplace_back(textFeatures, textFeaturesSize);
    }

    NCB::TDataProviderPtr BuildDataProvider() {
        size_t floatFeaturesCount = 0;
        size_t catFeaturesCount = 0;
        size_t textFeaturesCount = 0;
        for (auto [_, count] : FloatFeatures) {
            floatFeaturesCount += count;
        }
        for (auto [_, count] : CatFeatures) {
            catFeaturesCount += count;
        }
        for (auto [_, count] : TextFeatures) {
            textFeaturesCount += count;
        }
        TVector<ui32> catFeaturesIndices(catFeaturesCount);
        std::iota(catFeaturesIndices.begin(), catFeaturesIndices.end(), static_cast<ui32>(floatFeaturesCount));
        TVector<ui32> textFeaturesIndices(textFeaturesCount);
        std::iota(textFeaturesIndices.begin(), textFeaturesIndices.end(), static_cast<ui32>(floatFeaturesCount + catFeaturesCount));

        NCB::TDataMetaInfo metaInfo;
        metaInfo.TargetType = NCB::ERawTargetType::Float;
        metaInfo.TargetCount = 1;
        metaInfo.FeaturesLayout = MakeIntrusive<NCB::TFeaturesLayout>(
            (ui32)(floatFeaturesCount + catFeaturesCount + textFeaturesCount),
            catFeaturesIndices,
            textFeaturesIndices,
            TVector<ui32>{},
            TVector<TString>{}
        );
        NCB::TDataProviderClosure dataProviderClosure(
            NCB::EDatasetVisitorType::RawFeaturesOrder,
            NCB::TDataProviderBuilderOptions(),
            &NPar::LocalExecutor()
        );
        auto* visitor = dataProviderClosure.GetVisitor<NCB::IRawFeaturesOrderDataVisitor>();
        CB_ENSURE(visitor);
        visitor->Start(metaInfo, DocsCount, NCB::EObjectsOrder::Undefined, {});
        {
            ui32 addedFloatFeaturesCount = 0;
            for (auto [arr, count] : FloatFeatures) {
                for (size_t i = 0; i < count; ++i, ++arr, ++addedFloatFeaturesCount) {
                    const float* column = *arr;
                    visitor->AddFloatFeature(
                        addedFloatFeaturesCount,
                        MakeIntrusive<NCB::TTypeCastArrayHolder<float, float>>(TVector<float>(column, column + DocsCount))
                    );
                }
            }
        }
        {
            CatFeaturesVec.assign(catFeaturesIndices.size(), TVector<TStringBuf>(DocsCount));
            ui32 addedCatFeaturesCount = 0;
            for (auto [arr, count] : CatFeatures) {
                for (size_t i = 0; i < count; ++i, ++addedCatFeaturesCount) {
                    for (size_t d = 0; d < DocsCount; ++d) {
                        CatFeaturesVec[addedCatFeaturesCount][d] = arr[i][d];
                    }
                    visitor->AddCatFeature(
                        catFeaturesIndices[addedCatFeaturesCount],
                        CatFeaturesVec[addedCatFeaturesCount]
                    );
                }
            }
        }
        {
            TextFeaturesVec.assign(textFeaturesIndices.size(), TVector<TString>(DocsCount));
            ui32 addedTextFeaturesCount = 0;
            for (auto [arr, count] : TextFeatures) {
                for (size_t i = 0; i < count; ++i, ++addedTextFeaturesCount) {
                    for (size_t d = 0; d < DocsCount; ++d) {
                        TextFeaturesVec[addedTextFeaturesCount][d] = arr[i][d];
                    }
                    visitor->AddTextFeature(
                        textFeaturesIndices[addedTextFeaturesCount],
                        TextFeaturesVec[addedTextFeaturesCount]
                    );
                }
            }
        }
        visitor->Finish();
        DataProvider = dataProviderClosure.GetResult();
        return DataProvider;
    }

private:
    TVector<std::pair<const float**, size_t>> FloatFeatures;
    TVector<std::pair<const char***, size_t>> CatFeatures;
    TVector<std::pair<const char***, size_t>> TextFeatures;
    TVector<TVector<TStringBuf>> CatFeaturesVec;
    TVector<TVector<TString>> TextFeaturesVec;
    NCB::TDataProviderPtr DataProvider;
    size_t DocsCount = 0;
};

namespace {
    void GetSpecificClass(int classId, TArrayRef<double> predictions, size_t dim, TArrayRef<double> result) {
        for (size_t docId = 0; docId < result.size(); ++docId) {
            result[docId] = predictions[docId * dim + classId];
        }
    }
}  // namespace

extern "C" {
CATBOOST_API DataWrapperHandle* DataWrapperCreate(size_t docsCount) {
    try {
        return new TFeaturesDataWrapper(docsCount);
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
    }
    return nullptr;
}

CATBOOST_API void DataWrapperDelete(DataWrapperHandle* dataWrapperHandle) {
    if (dataWrapperHandle != nullptr) {
        delete DATA_WRAPPER_PTR(dataWrapperHandle);
    }
}

CATBOOST_API void AddFloatFeatures(DataWrapperHandle* dataWrapperHandle, const float** floatFeatures, size_t floatFeaturesSize) {
    DATA_WRAPPER_PTR(dataWrapperHandle)->AddFloatFeatures(floatFeatures, floatFeaturesSize);
}

CATBOOST_API void AddCatFeatures(DataWrapperHandle* dataWrapperHandle, const char*** catFeatures, size_t catFeaturesSize) {
    DATA_WRAPPER_PTR(dataWrapperHandle)->AddCatFeatures(catFeatures, catFeaturesSize);
}

CATBOOST_API void AddTextFeatures(DataWrapperHandle* dataWrapperHandle, const char*** textFeatures, size_t textFeaturesSize) {
    DATA_WRAPPER_PTR(dataWrapperHandle)->AddTextFeatures(textFeatures, textFeaturesSize);
}

CATBOOST_API DataProviderHandle* BuildDataProvider(DataWrapperHandle* dataWrapperHandle) {
    return DATA_WRAPPER_PTR(dataWrapperHandle)->BuildDataProvider().Get();
}

CATBOOST_API ModelCalcerHandle* ModelCalcerCreate() {
    try {
        auto* fullModel = new TFullModel;
        auto evaluator = fullModel->GetCurrentEvaluator();
        return new TModelHandleContent{.FullModel = fullModel, .Evaluator = std::move(evaluator)};
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
        delete MODEL_HANDLE_CONTENT_PTR(modelHandle);
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
        EVALUATOR_PTR(modelHandle) = FULL_MODEL_PTR(modelHandle)->GetCurrentEvaluator();
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

CATBOOST_API bool SetPredictionType(ModelCalcerHandle* modelHandle, EApiPredictionType predictionType) {
    try {
        FULL_MODEL_PTR(modelHandle)->SetPredictionType(static_cast<NCB::NModelEvaluation::EPredictionType>(predictionType));
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

CATBOOST_API bool CalcModelPredictionWithHashedCatAndTextFeatures(ModelCalcerHandle* modelHandle, size_t docCount,
                                                    const float** floatFeatures, size_t floatFeaturesSize,
                                                    const int** catFeatures, size_t catFeaturesSize,
                                                    const char*** textFeatures, size_t textFeaturesSize,
                                                    double* result, size_t resultSize) {
    try {
        TVector<TConstArrayRef<float>> floatFeaturesVec(docCount);
        TVector<TConstArrayRef<int>> catFeaturesVec(docCount);
        TVector<TVector<TStringBuf>> textFeaturesVec(docCount, TVector<TStringBuf>(textFeaturesSize));
        for (size_t i = 0; i < docCount; ++i) {
            floatFeaturesVec[i] = TConstArrayRef<float>(floatFeatures[i], floatFeaturesSize);
            catFeaturesVec[i] = TConstArrayRef<int>(catFeatures[i], catFeaturesSize);
            for (size_t textFeatureIdx = 0; textFeatureIdx < textFeaturesSize; ++textFeatureIdx) {
                textFeaturesVec[i][textFeatureIdx] = textFeatures[i][textFeatureIdx];
            }
        }
        FULL_MODEL_PTR(modelHandle)->CalcWithHashedCatAndText(floatFeaturesVec, catFeaturesVec, textFeaturesVec, TArrayRef<double>(result, resultSize));
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

CATBOOST_API bool PredictSpecificClassFlat(
        ModelCalcerHandle* modelHandle,
        size_t docCount,
        const float** floatFeatures, size_t floatFeaturesSize,
        int classId,
        double* result, size_t resultSize) {
    try {
        const size_t dim = FULL_MODEL_PTR(modelHandle)->GetDimensionsCount();
        TVector<double> rawResult(docCount * dim);
        if (!CalcModelPredictionFlat(modelHandle, docCount, floatFeatures, floatFeaturesSize, rawResult.data(), rawResult.size())) {
            return false;
        }
        GetSpecificClass(classId, rawResult, dim, TArrayRef<double>(result, resultSize));
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

CATBOOST_API bool PredictSpecificClass(
        ModelCalcerHandle* modelHandle,
        size_t docCount,
        const float** floatFeatures, size_t floatFeaturesSize,
        const char*** catFeatures, size_t catFeaturesSize,
        int classId,
        double* result, size_t resultSize) {
    try {
        const size_t dim = FULL_MODEL_PTR(modelHandle)->GetDimensionsCount();
        TVector<double> rawResult(docCount * dim);
        if (!CalcModelPrediction(
                modelHandle, docCount,
                floatFeatures, floatFeaturesSize,
                catFeatures, catFeaturesSize,
                rawResult.data(), rawResult.size()))
        {
            return false;
        }
        GetSpecificClass(classId, rawResult, dim, TArrayRef<double>(result, resultSize));
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

CATBOOST_API bool PredictSpecificClassText(
        ModelCalcerHandle* modelHandle,
        size_t docCount,
        const float** floatFeatures, size_t floatFeaturesSize,
        const char*** catFeatures, size_t catFeaturesSize,
        const char*** textFeatures, size_t textFeaturesSize,
        int classId,
        double* result, size_t resultSize) {
    try {
        const size_t dim = FULL_MODEL_PTR(modelHandle)->GetDimensionsCount();
        TVector<double> rawResult(docCount * dim);
        if (!CalcModelPredictionText(
                modelHandle, docCount,
                floatFeatures, floatFeaturesSize,
                catFeatures, catFeaturesSize,
                textFeatures, textFeaturesSize,
                rawResult.data(), rawResult.size()))
        {
            return false;
        }
        GetSpecificClass(classId, rawResult, dim, TArrayRef<double>(result, resultSize));
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

CATBOOST_API bool PredictSpecificClassSingle(
        ModelCalcerHandle* modelHandle,
        const float* floatFeatures, size_t floatFeaturesSize,
        const char** catFeatures, size_t catFeaturesSize,
        int classId,
        double* result, size_t resultSize) {
    try {
        const size_t dim = FULL_MODEL_PTR(modelHandle)->GetDimensionsCount();
        TVector<double> rawResult(dim);
        if (!CalcModelPredictionSingle(
                modelHandle,
                floatFeatures, floatFeaturesSize,
                catFeatures, catFeaturesSize,
                rawResult.data(), rawResult.size()))
        {
            return false;
        }
        GetSpecificClass(classId, rawResult, dim, TArrayRef<double>(result, resultSize));
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

CATBOOST_API bool PredictSpecificClassWithHashedCatFeatures(
        ModelCalcerHandle* modelHandle,
        size_t docCount,
        const float** floatFeatures, size_t floatFeaturesSize,
        const int** catFeatures, size_t catFeaturesSize,
        int classId,
        double* result, size_t resultSize) {
    try {
        const size_t dim = FULL_MODEL_PTR(modelHandle)->GetDimensionsCount();
        TVector<double> rawResult(docCount * dim);
        if (!CalcModelPredictionWithHashedCatFeatures(
                modelHandle, docCount,
                floatFeatures, floatFeaturesSize,
                catFeatures, catFeaturesSize,
                rawResult.data(), rawResult.size()))
        {
            return false;
        }
        GetSpecificClass(classId, rawResult, dim, TArrayRef<double>(result, resultSize));
    } catch (...) {
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }
    return true;
}

CATBOOST_API bool PredictSpecificClassWithHashedCatAndTextFeatures(
        ModelCalcerHandle* modelHandle,
        size_t docCount,
        const float** floatFeatures, size_t floatFeaturesSize,
        const int** catFeatures, size_t catFeaturesSize,
        const char*** textFeatures, size_t textFeaturesSize,
        int classId,
        double* result, size_t resultSize) {
    try {
        const size_t dim = FULL_MODEL_PTR(modelHandle)->GetDimensionsCount();
        TVector<double> rawResult(docCount * dim);
        if (!CalcModelPredictionWithHashedCatAndTextFeatures(
                modelHandle, docCount,
                floatFeatures, floatFeaturesSize,
                catFeatures, catFeaturesSize,
                textFeatures, textFeaturesSize,
                rawResult.data(), rawResult.size()))
        {
            return false;
        }
        GetSpecificClass(classId, rawResult, dim, TArrayRef<double>(result, resultSize));
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
