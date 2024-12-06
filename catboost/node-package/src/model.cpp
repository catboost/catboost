#include "model.h"

#include "api_helpers.h"

#include <vector>


namespace {

// Collect pointers to matrix rows into a vector.
template <typename T, typename V = const T*, typename C = const std::vector<T>>
std::vector<V> CollectMatrixRowPointers(C& matrix, uint32_t rowLength) {
    std::vector<V> pointers;
    for (uint32_t i = 0; i < matrix.size(); i += rowLength) {
        pointers.push_back(matrix.data() + i);
    }

    return pointers;
}

}

namespace NNodeCatBoost {

TModel::TModel(const Napi::CallbackInfo& info): Napi::ObjectWrap<TModel>(info) {
    Napi::Env env = info.Env();

    this->Handle = ModelCalcerCreate();
    NHelper::CheckNotNullHandle(env, this->Handle);

    if (info.Length() == 0) {
        return;
    }

    NHelper::Check(env, info[0].IsString(), "File name argument should be a string");
    const bool status = LoadFullModelFromFile(
        this->Handle,
        info[0].As<Napi::String>().Utf8Value().c_str()
    );
    // Even if it fails, this check schedules NodeJS exception, not C++ one.
    // The C++ object is considered to be successfully created and will be destroyed by Node runtime
    // later as usual.
    NHelper::CheckStatus(env, status);
    if (status) {
        this->ModelLoaded = true;
    }
}

TModel::~TModel() {
    if (this->Handle != nullptr) {
       ModelCalcerDelete(this->Handle);
    }
}

Napi::Function TModel::GetClass(Napi::Env env) {
    return DefineClass(env, "Model", {
        TModel::InstanceMethod("loadModel", &TModel::LoadFullFromFile),
        TModel::InstanceMethod("predict", &TModel::CalcPrediction),
        TModel::InstanceMethod("enableGPUEvaluation", &TModel::EvaluateOnGPU),
        TModel::InstanceMethod("setPredictionType", &TModel::SetPredictionType),
        TModel::InstanceMethod("getFloatFeaturesCount", &TModel::GetModelFloatFeaturesCount),
        TModel::InstanceMethod("getCatFeaturesCount", &TModel::GetModelCatFeaturesCount),
        TModel::InstanceMethod("getTextFeaturesCount", &TModel::GetModelTextFeaturesCount),
        TModel::InstanceMethod("getEmbeddingFeaturesCount", &TModel::GetModelEmbeddingFeaturesCount),
        TModel::InstanceMethod("getTreeCount", &TModel::GetModelTreeCount),
        TModel::InstanceMethod("getDimensionsCount", &TModel::GetModelDimensionsCount),
        TModel::InstanceMethod("getPredictionDimensionsCount", &TModel::GetPredictionDimensionsCount),
    });
}


void TModel::LoadFullFromFile(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!NHelper::Check(env, info.Length() >= 1, "Wrong number of arguments") ||
        !NHelper::Check(env, info[0].IsString(), "File name string is required"))
    {
        return;
    }

    NHelper::CheckNotNullHandle(env, this->Handle);
    const bool status = LoadFullModelFromFile(
        this->Handle,
        info[0].As<Napi::String>().Utf8Value().c_str()
    );
    NHelper::CheckStatus(env, status);
    if (status) {
        this->ModelLoaded = true;
    }
}

// Set model predictions postprocessing type.
void TModel::SetPredictionType(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!NHelper::Check(env, info.Length() >= 1, "Wrong number of arguments") ||
        !NHelper::Check(env, info[0].IsString(), "predictionType argument must have string type"))
    {
        return;
    }

    NHelper::CheckNotNullHandle(env, this->Handle);
    const bool status = SetPredictionTypeString(
        this->Handle,
        info[0].As<Napi::String>().Utf8Value().c_str()
    );
    NHelper::CheckStatus(env, status);
}

Napi::Value TModel::CalcPrediction(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!NHelper::Check(env, this->ModelLoaded, "Trying to predict from the empty model")) {
        return env.Undefined();
    }

    if (!NHelper::Check(env, info.Length() >= 1, "Wrong number of arguments - expected at least 1")) {
        return env.Undefined();
    }


    // Numerical features

    if (!NHelper::CheckIsMatrix(
            env,
            info[0],
            NHelper::NAT_NUMBER,
            "Expected the first argument to be a matrix of floats - "
        ))
    {
        return env.Undefined();
    }

    const Napi::Array floatFeatures = info[0].As<Napi::Array>();
    const uint32_t sampleCount = floatFeatures.Length();
    if (sampleCount == 0) {
        return Napi::Array::New(env);
    }


    // Categorical features
    Napi::Value catFeatures;
    bool catFeaturesAreHashes = false;

    if (info.Length() >= 2) {
        catFeatures = info[1];
        if (!NHelper::CheckIsMatrix(
                env,
                catFeatures,
                NHelper::NAT_NUMBER_OR_STRING,
                "Expected the second argument to be a matrix of strings or numbers - "
            ))
        {
            return env.Undefined();
        }
        const Napi::Array catFeaturesArray = catFeatures.As<Napi::Array>();

        if (!NHelper::Check(
                env,
                catFeaturesArray.Length() == sampleCount,
                "Expected the number of samples to be the same for both float and categorical features"
            ))
        {
            return env.Undefined();
        }
        if (sampleCount) {
            const Napi::Array catRow = catFeaturesArray[0u].As<Napi::Array>();
            if (catRow.Length()) {
                catFeaturesAreHashes = catRow[0u].IsNumber();
            }
        }
    }


    // Text features
    Napi::Value textFeatures;
    if (info.Length() >= 3) {
        textFeatures = info[2];
        if (!NHelper::CheckIsMatrix(
                env,
                textFeatures,
                NHelper::NAT_STRING,
                "Expected the third argument to be a matrix of strings - "
            ))
        {
            return env.Undefined();
        }

        if (!NHelper::Check(
                env,
                textFeatures.As<Napi::Array>().Length() == sampleCount,
                "Expected the number of samples to be the same for both float and text features"
            ))
        {
            return env.Undefined();
        }
    }

    // Embedding features
    Napi::Value embeddingFeatures;
    if (info.Length() == 4) {
        embeddingFeatures = info[3];
        if (!NHelper::CheckIsMatrix(
                env,
                embeddingFeatures,
                NHelper::NAT_ARRAY_OR_NUMBERS,
                "Expected the fourth argument to be a matrix of arrays of numbers - "
            ))
        {
            return env.Undefined();
        }

        if (!NHelper::Check(
                env,
                embeddingFeatures.As<Napi::Array>().Length() == sampleCount,
                "Expected the number of samples to be the same for both float and embedding features"
            ))
        {
            return env.Undefined();
        }
    }


    if (catFeaturesAreHashes) {
        return CalcPredictionWithCatFeaturesAsHashes(
            env,
            sampleCount,
            floatFeatures,
            catFeatures,
            textFeatures,
            embeddingFeatures
        );
    }
    return CalcPredictionWithCatFeaturesAsStrings(
        env,
        sampleCount,
        floatFeatures,
        catFeatures,
        textFeatures,
        embeddingFeatures
    );
}

void TModel::EvaluateOnGPU(const Napi::CallbackInfo& info) {
   Napi::Env env = info.Env();
    if (!NHelper::Check(env, info.Length() >= 1, "Wrong number of arguments - expected 1") ||
        !NHelper::Check(env, info[0].IsNumber(), "Expected the first argument to be a numeric deviceId"))
    {
        return;
    }

    const bool status = EnableGPUEvaluation(this->Handle, info[0].As<Napi::Number>().Int32Value());
    NHelper::CheckStatus(env, status);
    return;
}

Napi::Value TModel::GetModelFloatFeaturesCount(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    const size_t count = GetFloatFeaturesCount(this->Handle);

    return Napi::Number::New(env, count);
}

Napi::Value TModel::GetModelCatFeaturesCount(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    const size_t count = GetCatFeaturesCount(this->Handle);

    return Napi::Number::New(env, count);
}

Napi::Value TModel::GetModelTextFeaturesCount(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    const size_t count = GetTextFeaturesCount(this->Handle);

    return Napi::Number::New(env, count);
}

Napi::Value TModel::GetModelEmbeddingFeaturesCount(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    const size_t count = GetEmbeddingFeaturesCount(this->Handle);

    return Napi::Number::New(env, count);
}

Napi::Value TModel::GetModelTreeCount(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    const size_t count = GetTreeCount(this->Handle);

    return Napi::Number::New(env, count);
}

Napi::Value TModel::GetModelDimensionsCount(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    const size_t count = GetDimensionsCount(this->Handle);

    return Napi::Number::New(env, count);
}

Napi::Value TModel::GetPredictionDimensionsCount(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    const size_t count = ::GetPredictionDimensionsCount(this->Handle);

    return Napi::Number::New(env, count);
}

static void GetNumericFeaturesData(
    const uint32_t sampleCount,
    const Napi::Array& floatFeatures,
    uint32_t* floatFeatureCount,
    std::vector<float>* storage,
    std::vector<const float*>* sampleDataPtrs
) {
    const uint32_t floatFeatureCountLocal = floatFeatures[0u].As<Napi::Array>().Length();
    *floatFeatureCount = floatFeatureCountLocal;

    storage->clear();
    storage->reserve(floatFeatureCountLocal * sampleCount);

    for (uint32_t i = 0; i < sampleCount; ++i) {
        const Napi::Array row = floatFeatures[i].As<Napi::Array>();
        for (uint32_t j = 0; j < floatFeatureCountLocal; ++j) {
            storage->push_back(row[j].As<Napi::Number>().FloatValue());
        }
    }

    *sampleDataPtrs = CollectMatrixRowPointers<float>(*storage, floatFeatureCountLocal);
}

static void GetTextFeaturesData(
    const uint32_t sampleCount,
    const Napi::Value& textFeatures, // array or empty
    uint32_t* textFeatureCount,
    std::vector<std::string>* storage,
    std::vector<const char*>* dataPtrsStorage,
    std::vector<const char**>* sampleDataPtrs
) {
    storage->clear();
    dataPtrsStorage->clear();
    sampleDataPtrs->clear();
    if (textFeatures.IsEmpty()) {
        *textFeatureCount = 0;
    } else {
        const Napi::Array textFeaturesArray = textFeatures.As<Napi::Array>();
        const uint32_t textFeatureCountLocal = textFeaturesArray[0u].As<Napi::Array>().Length();
        *textFeatureCount = textFeatureCountLocal;

        storage->reserve(textFeatureCountLocal * sampleCount);
        dataPtrsStorage->reserve(textFeatureCountLocal * sampleCount);

        for (uint32_t i = 0; i < sampleCount; ++i) {
            const Napi::Array row = textFeaturesArray[i].As<Napi::Array>();
            for (uint32_t j = 0; j < textFeatureCountLocal; ++j) {
                storage->push_back(row[j].As<Napi::String>().Utf8Value());
                dataPtrsStorage->push_back(storage->back().c_str());
            }
        }

        *sampleDataPtrs = CollectMatrixRowPointers<const char*, const char**>(
            *dataPtrsStorage,
            textFeatureCountLocal
        );
    }
}

static bool GetEmbeddingFeaturesData(
    Napi::Env env,
    const uint32_t sampleCount,
    const Napi::Value& embeddingFeatures, // array or empty
    uint32_t* embeddingFeatureCount,
    std::vector<size_t>* embeddingDimensions,
    std::vector<float>* storage,
    std::vector<const float*>* dataPtrsStorage,
    std::vector<const float**>* sampleDataPtrs
) {
    embeddingDimensions->clear();
    storage->clear();
    dataPtrsStorage->clear();
    sampleDataPtrs->clear();
    if (embeddingFeatures.IsEmpty()) {
        *embeddingFeatureCount = 0;
    } else {
        const Napi::Array embeddingsFeaturesArray = embeddingFeatures.As<Napi::Array>();
        const uint32_t embeddingFeatureCountLocal = embeddingsFeaturesArray[0u].As<Napi::Array>().Length();
        *embeddingFeatureCount = embeddingFeatureCountLocal;

        embeddingDimensions->reserve(embeddingFeatureCountLocal);
        // this is a lower bound, final allocation is delayed until the first sample is processed and
        // embedding dimensions become known
        storage->reserve(embeddingFeatureCountLocal * sampleCount);
        dataPtrsStorage->reserve(embeddingFeatureCountLocal * sampleCount);

        size_t perSampleValuesSize = 0;

        for (uint32_t i = 0; i < sampleCount; ++i) {
            const Napi::Array row = embeddingsFeaturesArray[i].As<Napi::Array>();
            for (uint32_t j = 0; j < embeddingFeatureCountLocal; ++j) {
                const Napi::Array embeddingValues = row[j].As<Napi::Array>();
                auto embeddingSize = embeddingValues.Length();
                if (i == 0) {
                    embeddingDimensions->push_back(embeddingSize);
                } else {
                    if (!NHelper::Check(
                            env,
                            (*embeddingDimensions)[j] == embeddingSize,
                            "Embedding values arrays have different lengths"
                        ))
                    {
                        return false;
                    }
                }

                for (uint32_t k = 0; k < embeddingSize; ++k) {
                    storage->push_back(embeddingValues[k].As<Napi::Number>().FloatValue());
                }
                // can't update dataPtrsStorage just yet as it is not reserved to final size
            }
            if (i == 0) {
                perSampleValuesSize = storage->size();
                storage->reserve(perSampleValuesSize * sampleCount);
            }
            const float* dataPtr = storage->data() + perSampleValuesSize * i;
            for (uint32_t j = 0; j < embeddingFeatureCountLocal; ++j) {
                dataPtrsStorage->push_back(dataPtr);
                dataPtr += (*embeddingDimensions)[j];
            }
        }

        *sampleDataPtrs = CollectMatrixRowPointers<const float*, const float**>(
            *dataPtrsStorage,
            embeddingFeatureCountLocal
        );
    }

    return true;
}


Napi::Array TModel::CalcPredictionWithCatFeaturesAsHashes(
    Napi::Env env,
    const uint32_t sampleCount,
    const Napi::Array& floatFeatures,
    const Napi::Value& catFeatures,
    const Napi::Value& textFeatures,
    const Napi::Value& embeddingFeatures
) {
    uint32_t floatFeaturesSize = 0;
    std::vector<float> floatFeaturesStorage;
    std::vector<const float*> floatPtrs;

    GetNumericFeaturesData(sampleCount, floatFeatures, &floatFeaturesSize, &floatFeaturesStorage, &floatPtrs);


    uint32_t catFeaturesSize = 0;
    std::vector<int> catHashValues;
    std::vector<const int*> catPtrs;

    if (!catFeatures.IsEmpty()) {
        const Napi::Array catFeaturesArray = catFeatures.As<Napi::Array>();
        catFeaturesSize = catFeaturesArray[0u].As<Napi::Array>().Length();

        catHashValues.reserve(catFeaturesSize * sampleCount);

        for (uint32_t i = 0; i < sampleCount; ++i) {
            const Napi::Array row = catFeaturesArray[i].As<Napi::Array>();
            for (uint32_t j = 0; j < catFeaturesSize; ++j) {
                catHashValues.push_back(row[j].As<Napi::Number>().Int32Value());
            }
        }

        catPtrs = CollectMatrixRowPointers<int>(catHashValues, catFeaturesSize);
    }


    uint32_t textFeaturesSize = 0;
    std::vector<std::string> textFeaturesStorage;
    std::vector<const char*> textFeaturesDataPtrsStorage;
    std::vector<const char**> textFeaturesSampleDataPtrs;

    GetTextFeaturesData(
        sampleCount,
        textFeatures,
        &textFeaturesSize,
        &textFeaturesStorage,
        &textFeaturesDataPtrsStorage,
        &textFeaturesSampleDataPtrs
    );


    uint32_t embeddingFeaturesSize = 0;
    std::vector<size_t> embeddingDimensions;
    std::vector<float> embeddingFeaturesStorage;
    std::vector<const float*> embeddingFeaturesDataPtrsStorage;
    std::vector<const float**> embeddingFeaturesSampleDataPtrs;

    if (!NHelper::Check(
            env,
            GetEmbeddingFeaturesData(
                env,
                sampleCount,
                embeddingFeatures,
                &embeddingFeaturesSize,
                &embeddingDimensions,
                &embeddingFeaturesStorage,
                &embeddingFeaturesDataPtrsStorage,
                &embeddingFeaturesSampleDataPtrs
            ),
            "Failed to get embedding features data"
        ))
    {
        return Napi::Array::New(env);
    }


    const auto predictionDimensions = ::GetPredictionDimensionsCount(this->Handle);
    std::vector<double> resultValues;
    resultValues.resize(sampleCount * predictionDimensions);

    NHelper::CheckStatus(
        env,
        CalcModelPredictionWithHashedCatFeaturesAndTextAndEmbeddingFeatures(
            this->Handle,
            sampleCount,
            floatPtrs.data(), floatFeaturesSize,
            catPtrs.data(), catFeaturesSize,
            textFeaturesSampleDataPtrs.data(), textFeaturesSize,
            embeddingFeaturesSampleDataPtrs.data(), embeddingDimensions.data(), embeddingFeaturesSize,
            resultValues.data(), resultValues.size()
        )
    );

    return NHelper::ConvertToArray(env, resultValues);
}

Napi::Array TModel::CalcPredictionWithCatFeaturesAsStrings(
    Napi::Env env,
    const uint32_t sampleCount,
    const Napi::Array& floatFeatures,
    const Napi::Value& catFeatures,
    const Napi::Value& textFeatures,
    const Napi::Value& embeddingFeatures
) {
    uint32_t floatFeaturesSize = 0;
    std::vector<float> floatFeaturesStorage;
    std::vector<const float*> floatPtrs;

    GetNumericFeaturesData(sampleCount, floatFeatures, &floatFeaturesSize, &floatFeaturesStorage, &floatPtrs);

    uint32_t catFeaturesSize = 0;
    std::vector<std::string> catStrings;
    std::vector<const char*> catStringValues;
    std::vector<const char**> catPtrs;

    if (!catFeatures.IsEmpty()) {
        const Napi::Array catFeaturesArray = catFeatures.As<Napi::Array>();
        catFeaturesSize = catFeaturesArray[0u].As<Napi::Array>().Length();

        catStrings.reserve(catFeaturesSize * sampleCount);
        catStringValues.reserve(catFeaturesSize * sampleCount);

        for (uint32_t i = 0; i < sampleCount; ++i) {
            const Napi::Array row = catFeaturesArray[i].As<Napi::Array>();
            for (uint32_t j = 0; j < catFeaturesSize; ++j) {
                catStrings.push_back(row[j].As<Napi::String>().Utf8Value());
                catStringValues.push_back(catStrings.back().c_str());
            }
        }
        catPtrs = CollectMatrixRowPointers<const char*, const char**>(
            catStringValues,
            catFeaturesSize
        );
    }

    uint32_t textFeaturesSize = 0;
    std::vector<std::string> textFeaturesStorage;
    std::vector<const char*> textFeaturesDataPtrsStorage;
    std::vector<const char**> textFeaturesSampleDataPtrs;

    GetTextFeaturesData(
        sampleCount,
        textFeatures,
        &textFeaturesSize,
        &textFeaturesStorage,
        &textFeaturesDataPtrsStorage,
        &textFeaturesSampleDataPtrs
    );


    uint32_t embeddingFeaturesSize = 0;
    std::vector<size_t> embeddingDimensions;
    std::vector<float> embeddingFeaturesStorage;
    std::vector<const float*> embeddingFeaturesDataPtrsStorage;
    std::vector<const float**> embeddingFeaturesSampleDataPtrs;

    if (!NHelper::Check(
            env,
            GetEmbeddingFeaturesData(
                env,
                sampleCount,
                embeddingFeatures,
                &embeddingFeaturesSize,
                &embeddingDimensions,
                &embeddingFeaturesStorage,
                &embeddingFeaturesDataPtrsStorage,
                &embeddingFeaturesSampleDataPtrs
            ),
            "Failed to get embedding features data"
        ))
    {
        return Napi::Array::New(env);
    }


    const auto predictionDimensions = ::GetPredictionDimensionsCount(this->Handle);
    std::vector<double> resultValues;
    resultValues.resize(sampleCount * predictionDimensions);

    if (!NHelper::CheckStatus(
            env,
            CalcModelPredictionTextAndEmbeddings(
                this->Handle,
                sampleCount,
                floatPtrs.data(), floatFeaturesSize,
                catPtrs.data(), catFeaturesSize,
                textFeaturesSampleDataPtrs.data(), textFeaturesSize,
                embeddingFeaturesSampleDataPtrs.data(), embeddingDimensions.data(), embeddingFeaturesSize,
                resultValues.data(), resultValues.size()
            )
        ))
    {
        return Napi::Array::New(env);
    }

    return NHelper::ConvertToArray(env, resultValues);
}

}
