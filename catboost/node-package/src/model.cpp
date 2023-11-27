#include "model.h"

#include "api_helpers.h"

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
    // The C++ object is considered to be successfully created and will be destoryed by Node runtime
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
    if (!NHelper::Check(env, info.Length() >= 2, "Wrong number of arguments - expected 2") ||
        !NHelper::CheckIsMatrix(
            env,
            info[0],
            NHelper::NAT_NUMBER,
            "Expected the first argument to be a matrix of floats - "
        ) ||
        !NHelper::Check(env, this->ModelLoaded, "Trying to predict from the empty model"))
    {
        return env.Undefined();
    }

    const Napi::Array floatFeatures = info[0].As<Napi::Array>();
    const uint32_t sampleCount = floatFeatures.Length();
    if (sampleCount == 0) {
        return Napi::Array::New(env);
    }

    const uint32_t floatFeaturesSize = floatFeatures[0u].As<Napi::Array>().Length();

    std::vector<float> floatFeatureValues;
    floatFeatureValues.reserve(floatFeaturesSize * sampleCount);

    for (uint32_t i = 0; i < sampleCount; ++i) {
        const Napi::Array row = floatFeatures[i].As<Napi::Array>();
        for (uint32_t j = 0; j < floatFeaturesSize; ++j) {
            floatFeatureValues.push_back(row[j].As<Napi::Number>().FloatValue());
        }
    }

    if (!NHelper::CheckIsMatrix(
        env,
        info[1],
        NHelper::NAT_NUMBER_OR_STRING, "Expected second argument to be a matrix of strings or numbers - "))
    {
        return env.Undefined();
    }
    const Napi::Array catFeatures = info[1].As<Napi::Array>();

    if (!NHelper::Check(
            env,
            catFeatures.Length() == sampleCount,
            "Expected the number of samples to be the same for both float and categorial features"
        ))
    {
        return env.Undefined();
    }
    const Napi::Array catRow = catFeatures[0u].As<Napi::Array>();
    if (catRow.Length() == 0 || catRow[0u].IsNumber()) {
        return CalcPredictionHash(env, sampleCount, floatFeatureValues, catFeatures);
    }
    return CalcPredictionString(env, sampleCount, floatFeatureValues, catFeatures);
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


Napi::Array TModel::CalcPredictionHash(
    Napi::Env env,
    const uint32_t sampleCount,
    const std::vector<float>& floatFeatures,
    const Napi::Array& catFeatures
) {
    const uint32_t catFeaturesSize = catFeatures[0u].As<Napi::Array>().Length();
    const uint32_t floatFeaturesSize = floatFeatures.size() / sampleCount;

    std::vector<int> catHashValues;
    catHashValues.reserve(catFeaturesSize * sampleCount);

    for (uint32_t i = 0; i < sampleCount; ++i) {
        const Napi::Array row = catFeatures[i].As<Napi::Array>();
        for (uint32_t j = 0; j < catFeaturesSize; ++j) {
            catHashValues.push_back(row[j].As<Napi::Number>().Int32Value());
        }
    }
    const auto predictionDimensions = ::GetPredictionDimensionsCount(this->Handle);
    std::vector<double> resultValues;
    resultValues.resize(sampleCount * predictionDimensions);

    std::vector<const float*> floatPtrs = CollectMatrixRowPointers<float>(floatFeatures, floatFeaturesSize);
    std::vector<const int*> catPtrs = CollectMatrixRowPointers<int>(catHashValues, catFeaturesSize);
    NHelper::CheckStatus(
        env,
        CalcModelPredictionWithHashedCatFeatures(
            this->Handle,
            sampleCount,
            floatPtrs.data(), floatFeaturesSize,
            catPtrs.data(), catFeaturesSize,
            resultValues.data(), resultValues.size()
        )
    );

    return NHelper::ConvertToArray(env, resultValues);
}

Napi::Array TModel::CalcPredictionString(
    Napi::Env env,
    const uint32_t sampleCount,
    const std::vector<float>& floatFeatures,
    const Napi::Array& catFeatures
) {
    const uint32_t catFeaturesSize = catFeatures[0u].As<Napi::Array>().Length();
    const uint32_t floatFeaturesSize = floatFeatures.size() / sampleCount;

    std::vector<std::string> catStrings;
    std::vector<const char*> catStringValues;
    catStrings.reserve(catFeaturesSize * sampleCount);
    catStringValues.reserve(catFeaturesSize * sampleCount);

    for (uint32_t i = 0; i < sampleCount; ++i) {
        const Napi::Array row = catFeatures[i].As<Napi::Array>();
        for (uint32_t j = 0; j < catFeaturesSize; ++j) {
            catStrings.push_back(row[j].As<Napi::String>().Utf8Value());
            catStringValues.push_back(catStrings.back().c_str());
        }
    }
    const auto predictionDimensions = ::GetPredictionDimensionsCount(this->Handle);
    std::vector<double> resultValues;
    resultValues.resize(sampleCount * predictionDimensions);

    std::vector<const float*> floatPtrs = CollectMatrixRowPointers<float>(floatFeatures, floatFeaturesSize);
    std::vector<const char**> catPtrs = CollectMatrixRowPointers<const char*, const char**>(
        catStringValues,
        catFeaturesSize
    );

    if (!NHelper::CheckStatus(
            env,
            CalcModelPrediction(
                this->Handle,
                sampleCount,
                floatPtrs.data(), floatFeaturesSize,
                catPtrs.data(), catFeaturesSize,
                resultValues.data(), resultValues.size()
            )
        ))
    {
        return Napi::Array::New(env);
    }

    return NHelper::ConvertToArray(env, resultValues);
}

}
