#include "model.h"

#include "api_helpers.h"


namespace {

template <typename T, typename V = const T*, typename C = const TVector<T>>
TVector<V> CollectMatrixRowPointers(C& matrix, uint32_t rowLength) {
    TVector<V> pointers;
    for (uint32_t i = 0; i < matrix.size(); i += rowLength) {
        pointers.push_back(matrix.data() + i * rowLength);
    }

    return pointers;
}

template <typename T>
Napi::Array ConvertToArray(Napi::Env env, const TVector<T>& values) {
    Napi::Array result = Napi::Array::New(env);
    uint32_t index = 0;
    for (const auto value: values) {
        result[index++] = Napi::Number::New(env, value);
    }

    return result;
}

} 

namespace NNodeCatBoost {

TModel::TModel(const Napi::CallbackInfo& info): Napi::ObjectWrap<TModel>(info) {
    Napi::Env env = info.Env();

    this->Handle = ModelCalcerCreate(); 
    NHelper::CheckNotNullHandle(env, this->Handle);
}

TModel::~TModel() {
    if (this->Handle != nullptr) {
       ModelCalcerDelete(this->Handle); 
    }
}

Napi::Function TModel::GetClass(Napi::Env env) {
    return DefineClass(env, "Model", {
        TModel::InstanceMethod("loadFullFromFile", &TModel::LoadFullFromFile),
        TModel::InstanceMethod("calcPrediction", &TModel::CalcPrediction),
        TModel::InstanceMethod("getFloatFeaturesCount", &TModel::GetModelFloatFeaturesCount),
    });
}


void TModel::LoadFullFromFile(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    NHelper::Check(env, info.Length() >= 1, "Wrong number of arguments");
    NHelper::Check(env, info[0].IsString(), "File name string is required");

    NHelper::CheckNotNullHandle(env, this->Handle);
    const bool status = LoadFullModelFromFile(this->Handle, 
                                              info[0].As<Napi::String>().Utf8Value().c_str());
    NHelper::CheckStatus(env, status);
}

Napi::Value TModel::CalcPrediction(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    NHelper::Check(env, info.Length() >= 2, "Wrong number of arguments - expected 2");
    NHelper::Check(env, NHelper::IsMatrix(info[0], NHelper::NAT_NUMBER),  
        "Expected the first argument to be a matrix of floats");

    const Napi::Array floatFeatures = info[0].As<Napi::Array>();
    const uint32_t docsCount = floatFeatures.Length();
    if (docsCount == 0) {
        return Napi::Array::New(env);
    }

    const uint32_t floatFeaturesSize = floatFeatures[0u].As<Napi::Array>().Length();
    
    TVector<float> floatFeatureValues;
    floatFeatureValues.reserve(floatFeaturesSize * docsCount);

    for (uint32_t i = 0; i < docsCount; ++i) {
        const Napi::Array row = floatFeatures[i].As<Napi::Array>();
        for (uint32_t j = 0; j < floatFeaturesSize; ++j) {
            floatFeatureValues.push_back(row[j].As<Napi::Number>().FloatValue());
        }
    }

    NHelper::Check(env, 
        NHelper::IsMatrix(info[1], NHelper::NAT_NUMBER) || NHelper::IsMatrix(info[1], NHelper::NAT_STRING),
        "Expected second argument to be a matrix of strings or numbers");    
    const Napi::Array catFeatures = info[1].As<Napi::Array>();

    NHelper::Check(env, catFeatures.Length() == docsCount,
        "Expected the number of docs to be the same for both float and categorial features");
    const Napi::Array catRow = catFeatures[0u].As<Napi::Array>();
    if (catRow.Length() == 0 || catRow[0u].IsNumber()) {
        return CalcPredictionHash(env, floatFeatureValues, catFeatures);
    }
    return CalcPredictionString(env, floatFeatureValues, catFeatures);
}

Napi::Value TModel::GetModelFloatFeaturesCount(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    const size_t count = GetFloatFeaturesCount(this->Handle);

    return Napi::Number::New(env, count);
}

Napi::Array TModel::CalcPredictionHash(Napi::Env env, 
                                   const TVector<float>& floatFeatures, 
                                   const Napi::Array& catFeatures) {
    const uint32_t docsCount = catFeatures.Length();
    const uint32_t catFeaturesSize = catFeatures[0u].As<Napi::Array>().Length();
    const uint32_t floatFeaturesSize = floatFeatures.size() / docsCount;

    TVector<int> catHashValues;
    catHashValues.reserve(catFeaturesSize * docsCount);

    for (uint32_t i = 0; i < docsCount; ++i) {
        const Napi::Array row = catFeatures[i].As<Napi::Array>();
        for (uint32_t j = 0; j < catFeaturesSize; ++j) {
            catHashValues.push_back(row[j].As<Napi::Number>().Int32Value());
        }
    }

    TVector<double> resultValues;
    resultValues.resize(docsCount);

    TVector<const float*> floatPtrs = CollectMatrixRowPointers<float>(floatFeatures, floatFeaturesSize);
    TVector<const int*> catPtrs = CollectMatrixRowPointers<int>(catHashValues, catFeaturesSize);
    NHelper::CheckStatus(env, 
        CalcModelPredictionWithHashedCatFeatures(this->Handle, docsCount, 
                        floatPtrs.data(), floatFeaturesSize,
                        catPtrs.data(), catFeaturesSize,
                        resultValues.data(), docsCount));

    return ConvertToArray(env, resultValues);
}

Napi::Array TModel::CalcPredictionString(Napi::Env env, 
                                   const TVector<float>& floatFeatures, 
                                   const Napi::Array& catFeatures) {
    const uint32_t docsCount = catFeatures.Length();
    const uint32_t catFeaturesSize = catFeatures[0u].As<Napi::Array>().Length();
    const uint32_t floatFeaturesSize = floatFeatures.size() / docsCount;

    TVector<std::string> catStrings;
    TVector<const char*> catStringValues;
    catStrings.reserve(catFeaturesSize * docsCount);
    catStringValues.reserve(catFeaturesSize * docsCount);

    for (uint32_t i = 0; i < docsCount; ++i) {
        const Napi::Array row = catFeatures[i].As<Napi::Array>();
        for (uint32_t j = 0; j < catFeaturesSize; ++j) {
            catStrings.push_back(row[j].As<Napi::String>().Utf8Value());
            catStringValues.push_back(catStrings.back().c_str());
        }
    }

    TVector<double> resultValues;
    resultValues.resize(docsCount);

    TVector<const float*> floatPtrs = CollectMatrixRowPointers<float>(floatFeatures, floatFeaturesSize);
    TVector<const char**> catPtrs = CollectMatrixRowPointers<const char*, const char**>(catStringValues, catFeaturesSize);
    NHelper::CheckStatus(env, 
        CalcModelPrediction(this->Handle, docsCount, 
                        floatPtrs.data(), floatFeaturesSize,
                        catPtrs.data(), catFeaturesSize,
                        resultValues.data(), docsCount));
    
    return ConvertToArray(env, resultValues);
}

}
