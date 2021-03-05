#include "model.h"

#include <vector>

static void check(Napi::Env env, bool condition, const std::string& message) {
    if (!condition) {
        Napi::TypeError::New(env, message)
	    .ThrowAsJavaScriptException();
    }
}

template <typename T>
static void checkNotNull(Napi::Env env, T* ptr, const std::string& message) {
    check(env, ptr != nullptr, message);
}

static void checkNotNullHandle(Napi::Env env, ModelCalcerHandle* handle) {
    return checkNotNull(env, handle, "Internal error - null handle encountered");
}

static void checkStatus(Napi::Env& env, bool status) {
    if (!status) {
        const char* errorMessage = GetErrorString();
        checkNotNull(env, errorMessage, "Internal error - error message expected, but missing");
        Napi::Error::New(env, errorMessage).ThrowAsJavaScriptException();
    }
}

namespace NodeCatBoost {

Model::Model(const Napi::CallbackInfo& info): Napi::ObjectWrap<Model>(info) {
    Napi::Env env = info.Env();

    this->Handle = ModelCalcerCreate(); 
    checkNotNullHandle(env, this->Handle);
}

Model::~Model() {
    if (this->Handle != nullptr) {
       ModelCalcerDelete(this->Handle); 
    }
}

Napi::Function Model::GetClass(Napi::Env env) {
    return DefineClass(env, "Model", {
        Model::InstanceMethod("loadFullFromFile", &Model::LoadFullFromFile),
        Model::InstanceMethod("calcPrediction", &Model::CalcPrediction),
    });
}


void Model::LoadFullFromFile(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    check(env, info.Length() >= 1, "Wrong number of arguments");
    check(env, info[0].IsString(), "File name string is required");

    checkNotNullHandle(env, this->Handle);
    const bool status = LoadFullModelFromFile(this->Handle, 
                                              info[0].As<Napi::String>().Utf8Value().c_str());
    checkStatus(env, status);
}

Napi::Value Model::CalcPrediction(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    check(env, info.Length() >= 2, "Wrong number of arguments");
    check(env, info[0].IsArray(), "Expected first argument to be array of floats");

    const Napi::Array floatFeatures = info[0].As<Napi::Array>();
    const uint32_t floatFeaturesSize = floatFeatures.Length();
    
    std::vector<float> floatFeatureValues;
    floatFeatureValues.reserve(floatFeaturesSize);

    for (uint32_t i = 0; i < floatFeaturesSize; ++i) {
        check(env, floatFeatures[i].IsNumber(), "Expected first argument to be array of floats");
        floatFeatureValues.push_back(floatFeatures[i].As<Napi::Number>().FloatValue());
    }

    check(env, info[0].IsArray(), 
               "Expected first argument to be array of strings or numbers");    
    const Napi::Array catFeatures = info[1].As<Napi::Array>();
    if (catFeatures == 0 || catFeatures[0u].IsNumber()) {
        return calcPredictionHash(env, floatFeatureValues, catFeatures);
    }
    return calcPredictionString(env, floatFeatureValues, catFeatures);
}

Napi::Array Model::calcPredictionHash(Napi::Env env, 
                                   const std::vector<float>& floatFeatures, 
                                   const Napi::Array& catFeatures) {
    std::vector<int> catHashValues;
    catHashValues.reserve(catFeatures.Length());

    for (uint32_t i = 0; i < catFeatures.Length(); ++i) {
        check(env, catFeatures[i].IsNumber(), "Expected first argument to be array of strings or integers");
        catHashValues.push_back(catFeatures[i].As<Napi::Number>().Int32Value());
    }

    double resultValue = 0;
    const float* floatPtr = floatFeatures.data();
    const int* catPtr = catHashValues.data();
    checkStatus(env, 
        CalcModelPredictionWithHashedCatFeatures(this->Handle, 1, 
                        &floatPtr, floatFeatures.size(),
                        &catPtr, catHashValues.size(),
                        &resultValue, 1));
    Napi::Array result = Napi::Array::New(env);
    result[0u] = Napi::Number::New(env, resultValue);
    return result;
}

Napi::Array Model::calcPredictionString(Napi::Env env, 
                                   const std::vector<float>& floatFeatures, 
                                   const Napi::Array& catFeatures) {
    std::vector<std::string> catStrings;
    std::vector<const char*> catStringValues;
    catStrings.reserve(catFeatures.Length());
    catStringValues.reserve(catFeatures.Length());

    for (uint32_t i = 0; i < catFeatures.Length(); ++i) {
        check(env, catFeatures[i].IsString(), "Expected second argument to be array of strings or integers");
        catStrings.push_back(catFeatures[i].As<Napi::String>().Utf8Value());
        catStringValues.push_back(catStrings.back().c_str());
    }

    double resultValue = 0;
    const float* floatPtr = floatFeatures.data();
    const char** catPtr = catStringValues.data();
    checkStatus(env, 
        CalcModelPrediction(this->Handle, 1, 
                        &floatPtr, floatFeatures.size(),
                        &catPtr, catStringValues.size(),
                        &resultValue, 1));
    Napi::Array result = Napi::Array::New(env);
    result[0u] = Napi::Number::New(env, resultValue);
    return result;
}

}
