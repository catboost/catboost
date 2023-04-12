#pragma once

// N-API (node-addon-api)
#include <napi.h>
// Catboost C API
#include <c_api.h>

#include <util/generic/vector.h>

namespace NNodeCatBoost {

// Wrapper over Calcer handle.
class TModel: public Napi::ObjectWrap<TModel> {
public:
    explicit TModel(const Napi::CallbackInfo&);
    virtual ~TModel();

    static Napi::Function GetClass(Napi::Env);

    void LoadFullFromFile(const Napi::CallbackInfo& info);

    // Set model predictions postprocessing type.
    void SetPredictionType(const Napi::CallbackInfo& info);

    // Calculate prediction for matrices of numeric and categorial features.
    Napi::Value CalcPrediction(const Napi::CallbackInfo& info);

    // Enable GPU evaluation on the specified deivce.
    void EvaluateOnGPU(const Napi::CallbackInfo& info);

    // Model parameter getters.
    Napi::Value GetModelFloatFeaturesCount(const Napi::CallbackInfo& info);
    Napi::Value GetModelCatFeaturesCount(const Napi::CallbackInfo& info);
    Napi::Value GetModelTreeCount(const Napi::CallbackInfo& info);
    Napi::Value GetModelDimensionsCount(const Napi::CallbackInfo& info);
    Napi::Value GetPredictionDimensionsCount(const Napi::CallbackInfo& info);

private:
    ModelCalcerHandle* Handle = nullptr;
    bool ModelLoaded = false;

    Napi::Array CalcPredictionHash(Napi::Env env,
                                   const TVector<float>& floatFeatures,
                                   const Napi::Array& catFeatures);
    Napi::Array CalcPredictionString(Napi::Env env,
                                   const TVector<float>& floatFeatures,
                                   const Napi::Array& catFeatures);

    TModel(const TModel&) = delete;
    TModel(TModel&&) = delete;
};

}
