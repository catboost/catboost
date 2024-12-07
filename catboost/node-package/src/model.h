#pragma once

// N-API (node-addon-api)
#include <napi.h>
// Catboost C API
#include <catboost/libs/model_interface/c_api.h>


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

    // Calculate prediction for matrices of features.
    Napi::Value CalcPrediction(const Napi::CallbackInfo& info);

    // Enable GPU evaluation on the specified deivce.
    void EvaluateOnGPU(const Napi::CallbackInfo& info);

    // Model parameter getters.
    Napi::Value GetModelFloatFeaturesCount(const Napi::CallbackInfo& info);
    Napi::Value GetModelCatFeaturesCount(const Napi::CallbackInfo& info);
    Napi::Value GetModelTextFeaturesCount(const Napi::CallbackInfo& info);
    Napi::Value GetModelEmbeddingFeaturesCount(const Napi::CallbackInfo& info);
    Napi::Value GetModelTreeCount(const Napi::CallbackInfo& info);
    Napi::Value GetModelDimensionsCount(const Napi::CallbackInfo& info);
    Napi::Value GetPredictionDimensionsCount(const Napi::CallbackInfo& info);

private:
    ModelCalcerHandle* Handle = nullptr;
    bool ModelLoaded = false;

    Napi::Array CalcPredictionWithCatFeaturesAsHashes(
        Napi::Env env,
        const uint32_t sampleCount,
        const Napi::Array& floatFeatures,
        const Napi::Value& catFeatures,         // array or undefined
        const Napi::Value& textFeatures,        // array or undefined
        const Napi::Value& embeddingFeatures    // array or undefined
    );
    Napi::Array CalcPredictionWithCatFeaturesAsStrings(
        Napi::Env env,
        const uint32_t sampleCount,
        const Napi::Array& floatFeatures,
        const Napi::Value& catFeatures,         // array or undefined
        const Napi::Value& textFeatures,        // array or undefined
        const Napi::Value& embeddingFeatures    // array or undefined
    );

    TModel(const TModel&) = delete;
    TModel(TModel&&) = delete;
};

}
