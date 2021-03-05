#pragma once

#include <c_api.h>

#include <napi.h>

#include <vector>

namespace NodeCatBoost {

class Model: public Napi::ObjectWrap<Model> {
public:
    explicit Model(const Napi::CallbackInfo&);
    virtual ~Model();
    
    static Napi::Function GetClass(Napi::Env);
    
    void LoadFullFromFile(const Napi::CallbackInfo& info);
    Napi::Value CalcPrediction(const Napi::CallbackInfo& info);

private:
    ModelCalcerHandle* Handle = nullptr;

    Napi::Array calcPredictionHash(Napi::Env env, 
                                   const std::vector<float>& floatFeatures, 
                                   const Napi::Array& catFeatures);
    Napi::Array calcPredictionString(Napi::Env env, 
                                   const std::vector<float>& floatFeatures, 
                                   const Napi::Array& catFeatures);

    Model(const Model&) = delete;
    Model(Model&&) = delete;
};

}
