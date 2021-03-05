#include "model.h"

#include <c_api.h>

#include <napi.h>

Napi::String CreateHandleMethod(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    ModelCalcerHandle* handle = ModelCalcerCreate();
    ModelCalcerDelete(handle);

    return Napi::String::New(env, "CatBoost model got successfully created and destroyed");
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set(Napi::String::New(env, "DESCRIPTION"),
		            Napi::String::New(env, "CatBoost is a machine learning method based on gradient boosting "
			                                 "over decision trees."));
    exports.Set(Napi::String::New(env, "CreateHandle"),
		            Napi::Function::New(env, CreateHandleMethod));
    exports.Set(Napi::String::New(env, "Model"),
		            NodeCatBoost::Model::GetClass(env));
    return exports;
}

NODE_API_MODULE(catboost, Init)
