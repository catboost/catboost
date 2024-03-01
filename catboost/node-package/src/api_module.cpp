#include "model.h"

#include <napi.h>

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set(Napi::String::New(env, "Model"), NNodeCatBoost::TModel::GetClass(env));
    return exports;
}

NODE_API_MODULE(catboost, Init)
