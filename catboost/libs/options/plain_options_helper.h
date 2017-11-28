#pragma once

#include <library/json/json_value.h>

namespace NCatboostOptions {
    void PlainJsonToOptions(const NJson::TJsonValue& plainOptions, NJson::TJsonValue* catboostOptions,  NJson::TJsonValue* outputOptions);
}
