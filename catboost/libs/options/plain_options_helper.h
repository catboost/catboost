#pragma once

namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    void PlainJsonToOptions(const NJson::TJsonValue& plainOptions, NJson::TJsonValue* catboostOptions,  NJson::TJsonValue* outputOptions);
    void OptionsToPlainJson(const NJson::TJsonValue& catboostOptions, const NJson::TJsonValue& outputOptions, NJson::TJsonValue* plainOptions);
}
