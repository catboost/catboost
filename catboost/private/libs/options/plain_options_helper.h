#pragma once

namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    void PlainJsonToOptions(const NJson::TJsonValue& plainOptions, NJson::TJsonValue* catboostOptions, NJson::TJsonValue* outputOptions, NJson::TJsonValue* featuresSelectOptions = nullptr);
    void ConvertOptionsToPlainJson(const NJson::TJsonValue& catboostOptions, const NJson::TJsonValue& outputOptions, NJson::TJsonValue* plainOptions);
    // TODO(d-kruchinin): change hasTextFeatures when text features are added to the model.
    void CleanPlainJson(bool hasCatFeatures,
                        NJson::TJsonValue* plainOptionsJsonEfficient,
                        bool hasTextFeatures = false,
                        bool hasEmbeddingFeatures = false
    );
}
