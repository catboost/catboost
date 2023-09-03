#include "utils.h"

#include "model.h"

#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/options/plain_options_helper.h>

NJson::TJsonValue GetPlainJsonWithAllOptions(const TFullModel& model) {
    NJson::TJsonValue trainOptions = ReadTJsonValue(model.ModelInfo.at("params"));
    NJson::TJsonValue outputOptions = ReadTJsonValue(model.ModelInfo.at("output_options"));
    NJson::TJsonValue plainOptions;
    NCatboostOptions::ConvertOptionsToPlainJson(trainOptions, outputOptions, &plainOptions);
    CB_ENSURE(!plainOptions.GetMapSafe().empty(), "plainOptions should not be empty.");
    NJson::TJsonValue cleanedOptions(plainOptions);
    CB_ENSURE(!cleanedOptions.GetMapSafe().empty(), "problems with copy constructor.");
    bool hasCatFeatures = !model.ModelTrees->GetCatFeatures().empty();
    bool hasTextFeatures = !model.ModelTrees->GetTextFeatures().empty();
    bool hasEmbeddingFeatures = !model.ModelTrees->GetEmbeddingFeatures().empty();
    NCatboostOptions::CleanPlainJson(hasCatFeatures, &cleanedOptions, hasTextFeatures, hasEmbeddingFeatures);
    CB_ENSURE(!cleanedOptions.GetMapSafe().empty(), "cleanedOptions should not be empty.");
    return cleanedOptions;
}
