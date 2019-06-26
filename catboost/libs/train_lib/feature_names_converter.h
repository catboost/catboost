#pragma once

#include <catboost/libs/options/load_options.h>
#include <catboost/libs/data_new/meta_info.h>
#include <library/json/json_reader.h>

void ConvertIgnoredFeaturesFromStringToIndices(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* catBoostJsonOptions);
void ConvertIgnoredFeaturesFromStringToIndices(const NCB::TDataMetaInfo& metaInfo, NJson::TJsonValue* catBoostJsonOptions);
