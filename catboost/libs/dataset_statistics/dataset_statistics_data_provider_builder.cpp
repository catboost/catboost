#include "dataset_statistics_data_provider_builder.h"

#include <catboost/libs/helpers/json_helpers.h>

using namespace NJson;

void TDatasetStatisticsProviderBuilder::OutputResult(const TString& outputPath) const {
    TFileOutput output(outputPath);
    WriteJsonWithCatBoostPrecision(this->GetResult(), true, &output);
}

NJson::TJsonValue TDatasetStatisticsProviderBuilder::GetResult() const {
    TJsonValue result;

    result.InsertValue("TargetsStatistics", DatasetStatistics.TargetsStatistics.ToJson());
    result.InsertValue("FeatureStatistics", DatasetStatistics.FeatureStatistics.ToJson());

    result.InsertValue("ObjectCount", ObjectCount);

    return result;
}
