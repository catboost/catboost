#include "dataset_statistics_data_provider_builder.h"

#include <catboost/libs/helpers/json_helpers.h>

using namespace NJson;

void TDatasetStatisticsProviderBuilder::OutputResult(const TString& outputPath) const {
    TFileOutput output(outputPath);
    WriteJsonWithCatBoostPrecision(this->GetDatasetStatistics().ToJson(), true, &output);
}

const TDatasetStatistics& TDatasetStatisticsProviderBuilder::GetDatasetStatistics() const {
    return DatasetStatistics;
}
