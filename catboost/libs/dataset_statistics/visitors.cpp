#include "visitors.h"

#include <catboost/libs/helpers/json_helpers.h>

using namespace NJson;

void TDatasetStatisticsFullVisitor::OutputResult(const TString& outputPath) const {
    TFileOutput output(outputPath);
    WriteJsonWithCatBoostPrecision(this->GetDatasetStatistics().ToJson(), true, &output);
}

const TDatasetStatistics& TDatasetStatisticsFullVisitor::GetDatasetStatistics() const {
    return DatasetStatistics;
}

void TDatasetStatisticsOnlyGroupVisitor::OutputResult(const TString& outputPath) const {
    TFileOutput output(outputPath);
    WriteJsonWithCatBoostPrecision(this->GetGroupwiseStats().ToJson(), true, &output);
}

const TGroupwiseStats& TDatasetStatisticsOnlyGroupVisitor::GetGroupwiseStats() const {
    return GroupwiseStats;
}
