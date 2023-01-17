#pragma once

#include "dataset_statistics_data_provider_builder.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/private/libs/options/dataset_reading_params.h>

#include <util/system/info.h>

using namespace NCB;

struct TCalculateStatisticsParams {
    TString OutputPath;
    NCatboostOptions::TDatasetReadingParams DatasetReadingParams;
    int ThreadCount = NSystemInfo::CachedNumberOfCpus();

    void BindParserOpts(NLastGetopt::TOpts& parser);

    void ProcessParams(int argc, const char* argv[]);
};

void CalculateDatasetStatics(
    const TCalculateStatisticsParams& calculateStatisticsParams,
    const TString& outputPath);
