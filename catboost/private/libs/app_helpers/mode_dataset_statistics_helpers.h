#pragma once

#include "implementation_type_enum.h"

#include <catboost/libs/data/data_provider.h>
#include "catboost/libs/dataset_statistics/visitors.h"
#include <catboost/private/libs/index_range/index_range.h>
#include <catboost/private/libs/options/dataset_reading_params.h>

#include <library/cpp/getopt/small/last_getopt_opts.h>
#include <library/cpp/object_factory/object_factory.h>

#include <util/generic/algorithm.h>
#include <util/generic/map.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


struct TCalculateStatisticsParams {
    TString OutputPath;
    TString HistogramPath;
    NCatboostOptions::TDatasetReadingParams DatasetReadingParams;
    int ThreadCount = -1; // -1 means undefined, set to CPU core count by default
    bool OnlyGroupStatistics = false;
    bool OnlyLightStatistics = false;
    bool ConvertStringTargets = false;
    size_t BorderCount = 64;
    NCB::TFeatureCustomBorders FeatureLimits;
    ui32 SpotSize = 0;
    ui32 SpotCount = 0;

    void ProcessParams(int argc, const char *argv[], NLastGetopt::TOpts* parserPtr = nullptr);

private:
    void BindParserOpts(NLastGetopt::TOpts& parser);
};

namespace NCB {
    class IModeDatasetStatisticsImplementation {
    public:
        virtual int mode_dataset_statistics(int argc, const char **argv) const = 0;

        virtual ~IModeDatasetStatisticsImplementation() = default;
    };

    using TModeDatasetStatisticsImplementationFactory = NObjectFactory::TParametrizedObjectFactory<IModeDatasetStatisticsImplementation, EImplementationType>;

    void CalculateDatasetStatisticsSingleHost(const TCalculateStatisticsParams& calculateStatisticsParams);

    TVector<TIndexRange<ui64>> GetSpots(ui64 datasetSize, ui64 spotSize, ui64 spotCount);
};
