#pragma once

#include "implementation_type_enum.h"

#include <catboost/libs/data/data_provider.h>
#include "catboost/libs/dataset_statistics/visitors.h"
#include <catboost/private/libs/options/dataset_reading_params.h>

#include <library/cpp/getopt/small/last_getopt_opts.h>
#include <library/cpp/object_factory/object_factory.h>

#include <util/system/info.h>

using namespace NCB;

struct TCalculateStatisticsParams {
    TString OutputPath;
    NCatboostOptions::TDatasetReadingParams DatasetReadingParams;
    int ThreadCount = NSystemInfo::CachedNumberOfCpus();
    bool OnlyGroupStatistics = false;

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

    void CalculateDatasetStaticsSingleHost(const TCalculateStatisticsParams& calculateStatisticsParams);
};
