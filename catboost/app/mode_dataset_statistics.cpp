#include "modes.h"

#include <catboost/private/libs/app_helpers/mode_dataset_statistics_helpers.h>

namespace {
    class TOpenSourceModeDatasetStatisticsImplementation : public NCB::IModeDatasetStatisticsImplementation {
        int mode_dataset_statistics(int argc, const char *argv[]) const override {
            TCalculateStatisticsParams params;
            params.ProcessParams(argc, argv);
            CalculateDatasetStaticsSingleHost(params);
            return 0;
        }
    };
}

NCB::TModeDatasetStatisticsImplementationFactory::TRegistrator<TOpenSourceModeDatasetStatisticsImplementation>
    YandexSpecificModeDatasetStatisticsImplementationRegistrator(NCB::EImplementationType::OpenSource);
