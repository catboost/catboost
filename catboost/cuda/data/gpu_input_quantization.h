#pragma once

#include <catboost/libs/data/data_provider.h>

#include <catboost/private/libs/options/catboost_options.h>

class TRestorableFastRng64;

namespace NPar {
    class ILocalExecutor;
}

namespace NCB {

    TQuantizedObjectsDataProviderPtr QuantizeGpuInputData(
        const NCatboostOptions::TCatBoostOptions& params,
        TDataProviderPtr srcData,
        bool isLearnData,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        NPar::ILocalExecutor* localExecutor,
        TRestorableFastRng64* rand
    );

}

