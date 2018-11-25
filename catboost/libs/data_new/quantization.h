#pragma once

#include "data_provider.h"
#include "quantized_features_info.h"

#include <catboost/libs/helpers/restorable_rng.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/ylimits.h>


namespace NCB {

    struct TQuantizationOptions {
        bool CpuCompatibleFormat = true;
        bool GpuCompatibleFormat = true;
        ui64 CpuRamLimit = Max<ui64>();
        ui32 MaxSubsetSizeForSlowBuildBordersAlgorithms = 200000;

        // TODO(akhropov): remove after checking global tests consistency
        bool CpuCompatibilityShuffleOverFullData = true;
    };

    TQuantizedDataProviderPtr Quantize(
        const TQuantizationOptions& options,
        TRawDataProviderPtr rawDataProvider,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        TRestorableFastRng64* rand,
        NPar::TLocalExecutor* localExecutor
    );

}
