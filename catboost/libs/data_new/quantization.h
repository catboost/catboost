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
        bool PackBinaryFeaturesForCpu = true;
        bool AllowWriteFiles = true;

        // TODO(akhropov): remove after checking global tests consistency
        bool CpuCompatibilityShuffleOverFullData = true;
    };

    void CalcBordersAndNanMode(
        const TQuantizationOptions& options,
        TRawDataProviderPtr rawDataProvider,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        TRestorableFastRng64* rand,
        NPar::TLocalExecutor* localExecutor
    );


    TQuantizedObjectsDataProviderPtr Quantize(
        const TQuantizationOptions& options,
        TRawObjectsDataProviderPtr rawObjectsDataProvider,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        TRestorableFastRng64* rand,
        NPar::TLocalExecutor* localExecutor
    );


    TQuantizedDataProviderPtr Quantize(
        const TQuantizationOptions& options,
        TRawDataProviderPtr rawDataProvider,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        TRestorableFastRng64* rand,
        NPar::TLocalExecutor* localExecutor
    );

    TQuantizedDataProviders Quantize(
        const TQuantizationOptions& options,
        const NCatboostOptions::TBinarizationOptions floatFeaturesBinarization,
        bool floatFeaturesAllowNansInTestOnly,
        TConstArrayRef<ui32> ignoredFeatures,
        TRawDataProviders rawDataProviders,
        TRestorableFastRng64* rand,
        NPar::TLocalExecutor* localExecutor
    );

}
