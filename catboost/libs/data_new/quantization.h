#pragma once

#include "data_provider.h"
#include "exclusive_feature_bundling.h"
#include "quantized_features_info.h"

#include <catboost/libs/helpers/restorable_rng.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/ylimits.h>
#include <util/system/types.h>

#include <functional>


namespace NCB {

    struct TQuantizationOptions {
        bool CpuCompatibleFormat = true;
        bool GpuCompatibleFormat = true;
        ui64 CpuRamLimit = Max<ui64>();
        ui32 MaxSubsetSizeForSlowBuildBordersAlgorithms = 200000;
        bool BundleExclusiveFeaturesForCpu = true;
        TExclusiveFeaturesBundlingOptions ExclusiveFeaturesBundlingOptions{};
        bool PackBinaryFeaturesForCpu = true;
        bool AllowWriteFiles = true;

        // TODO(akhropov): remove after checking global tests consistency
        bool CpuCompatibilityShuffleOverFullData = true;
    };

    /* arguments are idx, srcIdx from rawDataSubsetIndexing
     * each function returns quantized bin
     */
    using TGetBinFunction = std::function<ui32(size_t, size_t)>;

    TGetBinFunction GetQuantizedFloatFeatureFunction(
        const TRawObjectsData& rawObjectsData,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        TFloatFeatureIdx floatFeatureIdx
    );

    TGetBinFunction GetQuantizedCatFeatureFunction(
        const TRawObjectsData& rawObjectsData,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        TCatFeatureIdx catFeatureIdx
    );


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
        const TMap<ui32, NCatboostOptions::TBinarizationOptions> perFloatFeatureBinarization,
        bool floatFeaturesAllowNansInTestOnly,
        TConstArrayRef<ui32> ignoredFeatures,
        TRawDataProviders rawDataProviders,
        TRestorableFastRng64* rand,
        NPar::TLocalExecutor* localExecutor
    );

}
