#pragma once

#include "data_provider.h"
#include "exclusive_feature_bundling.h"
#include "quantized_features_info.h"

#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/options/data_processing_options.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/array_ref.h>
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


    /*
     * argument is src indices for up to 64 documents
     * the return value is a bit mask whether the corresponding quantized feature value bins are non-default
     */
    using TGetNonDefaultValuesMask = std::function<ui64(TConstArrayRef<ui32>)>;

    TGetNonDefaultValuesMask GetQuantizedFloatNonDefaultValuesMaskFunction(
        const TRawObjectsData& rawObjectsData,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        TFloatFeatureIdx floatFeatureIdx
    );

    TGetNonDefaultValuesMask GetQuantizedCatNonDefaultValuesMaskFunction(
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
        const NCatboostOptions::TDataProcessingOptions& dataProcessingOptions,
        bool floatFeaturesAllowNansInTestOnly,
        TConstArrayRef<ui32> ignoredFeatures,
        TRawDataProviders rawDataProviders,
        TRestorableFastRng64* rand,
        NPar::TLocalExecutor* localExecutor
    );

}
