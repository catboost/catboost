#pragma once

#include "columns.h"
#include "data_provider.h"
#include "exclusive_feature_bundling.h"
#include "feature_grouping.h"
#include "catboost/libs/data/quantized_features_info.h"

#include <catboost/private/libs/index_range/index_range.h>
#include <catboost/libs/helpers/dynamic_iterator.h>
#include <catboost/libs/helpers/sparse_array.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/private/libs/options/data_processing_options.h>
#include <catboost/private/libs/options/catboost_options.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/ylimits.h>
#include <util/system/types.h>

#include <functional>


namespace NPar {
    class ILocalExecutor;
}


namespace NCB {

    using TInitialBorders = TMaybe<TVector<TConstArrayRef<float>>>;

    struct TQuantizationOptions {
        ui64 CpuRamLimit = Max<ui64>();
        ui32 MaxSubsetSizeForBuildBordersAlgorithms = 200000;
        bool BundleExclusiveFeatures = true;
        TExclusiveFeaturesBundlingOptions ExclusiveFeaturesBundlingOptions{};
        bool PackBinaryFeaturesForCpu = true;
        bool GroupFeaturesForCpu = false;
        TFeaturesGroupingOptions FeaturesGroupingOptions{};

        TMaybe<float> DefaultValueFractionToEnableSparseStorage = Nothing();
        ESparseArrayIndexingType SparseArrayIndexingType = ESparseArrayIndexingType::Indices;
    };

    void PrepareQuantizationParameters(
        const NCatboostOptions::TCatBoostOptions& params,
        const TDataMetaInfo& metaInfo,
        const TMaybe<TString>& bordersFile,
        TQuantizationOptions* quantizationOptions,
        TQuantizedFeaturesInfoPtr* quantizedFeaturesInfo
    );

    void PrepareQuantizationParameters(
        NJson::TJsonValue plainJsonParams,
        const TDataMetaInfo& metaInfo,
        const TMaybe<TString>& bordersFile,
        TQuantizationOptions* quantizationOptions,
        TQuantizedFeaturesInfoPtr* quantizedFeaturesInfo
    );

    /*
     * Used for optimization.
     * It is number of times more effective to iterate over dense data in incremental order instead of random
     *   access
     */
    struct TIncrementalDenseIndexing {
        // indices in SrcData for dense features, TFullSubset if there're no dense src data
        TFeaturesArraySubsetIndexing SrcSubsetIndexing;

        // positions in dst data when iterating over dense SrcData in SrcSubsetIndexing order
        TFeaturesArraySubsetIndexing DstIndexing;
    public:
        TIncrementalDenseIndexing(
            const TFeaturesArraySubsetIndexing& srcSubsetIndexing,
            bool hasDenseData,
            NPar::ILocalExecutor* localExecutor
        );
    };


    /*
     * return values dstMasks will contain pairs:
     *  pair.first is 64-documents block index
     *  pair.second is bit mask whether the corresponding quantized feature value bins are non-default
     */

    void GetQuantizedNonDefaultValuesMasks(
        const TFloatValuesHolder& floatValuesHolder,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const TFeaturesArraySubsetIndexing& incrementalIndexing,
        const TFeaturesArraySubsetInvertedIndexing& invertedIncrementalIndexing,
        TVector<std::pair<ui32, ui64>>* dstMasks,
        ui32* nonDefaultCount
    );

    void GetQuantizedNonDefaultValuesMasks(
        const THashedCatValuesHolder& catValuesHolder,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const TFeaturesArraySubsetIndexing& incrementalIndexing,
        const TFeaturesArraySubsetInvertedIndexing& invertedIncrementalIndexing,
        TVector<std::pair<ui32, ui64>>* dstMasks,
        ui32* nonDefaultCount
    );


    void CalcBordersAndNanMode(
        const TQuantizationOptions& options,
        TRawDataProviderPtr rawDataProvider,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        TRestorableFastRng64* rand,
        NPar::ILocalExecutor* localExecutor
    );


    TQuantizedObjectsDataProviderPtr Quantize(
        const TQuantizationOptions& options,
        TRawObjectsDataProviderPtr rawObjectsDataProvider,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        TRestorableFastRng64* rand,
        NPar::ILocalExecutor* localExecutor,
        const TInitialBorders& initialBorders = Nothing()
    );


    TQuantizedDataProviderPtr Quantize(
        const TQuantizationOptions& options,
        TRawDataProviderPtr rawDataProvider,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        TRestorableFastRng64* rand,
        NPar::ILocalExecutor* localExecutor,
        const TInitialBorders& initialBorders = Nothing()
    );

    TQuantizedObjectsDataProviderPtr GetQuantizedObjectsData(
        const NCatboostOptions::TCatBoostOptions& params,
        TDataProviderPtr srcData,
        const TMaybe<TString>& bordersFile,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        NPar::ILocalExecutor* localExecutor,
        TRestorableFastRng64* rand,
        const TInitialBorders& initialBorders = Nothing()
    );

    TQuantizedObjectsDataProviderPtr ConstructQuantizedPoolFromRawPool(
        TDataProviderPtr pool,
        NJson::TJsonValue plainJsonParams,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo
    );

}
