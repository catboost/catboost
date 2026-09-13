#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/static_ctr_provider.h>
#include <catboost/metal/native/metal_ctrs.h>
#include <catboost/private/libs/options/catboost_options.h>

namespace NCB {
    // Original CatBoost hashes and dense bins for values present in this Pool.
    // OnAll retains the shared learn/evaluation cardinality used by CUDA.
    struct TMetalCategoryColumn {
        TVector<ui32> Hashes;
        TVector<ui32> Bins;
        ui32 UniqueValuesOnAll = 0;
    };

    TMetalCategoryColumn ReadMetalCategoryColumn(
        const TQuantizedObjectsDataProvider& objects,
        const TCatFeature& feature,
        NPar::ILocalExecutor* executor);

    // Runtime feature indices are local to this result. The trainer appends
    // them after its numeric columns and offsets CandidateFeatures accordingly.
    struct TMetalCategoricalData {
        TVector<TCatFeature> AllCatFeatures;
        TVector<ui8> Bins;
        // Dataset zero is Bins; each additional dataset uses the same feature
        // columns and split grids but its own exclusive CTR history.
        TVector<TVector<ui8>> AdditionalPermutationBins;
        TVector<ui32> CandidateFeatures;
        TVector<ui32> CandidateBins;
        TVector<ui8> CandidateTypes;
        TVector<TVector<TModelSplit>> SplitCandidates;
        // Zero for one-hot/constant columns. CTR columns carry the source
        // category count OnAll; the adapter caps it by learn+first-eval rows
        // before configuring CUDA's dynamic model_size_reg penalty.
        TVector<ui32> CtrUniqueValues;
        ui32 BinsPerFeature = 1;
        TIntrusivePtr<TStaticCtrProvider> CtrProvider;
        CBMCtrStats Stats = {};

        const TModelSplit& GetSplit(ui32 feature, ui32 bin, ui8 type) const;
        ui32 GetPermutationCount() const { return AdditionalPermutationBins.size() + 1; }
        TConstArrayRef<ui8> GetPermutationBins(ui32 permutation) const;
    };

    TMetalCategoricalData PrepareMetalCategoricalData(
        const TTrainingDataProvider& data,
        const NCatboostOptions::TCatBoostOptions& options,
        NPar::ILocalExecutor* executor);

    // Used only by the trainer with independent per-permutation GPU cursors
    // and leaf estimation. Merely switching these bins would be incorrect.
    TMetalCategoricalData PrepareMetalCategoricalPermutations(
        const TTrainingDataProvider& data,
        const NCatboostOptions::TCatBoostOptions& options,
        NPar::ILocalExecutor* executor);

    // Hash actual row values, not the mutable shared perfect-hash dictionary.
    ui32 CalcMetalCategoricalChecksum(const TQuantizedObjectsDataProvider& objects,
                                      NPar::ILocalExecutor* executor);
}
