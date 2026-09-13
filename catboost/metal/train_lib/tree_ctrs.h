#pragma once

#include "categorical.h"

#include <util/generic/ptr.h>
#include <util/stream/fwd.h>

namespace NCB {
    // Append this batch after the existing feature bank. CandidateFeatures are
    // local to the batch; ActiveFeatures are absolute runtime feature IDs and
    // describe every currently eligible dynamic feature (not just this batch).
    // Static numeric, one-hot and simple CTR features remain caller-owned.
    struct TMetalTreeCtrBatch {
        ui32 FirstFeature = 0;
        TVector<TVector<ui8>> PermutationBins;
        TVector<ui32> CandidateFeatures;
        TVector<ui32> CandidateBins;
        TVector<ui8> CandidateTypes;
        TVector<ui32> CtrUniqueValues;
        // One flag for each newly appended column. Registered grids include
        // selected winners and CUDA's eagerly cached category-only tensors.
        TVector<ui8> RegisteredCtrFlags;
        TVector<ui32> ActiveFeatures;
        ui32 BinsPerFeature = 1;
        CBMCtrStats Stats = {};

        ui32 GetFeatureCount() const { return CtrUniqueValues.size(); }
    };

    // CUDA FeatureParallel tree-dependent tensor scheduling, GPU projection
    // hashing/grouping and exclusive CTR histories. Explicit history orders
    // are source-row permutations; an empty collection means identity/P1.
    // The adapter owns boosting folds, search RNG and the begin/grow/finish
    // runtime state machine. This helper never changes targets or cursors.
    class TMetalTreeCtrFeatures {
    public:
        TMetalTreeCtrFeatures(
            const TTrainingDataProvider& data,
            const NCatboostOptions::TCatBoostOptions& options,
            NPar::ILocalExecutor* executor,
            ui32 firstFeature,
            ui32 learnAndFirstEvalRows,
            TIntrusivePtr<TStaticCtrProvider> provider = {},
            TConstArrayRef<TVector<ui32>> historyOrders = {});
        ~TMetalTreeCtrFeatures();

        // Start every tree with all dynamic features inactive. Prior IDs,
        // history-specific grid variants and inference tables persist; only
        // selected/eagerly registered grids are shared across search histories.
        void BeginTree();
        TMetalTreeCtrBatch AddSplit(const TModelSplit& split, ui32 borderPermutation = 0);
        // Call for every selected runtime feature, including the last depth.
        // Static feature IDs below firstFeature are ignored. The exact CTR
        // config (including binarization ID) becomes bound to this grid/ID.
        void MarkSelected(ui32 absoluteFeature);

        const TModelSplit& GetSplit(ui32 absoluteFeature, ui32 bin) const;
        ui32 GetFeatureCount() const;
        ui32 GetPermutationCount() const;
        // Absolute IDs of registered configurations, including inactive ones;
        // transient unused history variants are excluded from this set.
        TVector<ui32> GetRegisteredFeatures() const;
        TIntrusivePtr<TStaticCtrProvider> GetCtrProvider() const;

        // Save after a completed tree. Restore into a fresh helper and append
        // its returned banks before restoring optimizer state; saved grids and
        // descriptor order are reused exactly. The caller must validate its
        // complete data/options fingerprint before Restore. No in-progress
        // tree is restored.
        void Save(IOutputStream* output) const;
        TMetalTreeCtrBatch Restore(IInputStream* input);

    private:
        class TImpl;
        THolder<TImpl> Impl;
    };
}
