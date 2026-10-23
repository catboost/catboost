#pragma once

#include "tree_ctrs.h"
#include "snapshot.h"

#include <catboost/metal/native/metal_ordered_trainer.h>

#include <util/stream/str.h>

namespace NCB {
    // Completed snapshots still rebuild/export their trees without opening a
    // runtime. Validate against the regenerated registry before that branch,
    // so inconsistent metadata cannot be accepted only because training ends.
    inline void ValidateMetalTreeCtrSnapshot(const TMetalTreeCtrBatch& restored,
            TConstArrayRef<ui32> staticCounts, const TMetalSnapshot& snapshot, bool ordered,
            TConstArrayRef<float> staticWeights = {}) {
        snapshot.ValidateTreeCtrMetadata();
        CB_ENSURE(staticWeights.empty() || staticWeights.size() == staticCounts.size(),
            "Metal static feature weights have inconsistent dimensions");
        const ui64 totalFeatures = ui64(staticCounts.size()) + restored.GetFeatureCount();
        CB_ENSURE(restored.FirstFeature == staticCounts.size() &&
                  restored.RegisteredCtrFlags.size() == restored.GetFeatureCount() &&
                  snapshot.TreeCtrCounts.size() == totalFeatures &&
                  (ordered || snapshot.UsedFeatures == snapshot.TreeCtrUsed),
                  "Saved Metal tree CTR feature-use payloads or dimensions disagree");
        for (ui32 feature = 0; feature < totalFeatures; ++feature) {
            const bool dynamic = feature >= staticCounts.size();
            const ui32 index = dynamic ? feature - staticCounts.size() : feature;
            const ui32 count = dynamic ? restored.CtrUniqueValues[index] : staticCounts[index];
            if (dynamic) CB_ENSURE(restored.RegisteredCtrFlags[index] <= 1,
                                  "Restored Metal tree CTR registration flag is invalid");
            const ui8 flags = dynamic ? 1 | (restored.RegisteredCtrFlags[index] ? 2 : 0) : 2;
            // Derived tree CTRs have an independent unit user weight. Their
            // model-size penalty is configured separately by the runtime.
            const float expectedWeight = dynamic || staticWeights.empty() ? 1.0f : staticWeights[feature];
            CB_ENSURE(snapshot.TreeCtrCounts[feature] == count &&
                      snapshot.TreeCtrWeights[feature] == expectedWeight &&
                      snapshot.TreeCtrFlags[feature] == flags &&
                      (dynamic || snapshot.TreeCtrActive[feature]),
                      "Saved Metal tree CTR feature metadata differs from the restored registry");
        }
    }

    // Connects the append-only, history-specific CTR registry to either scalar
    // FeatureParallel runtime. Numeric/simple columns keep their original IDs;
    // only eligibility changes when a pure-tree tensor expires.
    class TMetalTreeCtrSession {
    public:
        TMetalTreeCtrSession(void* handle, bool ordered, TMetalTreeCtrFeatures& features,
                             ui32 firstFeature, ui32 binsPerFeature,
                             bool simplePermutationDependent)
            : Handle(handle)
            , Ordered(ordered)
            , Features(features)
            , FirstFeature(firstFeature)
            , FeatureCount(firstFeature)
            , BinsPerFeature(binsPerFeature)
            , SimplePermutationDependent(simplePermutationDependent)
        {
            CB_ENSURE(Handle && FirstFeature, "Metal tree CTR adapter requires a prepared session");
            // Activity setup also opts the old scalar ABI into FeatureParallel
            // scoring when no tree-dependent columns have been generated yet.
            SetActivity({});
        }

        void BeginTree() {
            Features.BeginTree();
            SetActivity({});
        }

        ui32 ScoreDraws() const {
            return 1 + (Features.GetPermutationCount() > 1 && SimplePermutationDependent) +
                HasActiveTreeFeatures;
        }

        void Selected(const TModelSplit& split, const CBMStructureInfo& info, ui32 permutation) {
            Features.MarkSelected(info.feature);
            // Registration includes the last chosen CTR, but computing a new
            // candidate pack after the final depth would change future grids
            // and eagerly registered configurations that CUDA never visits.
            if (info.finished) return;
            const auto batch = Features.AddSplit(split, permutation);
            Append(batch);
            SetActivity(batch.ActiveFeatures);
        }

        void Restore(const TMetalTreeCtrBatch& batch, const TMetalSnapshot& snapshot) {
            CB_ENSURE(Ordered || snapshot.UsedFeatures == snapshot.TreeCtrUsed,
                      "Saved Metal tree CTR feature-use payloads disagree");
            Append(batch);
            TVector<ui32> counts(FeatureCount);
            TVector<float> weights(FeatureCount);
            TVector<ui8> flags(FeatureCount), used(FeatureCount), active(FeatureCount);
            CopyMetadata(counts, weights, flags, used, active);
            CB_ENSURE(snapshot.TreeCtrCounts == counts && snapshot.TreeCtrWeights == weights &&
                      snapshot.TreeCtrFlags == flags && snapshot.TreeCtrUsed.size() == FeatureCount &&
                      snapshot.TreeCtrActive.size() == FeatureCount,
                      "Saved Metal tree CTR feature metadata differs from the restored registry");
            char error[2048] = {};
            const auto restore = Ordered ? cbm_ordered_session_restore_feature_metadata :
                                           cbm_session_restore_feature_metadata;
            CB_ENSURE(restore(Handle, FeatureCount, snapshot.TreeCtrFlags.data(),
                snapshot.TreeCtrUsed.data(), snapshot.TreeCtrActive.data(), error, sizeof(error)) == 0,
                "Metal tree CTR metadata restoration failed: " << error);
        }

        void Save(TMetalSnapshot* snapshot) const {
            snapshot->TreeCtrState.clear();
            TStringOutput output(snapshot->TreeCtrState);
            Features.Save(&output);
            snapshot->TreeCtrCounts.resize(FeatureCount);
            snapshot->TreeCtrWeights.resize(FeatureCount);
            snapshot->TreeCtrFlags.resize(FeatureCount);
            snapshot->TreeCtrUsed.resize(FeatureCount);
            snapshot->TreeCtrActive.resize(FeatureCount);
            CopyMetadata(snapshot->TreeCtrCounts, snapshot->TreeCtrWeights, snapshot->TreeCtrFlags,
                         snapshot->TreeCtrUsed, snapshot->TreeCtrActive);
        }

    private:
        void Append(const TMetalTreeCtrBatch& batch) {
            CB_ENSURE(batch.FirstFeature == FeatureCount &&
                      batch.PermutationBins.size() == Features.GetPermutationCount(),
                      "Metal tree CTR batch does not follow the runtime feature bank");
            if (!batch.GetFeatureCount()) return;
            TVector<const ui8*> bins;
            for (const auto& values : batch.PermutationBins) bins.push_back(values.data());
            TVector<ui8> flags;
            for (ui8 registered : batch.RegisteredCtrFlags) flags.push_back(1 | (registered ? 2 : 0));
            const ui32 binCapacity = Max(BinsPerFeature, batch.BinsPerFeature);
            CBMAppendFeatureOptions options = {};
            options.permutation_count = Features.GetPermutationCount();
            options.features = batch.GetFeatureCount();
            options.candidates = batch.CandidateFeatures.size();
            options.bins_per_feature = binCapacity;
            ui32 first = 0;
            char error[2048] = {};
            const auto append = Ordered ? cbm_ordered_session_append_features : cbm_session_append_features;
            CB_ENSURE(append(Handle, &options, bins.data(), batch.CandidateFeatures.data(),
                batch.CandidateBins.data(), batch.CandidateTypes.data(), batch.CtrUniqueValues.data(),
                nullptr, flags.data(), nullptr, &first, error, sizeof(error)) == 0,
                "Metal tree CTR feature append failed: " << error);
            CB_ENSURE(first == FeatureCount, "Metal tree CTR runtime assigned inconsistent feature IDs");
            FeatureCount += batch.GetFeatureCount();
            BinsPerFeature = binCapacity;
        }

        void SetActivity(TConstArrayRef<ui32> dynamicFeatures) {
            TVector<ui8> active(FeatureCount, 0);
            std::fill_n(active.begin(), FirstFeature, 1);
            for (ui32 feature : dynamicFeatures) {
                CB_ENSURE(feature >= FirstFeature && feature < FeatureCount,
                          "Metal tree CTR active feature is outside the runtime bank");
                active[feature] = 1;
            }
            char error[2048] = {};
            const auto set = Ordered ? cbm_ordered_session_set_feature_activity : cbm_session_set_feature_activity;
            CB_ENSURE(set(Handle, FeatureCount, active.data(), error, sizeof(error)) == 0,
                      "Metal tree CTR activity update failed: " << error);
            HasActiveTreeFeatures = !dynamicFeatures.empty();
        }

        void CopyMetadata(TArrayRef<ui32> counts, TArrayRef<float> weights, TArrayRef<ui8> flags,
                          TArrayRef<ui8> used, TArrayRef<ui8> active) const {
            char error[2048] = {};
            const auto copy = Ordered ? cbm_ordered_session_copy_feature_metadata : cbm_session_copy_feature_metadata;
            CB_ENSURE(copy(Handle, FeatureCount, counts.data(), weights.data(), flags.data(),
                used.data(), active.data(), error, sizeof(error)) == 0,
                "Metal tree CTR metadata copy failed: " << error);
        }

        void* Handle;
        const bool Ordered;
        TMetalTreeCtrFeatures& Features;
        const ui32 FirstFeature;
        ui32 FeatureCount;
        ui32 BinsPerFeature;
        const bool SimplePermutationDependent;
        bool HasActiveTreeFeatures = false;
    };
}
