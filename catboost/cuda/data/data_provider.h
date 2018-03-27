#pragma once

#include "columns.h"
#include <util/generic/yexception.h>

namespace NCatboostCuda {
    template <class T>
    inline bool AreEqualTo(const TVector<T>& entries, const T& value) {
        for (auto& entry : entries) {
            if (entry != value) {
                return false;
            }
        }
        return true;
    }

    class TDataProvider: public TMoveOnly {
    public:
        explicit TDataProvider()
            : HasTimeFlag(false)
            , ShuffleSeed(0)
        {
        }

        bool HasTime() const {
            CB_ENSURE(!(HasTimeFlag && IsShuffledFlag), "Error: dataProvider with time was shuffled");
            return HasTimeFlag;
        }

        bool IsEmpty() const {
            return GetSampleCount() == 0;
        }

        ui64 GetShuffleSeed() const {
            return ShuffleSeed;
        }

        size_t GetEffectiveFeatureCount() const {
            return Features.size();
        }

        bool HasFeatureId(ui32 featureId) const {
            return IndicesToLocalIndicesRemap.has(featureId);
        }

        const IFeatureValuesHolder& GetFeatureById(ui32 featureId) const {
            if (!IndicesToLocalIndicesRemap.has(featureId)) {
                ythrow TCatboostException() << "No feature with feature id #" << featureId << " found";
            }
            const ui32 localId = IndicesToLocalIndicesRemap.at(featureId);
            CB_ENSURE(Features[localId], "Error: nullptr feature found. something wrong");
            return *Features[localId];
        }

        const IFeatureValuesHolder& GetFeatureByIndex(ui32 index) const {
            CB_ENSURE(Features[index], "Error: nullptr feature found. something wrong");
            return *Features[index];
        }

        const TBinarizedFloatValuesHolder& GetBinarizedFloatFeatureById(ui32 id) const {
            auto index = IndicesToLocalIndicesRemap.at(id);
            CB_ENSURE(Features[index], "Error: nullptr feature found. something wrong");
            return dynamic_cast<const TBinarizedFloatValuesHolder&>(*Features[index]);
        }

        size_t GetSampleCount() const {
            return Targets.size();
        }

        const TDataProvider* Get() const {
            if (IsEmpty()) {
                return nullptr;
            }
            return this;
        }

        const TVector<float>& GetTargets() const {
            return Targets;
        }

        const TVector<float>& GetWeights() const {
            return Weights;
        }

        bool IsTrivialWeights() const {
            return AreEqualTo(Weights, 1.0f);
        }

        bool HasQueries() const {
            return QueryIds.size() == Targets.size();
        }

        const TVector<TGroupId>& GetQueryIds() const {
            CB_ENSURE(HasQueries(), "Current mode need query ids but they were not found in loaded data");
            return QueryIds;
        }

        bool HasSubgroupIds() const {
            return SubgroupIds.size() == QueryIds.size();
        }

        const TVector<ui32>& GetSubgroupIds() const {
            CB_ENSURE(HasSubgroupIds(), "Current mode need groups ids but they were not found in loaded data");
            return SubgroupIds;
        }

        const TVector<TString>& GetFeatureNames() const {
            return FeatureNames;
        }

        const TSet<int>& GetCatFeatureIds() const {
            return CatFeatureIds;
        }

        bool HasBaseline() const {
            return Baseline.size() && Baseline[0].size() == GetSampleCount();
        }

        const TVector<float>& GetBaseline() const {
            return Baseline[0];
        }

        void SetShuffleSeed(ui64 seed) {
            CB_ENSURE(!HasTimeFlag, "Error: unset has time flag first");
            IsShuffledFlag = true;
            ShuffleSeed = seed;
        }

        void SetHasTimeFlag(bool flag) {
            HasTimeFlag = flag;
        }

        const THashMap<TGroupId, TVector<TPair>>& GetPairs() const {
            return QueryPairs;
        };

    private:
        void FillQueryPairs(const TVector<TPair>& pairs) {
            CB_ENSURE(QueryIds.size(), "Error: provide query ids");
            THashMap<TGroupId, ui32> queryOffsets;
            for (ui32 doc = 0; doc < QueryIds.size(); ++doc) {
                const auto queryId = QueryIds[doc];
                if (!queryOffsets.has(queryId)) {
                    queryOffsets[queryId] = doc;
                }
            }
            for (const auto& pair : pairs) {
                CB_ENSURE(QueryIds[pair.LoserId] == QueryIds[pair.WinnerId], "Error: pair documents should be in one query");
                const auto queryId = QueryIds[pair.LoserId];
                TPair localPair = pair;
                ui32 offset = queryOffsets[queryId];
                localPair.WinnerId -= offset;
                localPair.LoserId -= offset;
                QueryPairs[queryId].push_back(localPair);
            }
        }

        inline void ApplyOrderToMetaColumns() {
            ApplyPermutation(Order, Targets);
            ApplyPermutation(Order, Weights);
            for (auto& baseline : Baseline) {
                ApplyPermutation(Order, baseline);
            }
            ApplyPermutation(Order, QueryIds);
            ApplyPermutation(Order, SubgroupIds);
            ApplyPermutation(Order, DocIds);
            ApplyPermutation(Order, Timestamp);
        }

    private:
        TVector<TFeatureColumnPtr> Features;
        TVector<ui64> Order;

        TVector<float> Targets;
        TVector<float> Weights;
        TVector<TVector<float>> Baseline;

        TVector<TGroupId> QueryIds;
        TVector<ui32> SubgroupIds;
        TVector<ui32> DocIds;
        TVector<ui64> Timestamp;

        THashMap<TGroupId, TVector<TPair>> QueryPairs;

        TMap<ui32, ui32> IndicesToLocalIndicesRemap;

        void BuildIndicesRemap() {
            IndicesToLocalIndicesRemap.clear();

            for (ui32 i = 0; i < Features.size(); ++i) {
                CB_ENSURE(Features[i], "Error: nullptr feature found. something wrong");
                IndicesToLocalIndicesRemap[Features[i]->GetId()] = i;
            }
        }

        //for cpu model conversion
        TVector<TString> FeatureNames;
        TSet<int> CatFeatureIds;

        bool HasTimeFlag = false;
        ui64 ShuffleSeed = 0;
        bool IsShuffledFlag = false;

        friend class TDataProviderBuilder;

        friend class TCpuPoolBasedDataProviderBuilder;

        friend class TCatBoostProtoPoolReader;
    };

    //TODO(noxoomo): move to proper place
    inline void Reweight(const TVector<float>& targets, const TVector<float>& targetWeights, TVector<float>* weights) {
        CB_ENSURE(targets.size() == weights->size());
        if (targetWeights.size()) {
            for (ui32 doc = 0; doc < targets.size(); ++doc) {
                CB_ENSURE(static_cast<ui32>(targets[doc]) == targets[doc], "Error: target should be natural for reweighting");
                CB_ENSURE(targetWeights[targets[doc]] > 0, "Target weight for class " << targets[doc] << " should be positive");
                (*weights)[doc] *= targetWeights[targets[doc]];
            }
        }
    }
}
