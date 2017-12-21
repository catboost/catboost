#pragma once

#include "columns.h"
#include <util/generic/yexception.h>

namespace NCatboostCuda {

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

        TDataProvider const* Get() const {
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

        bool HasQueries() const {
            return QueryIds.size() == Targets.size();
        }

        const TVector<ui32>& GetQueryIds() const {
            CB_ENSURE(HasQueries(), "Current mode need query ids but they were not found in loaded data");
            return QueryIds;
        }

        bool HasGroupIds() const {
            return GroupIds.size() == QueryIds.size();
        }

        const TVector<ui32>& GetGroupIds() const {
            CB_ENSURE(HasGroupIds(), "Current mode need groups ids but they were not found in loaded data");
            return GroupIds;
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

        const THashMap<ui32, TVector<TPair>>& GetPairs() const {
            return QueryPairs;
        };

    private:
        void FillQueryPairs(const TVector<TPair>& pairs) {
            CB_ENSURE(QueryIds.size(), "Error: provide query ids");
            THashMap<ui32, ui32> queryOffsets;
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

    private:
        TVector<TFeatureColumnPtr> Features;

        TVector<ui64> Order;
        TVector<ui32> QueryIds;
        TVector<ui32> GroupIds;
        THashMap<ui32, TVector<TPair>> QueryPairs;

        TVector<ui32> DocIds;
        TVector<float> Targets;
        TVector<float> Weights;
        TVector<TVector<float>> Baseline;

        TVector<ui64> Timestamp;

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
