#pragma once

#include "objects_grouping.h"
#include "util.h"
#include "weights.h"

#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/data_types/query.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/digest/multi.h>
#include <util/generic/array_ref.h>
#include <util/generic/hash.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/system/types.h>
#include <util/str_stl.h>


namespace NCB {
    /* TODO(akhropov): All target data is copied when creating subsets now, which can be suboptimal memorywise.
     *  Maybe implement it using TArraySubsets like for objects' feature columns data
     */


    template <class T>
    using TSharedVector = TAtomicSharedPtr<TVector<T>>;

    void CheckGroupWeights(
        TConstArrayRef<float> groupWeights,
        const TObjectsGrouping& objectsGrouping
    );

    void CheckPairs(TConstArrayRef<TPair> pairs, const TObjectsGrouping& objectsGrouping);

    // TODO(akhropov): maybe move to TQueryInfo implementation
    void CheckOneGroupInfo(const TQueryInfo& groupInfo);

    void CheckGroupInfo(
        TConstArrayRef<TQueryInfo> groupInfoVector,
        const TObjectsGrouping& objectsGrouping,
        bool mustContainPairData = false
    );

    // for use while building
    struct TRawTargetData {
    public:
        TMaybeData<TVector<TString>> Target; // [objectIdx], can be empty (if pairs are used)
        TVector<TVector<float>> Baseline; // [approxIdx][objectIdx], can be empty

        // if not specified in source data - do not forget to set as trivial, it is checked
        TWeights<float> Weights; // [objectIdx]

        // if not specified in source data - do not forget to set as trivial, it is checked
        // weights in each group must be equal, it's checked
        TWeights<float> GroupWeights; // [objectIdx]

        TVector<TPair> Pairs; // can be empty
    public:
        bool operator==(const TRawTargetData& rhs) const;

        void SetTrivialWeights(ui32 objectCount) {
            Weights = TWeights<float>(objectCount);
            GroupWeights = TWeights<float>(objectCount);
        }

        void Check(const TObjectsGrouping& objectsGrouping, NPar::TLocalExecutor* localExecutor) const;
    };


    using TBaselineArrayRef = TConstArrayRef<TConstArrayRef<float>>;


    class TRawTargetDataProvider {
    public:
        // skipCheck can be used to avoid repeated checking if we already know that data has been checked
        explicit TRawTargetDataProvider(
            TObjectsGroupingPtr objectsGrouping,
            TRawTargetData&& data,
            bool skipCheck,

            // used only if skipCheck == false, it's ok to pass nullptr if skipCheck is true
            NPar::TLocalExecutor* localExecutor
        ) {
            if (!skipCheck) {
                data.Check(*objectsGrouping, localExecutor);
            }
            ObjectsGrouping = std::move(objectsGrouping);
            Data = std::move(data);
            SetBaselineViewFromBaseline();
        }

        bool operator==(const TRawTargetDataProvider& rhs) const {
            return (*ObjectsGrouping == *rhs.ObjectsGrouping) && (Data == rhs.Data);
        }

        // just for convenience
        ui32 GetObjectCount() const {
            return ObjectsGrouping->GetObjectCount();
        }

        TObjectsGroupingPtr GetObjectsGrouping() const {
            return ObjectsGrouping;
        }

        // [objectIdx], can return empty array (if pairs are used)
        TMaybeData<TConstArrayRef<TString>> GetTarget() const {
            return Data.Target;
        }

        // can return empty array
        TMaybeData<TBaselineArrayRef> GetBaseline() const {  // [approxIdx][objectIdx]
            return !BaselineView.empty() ? TMaybeData<TBaselineArrayRef>(BaselineView) : Nothing();
        }

        const TWeights<float>& GetWeights() const { // [objectIdx]
            return Data.Weights;
        }

        const TWeights<float>& GetGroupWeights() const { // [objectIdx]
            return Data.GroupWeights;
        }

        TConstArrayRef<TPair> GetPairs() const { // can return empty array
            return Data.Pairs;
        }

        /* set functions are needed for current python mutable Pool interface
           builders should prefer to set fields directly to avoid unnecessary data copying
        */

        void SetBaseline(TConstArrayRef<TConstArrayRef<float>> baseline); // [approxIdx][objectIdx]

        void SetWeights(TConstArrayRef<float> weights) { // [objectIdx]
            CheckWeights(weights, GetObjectCount(), "Weights");
            AssignWeights(weights, &Data.Weights);
        }

        void SetGroupWeights(TConstArrayRef<float> groupWeights) { // [objectIdx]
            CheckGroupWeights(groupWeights, *ObjectsGrouping);
            AssignWeights(groupWeights, &Data.GroupWeights);
        }

        void SetPairs(TConstArrayRef<TPair> pairs) {
            CheckPairs(pairs, *ObjectsGrouping);
            Assign(pairs, &Data.Pairs);
        }

        TRawTargetDataProvider GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            NPar::TLocalExecutor* localExecutor
        ) const;

    private:
        void AssignWeights(TConstArrayRef<float> src, TWeights<float>* dst);

        void SetBaselineViewFromBaseline() { // call after setting Data.Baseline
            BaselineView.assign(Data.Baseline.begin(), Data.Baseline.end());
        }

    private:
        TObjectsGroupingPtr ObjectsGrouping;
        TRawTargetData Data;

        // for returning from GetBaseline
        TVector<TConstArrayRef<float>> BaselineView; // [approxIdx][objectIdx]
    };


    enum class ETargetType {
        BinClass,
        MultiClass,
        Regression,
        GroupwiseRanking,
        GroupPairwiseRanking
    };


    struct TTargetDataSpecification {
    public:
        ETargetType Type;

        // more details about target data if needed, used to lookup proper target in TTargetDataProviders
        TString Description;

    public:
        explicit TTargetDataSpecification(ETargetType type, const TString& description = TString())
            : Type(type)
            , Description(description)
        {}

        bool operator==(const TTargetDataSpecification& rhs) const {
            return (Type == rhs.Type) && (Description == rhs.Description);
        }
    };

}

template <>
struct THash<NCB::TTargetDataSpecification> {
    inline size_t operator()(const NCB::TTargetDataSpecification& targetDataSpecification) const {
        return MultiHash(targetDataSpecification.Type, targetDataSpecification.Description);
    }
};


namespace NCB {

    /*
     * data is stored in shared pointers (TIntrusivePtr is also a variant)
     * because it can be shared between several targets data providers
     *
     * for subset creation efficiency we had to somewhat violate incapsulation and store all shared data
     * in mapping cache in order to create subsets only once for each shared data
     *
     * Subset creation works in 3 stages:
     *
     *   1) all data gathered from all data providers to keys in TSubsetTargetDataCache using calling
     *       TTargetDataProvider::GetSourceDataForSubsetCreation
     *   2) subsets for all cached data are created in parallel
     *   3) all target data providers create subsets from cached subset mapping in TSubsetTargetDataCache
     *       in TTargetDataProvider::GetSubset
     *
     *   GetSubsets below is the function that does all these stages
     */

    template <class TSharedDataPtr> // TSharedDataPtr is TIntrusivePtr<...> or TAtomicSharedPtr<...>
    using TSrcToSubsetDataCache = THashMap<TSharedDataPtr, TSharedDataPtr>;


    struct TSubsetTargetDataCache {
        TSrcToSubsetDataCache<TSharedVector<float>> Targets;
        TSrcToSubsetDataCache<TSharedWeights<float>> Weights;

        // multidim baselines are stored as separate pointers for simplicity
        TSrcToSubsetDataCache<TSharedVector<float>> Baselines;
        TSrcToSubsetDataCache<TSharedVector<TQueryInfo>> GroupInfos;
    };


    class TTargetDataProvider : public TThrRefBase {
    public:
        TTargetDataProvider(TTargetDataSpecification&& specification, TObjectsGroupingPtr objectsGrouping)
            : Specification(std::move(specification))
            , ObjectsGrouping(std::move(objectsGrouping))
        {}

        bool operator==(const TTargetDataProvider& rhs) const {
            return (Specification == rhs.Specification) && (*ObjectsGrouping == *rhs.ObjectsGrouping);
        }

        const TTargetDataSpecification& GetSpecification() const {
            return Specification;
        }

        virtual void GetSourceDataForSubsetCreation(
            TSubsetTargetDataCache* subsetTargetDataCache
        ) const = 0;

        virtual TIntrusivePtr<TTargetDataProvider> GetSubset(
            TObjectsGroupingPtr objectsGrouping,
            const TSubsetTargetDataCache& subsetTargetDataCache
        ) const = 0;

        // just for convenience
        ui32 GetObjectCount() const {
            return ObjectsGrouping->GetObjectCount();
        }

        TObjectsGroupingPtr GetObjectsGrouping() const {
            return ObjectsGrouping;
        }

    protected:
        TTargetDataSpecification Specification;
        TObjectsGroupingPtr ObjectsGrouping;
    };

    using TTargetDataProviderPtr = TIntrusivePtr<TTargetDataProvider>;

    using TTargetDataProviders = THashMap<TTargetDataSpecification, TTargetDataProviderPtr>;


    TTargetDataProviders GetSubsets(
        const TTargetDataProviders& srcTargetDataProviders,
        const TObjectsGroupingSubset& objectsGroupingSubset,
        NPar::TLocalExecutor* localExecutor
    );


    // skipCheck can be used to avoid repeated checking if we already know that data has been checked

    class TBinClassTarget : public TTargetDataProvider {
    public:
        TBinClassTarget(
            const TString& description,
            TObjectsGroupingPtr objectsGrouping,
            TSharedVector<float> target,
            TSharedWeights<float> weights,
            TSharedVector<float> baseline, // сan be nullptr if baseline not available
            bool skipCheck = false
        );

        bool operator==(const TBinClassTarget& rhs) const {
            return ((const TTargetDataProvider&)(*this) == (const TTargetDataProvider&)rhs) &&
                (*Target == *rhs.Target) && (*Weights == *rhs.Weights) && (*Baseline == *rhs.Baseline);
        }

        void GetSourceDataForSubsetCreation(TSubsetTargetDataCache* subsetTargetDataCache) const override;

        TTargetDataProviderPtr GetSubset(
            TObjectsGroupingPtr objectsGrouping,
            const TSubsetTargetDataCache& subsetTargetDataCache
        ) const override;

        // after preprocessing - enumerating labels if necessary, etc.
        TConstArrayRef<float> GetTarget() const { // [objectIdx]
            return *Target;
        }

        // after preprocessing - adjusted for classes, group etc. weights
        const TWeights<float>& GetWeights() const { // [objectIdx]
            return *Weights;
        }

        TMaybeData<TConstArrayRef<float>> GetBaseline() const { // [objectIdx], can be empty
            return Baseline ? TMaybeData<TConstArrayRef<float>>(*Baseline) : Nothing();
        }

    protected:
        TSharedVector<float> Target; // [objectIdx]
        TSharedWeights<float> Weights; // [objectIdx]
        TSharedVector<float> Baseline; // [objectIdx], can be nullptr (means no Baseline)
    };

    class TMultiClassTarget : public TTargetDataProvider {
    public:
        TMultiClassTarget(
            const TString& description,
            TObjectsGroupingPtr objectsGrouping,
            TSharedVector<float> target,
            TSharedWeights<float> weights,
            TVector<TSharedVector<float>>&& baseline,  // сan be empty if baseline not available
            bool skipCheck = false
        );

        bool operator==(const TMultiClassTarget& rhs) const {
            return ((const TTargetDataProvider&)(*this) == (const TTargetDataProvider&)rhs) &&
                (*Target == *rhs.Target) && (*Weights == *rhs.Weights) && (BaselineView == rhs.BaselineView);
        }

        void GetSourceDataForSubsetCreation(TSubsetTargetDataCache* subsetTargetDataCache) const override;

        TTargetDataProviderPtr GetSubset(
            TObjectsGroupingPtr objectsGrouping,
            const TSubsetTargetDataCache& subsetTargetDataCache
        ) const override;

        // after preprocessing - enumerating labels if necessary, etc.
        TConstArrayRef<float> GetTarget() const { // [objectIdx]
            return *Target;
        }

        // after preprocessing - adjusted for classes, group etc. weights
        const TWeights<float>& GetWeights() const { // [objectIdx]
            return *Weights;
        }

        // [approxIdx][objectIdx], can be empty
        TMaybeData<TBaselineArrayRef> GetBaseline() const {
            return !BaselineView.empty() ? TMaybeData<TBaselineArrayRef>(BaselineView) : Nothing();
        }

    protected:
        TSharedVector<float> Target; // [objectIdx]
        TSharedWeights<float> Weights; // [objectIdx]
        TVector<TSharedVector<float>> Baseline; // [approxIdx][objectIdx], can be empty

        // for returning from GetBaseline
        TVector<TConstArrayRef<float>> BaselineView; // [approxIdx][objectIdx], can be empty
    };

    class TRegressionTarget : public TTargetDataProvider {
    public:
        TRegressionTarget(
            const TString& description,
            TObjectsGroupingPtr objectsGrouping,
            TSharedVector<float> target,
            TSharedWeights<float> weights,
            TSharedVector<float> baseline,  // сan be nullptr if baseline not available
            bool skipCheck = false
        );

        bool operator==(const TRegressionTarget& rhs) const {
            return ((const TTargetDataProvider&)(*this) == (const TTargetDataProvider&)rhs) &&
                (*Target == *rhs.Target) && (*Weights == *rhs.Weights) && (*Baseline == *rhs.Baseline);
        }

        void GetSourceDataForSubsetCreation(TSubsetTargetDataCache* subsetTargetDataCache) const override;

        TTargetDataProviderPtr GetSubset(
            TObjectsGroupingPtr objectsGrouping,
            const TSubsetTargetDataCache& subsetTargetDataCache
        ) const override;

        TConstArrayRef<float> GetTarget() const { // [objectIdx]
            return *Target;
        }

        // after preprocessing - adjusted for group etc. weights
        const TWeights<float>& GetWeights() const { // [objectIdx]
            return *Weights;
        }

        TMaybeData<TConstArrayRef<float>> GetBaseline() const { // [objectIdx], can be empty
            return Baseline ? TMaybeData<TConstArrayRef<float>>(*Baseline) : Nothing();
        }

    protected:
        TSharedVector<float> Target; // [objectIdx]
        TSharedWeights<float> Weights; // [objectIdx]
        TSharedVector<float> Baseline; // [objectIdx], can be nullptr (means no Baseline)
    };


    class TGroupwiseRankingTarget : public TTargetDataProvider {
    public:
        TGroupwiseRankingTarget(
            const TString& description,
            TObjectsGroupingPtr objectsGrouping,
            TSharedVector<float> target,
            TSharedWeights<float> weights,
            TSharedVector<float> baseline,  // сan be nullptr if baseline not available
            TSharedVector<TQueryInfo> groupInfo,
            bool skipCheck = false
        );

        bool operator==(const TGroupwiseRankingTarget& rhs) const {
            return ((const TTargetDataProvider&)(*this) == (const TTargetDataProvider&)rhs) &&
                (*Target == *rhs.Target) && (*Weights == *rhs.Weights) && (*Baseline == *rhs.Baseline) &&
                (*GroupInfo == *rhs.GroupInfo);
        }

        void GetSourceDataForSubsetCreation(TSubsetTargetDataCache* subsetTargetDataCache) const override;

        TTargetDataProviderPtr GetSubset(
            TObjectsGroupingPtr objectsGrouping,
            const TSubsetTargetDataCache& subsetTargetDataCache
        ) const override;

        TConstArrayRef<float> GetTarget() const { // [objectIdx]
            return *Target;
        }

        // after preprocessing - adjusted for group etc. weights
        const TWeights<float>& GetWeights() const { // [objectIdx]
            return *Weights;
        }

        TMaybeData<TConstArrayRef<float>> GetBaseline() const { // [objectIdx], can be empty
            return Baseline ? TMaybeData<TConstArrayRef<float>>(*Baseline) : Nothing();
        }

        TConstArrayRef<TQueryInfo> GetGroupInfo() const {
            return *GroupInfo;
        }

    protected:
        TSharedVector<float> Target; // [objectIdx]
        TSharedWeights<float> Weights; // [objectIdx]
        TSharedVector<float> Baseline; // [objectIdx], can be nullptr (means no Baseline)
        TSharedVector<TQueryInfo> GroupInfo;
    };

    class TGroupPairwiseRankingTarget : public TTargetDataProvider {
    public:
        TGroupPairwiseRankingTarget(
            const TString& description,
            TObjectsGroupingPtr objectsGrouping,
            TSharedVector<float> baseline,  // сan be nullptr if baseline not available
            TSharedVector<TQueryInfo> groupInfo,
            bool skipCheck = false
        );

        bool operator==(const TGroupPairwiseRankingTarget& rhs) const {
            return ((const TTargetDataProvider&)(*this) == (const TTargetDataProvider&)rhs) &&
                (*Baseline == *rhs.Baseline) && (*GroupInfo == *rhs.GroupInfo);
        }

        void GetSourceDataForSubsetCreation(TSubsetTargetDataCache* subsetTargetDataCache) const override;

        TTargetDataProviderPtr GetSubset(
            TObjectsGroupingPtr objectsGrouping,
            const TSubsetTargetDataCache& subsetTargetDataCache
        ) const override;

        TMaybeData<TConstArrayRef<float>> GetBaseline() const { // [objectIdx], can be empty
            return Baseline ? TMaybeData<TConstArrayRef<float>>(*Baseline) : Nothing();
        }

        TConstArrayRef<TQueryInfo> GetGroupInfo() const {
            return *GroupInfo;
        }

    protected:
        TSharedVector<float> Baseline; // [objectIdx], can be nullptr (means no Baseline)
        TSharedVector<TQueryInfo> GroupInfo;
    };


    /* temporary compatibility helpers for support of old single TDataProvider/TDataset interface
     * it is assumed that we can get target, weights, baseline, groupInfo data from any targetDataProvider
     *  - they all contain the same data, just filtered by target-specific data types
     *
     *  This situation will change in the future:
     *    TODO(akhropov): proper support of multiple targets - MLTOOLS-2337
     */

    /* will fail if only GroupPairwiseRanking targets are in targetDataProviders
     * this case has to be handled for now (e.g. by always adding non-GroupPairwiseRanking provider)
     */
    TConstArrayRef<float> GetTarget(const TTargetDataProviders& targetDataProviders);

    // will return empty result if weights are trivial
    TConstArrayRef<float> GetWeights(const TTargetDataProviders& targetDataProviders);

    // will return empty vector if there's no baseline data
    TVector<TConstArrayRef<float>> GetBaseline(const TTargetDataProviders& targetDataProviders);

    TConstArrayRef<TQueryInfo> GetGroupInfo(const TTargetDataProviders& targetDataProviders);

}
