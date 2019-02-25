#pragma once

#include "meta_info.h"
#include "objects_grouping.h"
#include "util.h"
#include "weights.h"

#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/data_types/query.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/serialization.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <library/binsaver/bin_saver.h>

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

        void PrepareForInitialization(const TDataMetaInfo& metaInfo, ui32 objectCount, ui32 prevTailSize);
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
            TMaybe<NPar::TLocalExecutor*> localExecutor
        ) {
            if (!skipCheck) {
                data.Check(*objectsGrouping, *localExecutor);
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

        void SetObjectsGrouping(TObjectsGroupingPtr objectsGrouping);

        void SetBaseline(TBaselineArrayRef baseline); // [approxIdx][objectIdx]

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
        friend class TQuantizationImpl;
        friend class TRawBuilderDataHelper;

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


    enum class ETargetType : ui32 {
        BinClass,
        MultiClass,
        Regression,
        GroupwiseRanking,
        GroupPairwiseRanking,
        Simple,
        UserDefined
    };


    struct TTargetDataSpecification {
    public:
        ETargetType Type;

        // more details about target data if needed, used to lookup proper target in TTargetDataProviders
        TString Description;

    public:
        // for BinSaver
        TTargetDataSpecification() = default;

        explicit TTargetDataSpecification(ETargetType type, const TString& description = TString())
            : Type(type)
            , Description(description)
        {}

        bool operator==(const TTargetDataSpecification& rhs) const {
            return (Type == rhs.Type) && (Description == rhs.Description);
        }

        SAVELOAD(Type, Description);
    };

}

template <>
struct THash<NCB::TTargetDataSpecification> {
    inline size_t operator()(const NCB::TTargetDataSpecification& targetDataSpecification) const {
        return MultiHash(targetDataSpecification.Type, targetDataSpecification.Description);
    }
};


namespace NCB {

    template <class TKey, class TSharedDataPtr>
    using TTargetSingleTypeDataCache = THashMap<TKey, TSharedDataPtr>;

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

    template <class TSharedDataPtr>
    using TSrcToSubsetDataCache = TTargetSingleTypeDataCache<TSharedDataPtr, TSharedDataPtr>;

    struct TSubsetTargetDataCache {
        TSrcToSubsetDataCache<TSharedVector<float>> Targets;
        TSrcToSubsetDataCache<TSharedWeights<float>> Weights;

        // multidim baselines are stored as separate pointers for simplicity
        TSrcToSubsetDataCache<TSharedVector<float>> Baselines;
        TSrcToSubsetDataCache<TSharedVector<TQueryInfo>> GroupInfos;
    };


    /*
     * Serialization using IBinSaver works as following (somewhat similar to subset creation):
     *
     * Save:
     *  1) Collect mapping of unique data ids to data itself in TSerializationTargetDataCache.
     *     Target data providers serialize with these data ids instead of actual data
     *     to a temporary binSaver stream.
     *  2) Save data in TSerializationTargetDataCache. Save binSaver stream with data providers
     *     descriptions (created at stage 1) with ids after it.
     *
     *  Load:
     *  1) Load data in TSerializationTargetDataCache.
     *  2) Read data with data providers
     *     descriptions with ids, create actual data providers, initializing with actual shared data loaded
     *     from cache at stage 1.
     *
     */

    // key value 0 is special - means this field is optional while serializing
    template <class TSharedDataPtr>
    using TSerializationTargetSingleTypeDataCache =
        TTargetSingleTypeDataCache<ui64, TSharedDataPtr>;

    struct TSerializationTargetDataCache {
        TSerializationTargetSingleTypeDataCache<TSharedVector<float>> Targets;
        TSerializationTargetSingleTypeDataCache<TSharedWeights<float>> Weights;

        // multidim baselines are stored as separate pointers for simplicity
        TSerializationTargetSingleTypeDataCache<TSharedVector<float>> Baselines;
        TSerializationTargetSingleTypeDataCache<TSharedVector<TQueryInfo>> GroupInfos;

    public:
        SAVELOAD_WITH_SHARED(Targets, Weights, Baselines, GroupInfos)
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
        friend class TTargetSerialization;

    protected:
        virtual void SaveWithCache(
            IBinSaver* binSaver,
            TSerializationTargetDataCache* cache
        ) const = 0;

        void SaveCommon(IBinSaver* binSaver) const {
            SaveMulti(binSaver, Specification);
        }

    protected:
        TTargetDataSpecification Specification;
        TObjectsGroupingPtr ObjectsGrouping;
    };

    using TTargetDataProviderPtr = TIntrusivePtr<TTargetDataProvider>;

    using TTargetDataProviders = THashMap<TTargetDataSpecification, TTargetDataProviderPtr>;


    void GetGroupInfosSubset(
        TConstArrayRef<TQueryInfo> src,
        const TObjectsGroupingSubset& objectsGroupingSubset,
        NPar::TLocalExecutor* localExecutor,
        TVector<TQueryInfo>* dstSubset
    );

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
        TMaybeData<TConstArrayRef<float>> GetTarget() const { // [objectIdx]
            return Target ? TMaybeData<TConstArrayRef<float>>(*Target) : Nothing();
        }

        // after preprocessing - adjusted for classes, group etc. weights
        const TWeights<float>& GetWeights() const { // [objectIdx]
            return *Weights;
        }

        TMaybeData<TConstArrayRef<float>> GetBaseline() const { // [objectIdx], can be empty
            return Baseline ? TMaybeData<TConstArrayRef<float>>(*Baseline) : Nothing();
        }

    protected:
        friend class TTargetSerialization;

    protected:
        void SaveWithCache(IBinSaver* binSaver, TSerializationTargetDataCache* cache) const override;

        static TBinClassTarget Load(
            const TString& description,
            TObjectsGroupingPtr objectsGrouping,
            const TSerializationTargetDataCache& cache,
            IBinSaver* binSaver
        );

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
            ui32 classCount,
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

        ui32 GetClassCount() const {
            return ClassCount;
        }

        // after preprocessing - enumerating labels if necessary, etc.
        TMaybeData<TConstArrayRef<float>> GetTarget() const { // [objectIdx]
            return Target ? TMaybeData<TConstArrayRef<float>>(*Target) : Nothing();
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
        friend class TTargetSerialization;

    protected:
        void SaveWithCache(IBinSaver* binSaver, TSerializationTargetDataCache* cache) const override;

        static TMultiClassTarget Load(
            const TString& description,
            TObjectsGroupingPtr objectsGrouping,
            const TSerializationTargetDataCache& cache,
            IBinSaver* binSaver
        );

    protected:
        ui32 ClassCount;
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


        TMaybeData<TConstArrayRef<float>> GetTarget() const { // [objectIdx]
            return Target ? TMaybeData<TConstArrayRef<float>>(*Target) : Nothing();
        }

        // after preprocessing - adjusted for group etc. weights
        const TWeights<float>& GetWeights() const { // [objectIdx]
            return *Weights;
        }

        TMaybeData<TConstArrayRef<float>> GetBaseline() const { // [objectIdx], can be empty
            return Baseline ? TMaybeData<TConstArrayRef<float>>(*Baseline) : Nothing();
        }

    protected:
        friend class TTargetSerialization;

    protected:
        void SaveWithCache(IBinSaver* binSaver, TSerializationTargetDataCache* cache) const override;

        static TRegressionTarget Load(
            const TString& description,
            TObjectsGroupingPtr objectsGrouping,
            const TSerializationTargetDataCache& cache,
            IBinSaver* binSaver
        );

    protected:
        TSharedVector<float> Target; // [objectIdx], can be nullptr
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


        TMaybeData<TConstArrayRef<float>> GetTarget() const { // [objectIdx]
            return Target ? TMaybeData<TConstArrayRef<float>>(*Target) : Nothing();
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
        friend class TTargetSerialization;

    protected:
        void SaveWithCache(IBinSaver* binSaver, TSerializationTargetDataCache* cache) const override;

        static TGroupwiseRankingTarget Load(
            const TString& description,
            TObjectsGroupingPtr objectsGrouping,
            const TSerializationTargetDataCache& cache,
            IBinSaver* binSaver
        );

    protected:
        TSharedVector<float> Target; // [objectIdx], can be nullptr
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
        friend class TTargetSerialization;

    protected:
        void SaveWithCache(IBinSaver* binSaver, TSerializationTargetDataCache* cache) const override;

        static TGroupPairwiseRankingTarget Load(
            const TString& description,
            TObjectsGroupingPtr objectsGrouping,
            const TSerializationTargetDataCache& cache,
            IBinSaver* binSaver
        );

    protected:
        TSharedVector<float> Baseline; // [objectIdx], can be nullptr (means no Baseline)
        TSharedVector<TQueryInfo> GroupInfo;
    };


    class TSimpleTarget : public TTargetDataProvider {
    public:
        TSimpleTarget(
            const TString& description,
            TObjectsGroupingPtr objectsGrouping,
            TSharedVector<float> target,
            bool skipCheck = false
        );

        bool operator==(const TSimpleTarget& rhs) const {
            return ((const TTargetDataProvider&)(*this) == (const TTargetDataProvider&)rhs) &&
                (*Target == *rhs.Target);
        }

        void GetSourceDataForSubsetCreation(TSubsetTargetDataCache* subsetTargetDataCache) const override;

        TTargetDataProviderPtr GetSubset(
            TObjectsGroupingPtr objectsGrouping,
            const TSubsetTargetDataCache& subsetTargetDataCache
        ) const override;


        TMaybeData<TConstArrayRef<float>> GetTarget() const { // [objectIdx]
            return Target ? TMaybeData<TConstArrayRef<float>>(*Target) : Nothing();
        }

    protected:
        friend class TTargetSerialization;

    protected:
        void SaveWithCache(IBinSaver* binSaver, TSerializationTargetDataCache* cache) const override;

        static TSimpleTarget Load(
            const TString& description,
            TObjectsGroupingPtr objectsGrouping,
            const TSerializationTargetDataCache& cache,
            IBinSaver* binSaver
        );

    protected:
        TSharedVector<float> Target; // [objectIdx], can be nullptr
    };

    // TODO(akhropov): remove when custom objective type can be properly specified. MLTOOLS-3022.
    class TUserDefinedTarget : public TTargetDataProvider {
    public:
        TUserDefinedTarget(
            const TString& description,
            TObjectsGroupingPtr objectsGrouping,
            TSharedVector<float> target,
            TSharedWeights<float> weights,
            bool skipCheck = false
        );

        bool operator==(const TUserDefinedTarget& rhs) const {
            return ((const TTargetDataProvider&)(*this) == (const TTargetDataProvider&)rhs) &&
                (*Target == *rhs.Target) && (*Weights == *rhs.Weights);
        }

        void GetSourceDataForSubsetCreation(TSubsetTargetDataCache* subsetTargetDataCache) const override;

        TTargetDataProviderPtr GetSubset(
            TObjectsGroupingPtr objectsGrouping,
            const TSubsetTargetDataCache& subsetTargetDataCache
        ) const override;


        TMaybeData<TConstArrayRef<float>> GetTarget() const { // [objectIdx]
            return Target ? TMaybeData<TConstArrayRef<float>>(*Target) : Nothing();
        }

        // after preprocessing - adjusted for group etc. weights
        const TWeights<float>& GetWeights() const { // [objectIdx]
            return *Weights;
        }

    protected:
        friend class TTargetSerialization;

    protected:
        void SaveWithCache(IBinSaver* binSaver, TSerializationTargetDataCache* cache) const override;

        static TUserDefinedTarget Load(
            const TString& description,
            TObjectsGroupingPtr objectsGrouping,
            const TSerializationTargetDataCache& cache,
            IBinSaver* binSaver
        );

    protected:
        TSharedVector<float> Target; // [objectIdx], can be nullptr
        TSharedWeights<float> Weights; // [objectIdx]
    };


    /* temporary compatibility helpers for support of old single TDataProvider/TDataset interface
     * it is assumed that we can get target, weights, baseline, groupInfo data from any targetDataProvider
     *  - they all contain the same data, just filtered by target-specific data types
     *
     *  This situation will change in the future:
     *    TODO(akhropov): proper support of multiple targets - MLTOOLS-2337
     */

    TMaybeData<TConstArrayRef<float>> GetMaybeTarget(const TTargetDataProviders& targetDataProviders);

    /* will fail if only GroupPairwiseRanking targets are in targetDataProviders
     * this case has to be handled for now (e.g. by always adding non-GroupPairwiseRanking provider)
     */
    TConstArrayRef<float> GetTarget(const TTargetDataProviders& targetDataProviders);

    // will return empty result if weights are trivial
    TConstArrayRef<float> GetWeights(const TTargetDataProviders& targetDataProviders);

    // will return empty vector if there's no baseline data
    TVector<TConstArrayRef<float>> GetBaseline(const TTargetDataProviders& targetDataProviders);

    TConstArrayRef<TQueryInfo> GetGroupInfo(const TTargetDataProviders& targetDataProviders);


    // needed to make friends with TTargetDataProvider s
    class TTargetSerialization {
    public:
        static void Load(
            TObjectsGroupingPtr objectsGrouping,
            IBinSaver* binSaver,
            TTargetDataProviders* targetDataProviders
        );

        static void SaveNonSharedPart(const TTargetDataProviders& targetDataProviders, IBinSaver* binSaver);
    };
}
