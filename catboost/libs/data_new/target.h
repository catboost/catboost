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

    /*
     * Processed target data is stored in hash maps to enable multiple variants of some data type to be
     * available for different purposes
     * (for example, it might be possible to compute metrics on several Targets simultaneously
     *  or use compute metrics with different Weights)
     *
     * Data is stored with empty string key ("") by default.
     *
     * Data is stored in shared pointers (TIntrusivePtr is also a variant)
     * because it can be shared between different keys
     */
    struct TProcessedTargetData {
    public:
        THashMap<TString, ui32> TargetsClassCount;
        THashMap<TString, TSharedVector<float>> Targets;
        THashMap<TString, TSharedWeights<float>> Weights;
        THashMap<TString, TVector<TSharedVector<float>>> Baselines;
        THashMap<TString, TSharedVector<TQueryInfo>> GroupInfos;

    public:
        bool operator==(const TProcessedTargetData& rhs) const;

        void Check(const TObjectsGrouping& objectsGrouping) const;

        void Load(IBinSaver* binSaver);
        void Save(IBinSaver* binSaver) const;
    };


    template <class TData, class TMapType>
    inline TMaybeData<TData> GetDataFromMap(const TMapType& map, const typename TMapType::key_type& key) {
        const auto dataPtr = MapFindPtr(map, key);
        if (dataPtr) {
            return **dataPtr;
        }
        return Nothing();
    }


    class TTargetDataProvider : public TThrRefBase {
    public:
        TTargetDataProvider(
            TObjectsGroupingPtr objectsGrouping,
            TProcessedTargetData&& processedTargetData,

            // skipCheck can be used to avoid repeated checking if we already know that data has been checked
            bool skipCheck = false);

        bool operator==(const TTargetDataProvider& rhs) const;

        TIntrusivePtr<TTargetDataProvider> GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            NPar::TLocalExecutor* localExecutor
        ) const;


        // just for convenience
        ui32 GetObjectCount() const {
            return ObjectsGrouping->GetObjectCount();
        }

        TObjectsGroupingPtr GetObjectsGrouping() const {
            return ObjectsGrouping;
        }


        // if Target with name does not contain classes returns Nothing()
        TMaybeData<ui32> GetTargetClassCount(const TString& name = "") const {
            auto* classCount = MapFindPtr(Data.TargetsClassCount, name);
            return classCount ? TMaybeData<ui32>(*classCount) : Nothing();
        }

        // after preprocessing - enumerating labels if necessary, etc.
        TMaybeData<TConstArrayRef<float>> GetTarget(const TString& name = "") const { // [objectIdx]
            return GetDataFromMap<TConstArrayRef<float>>(Data.Targets, name);
        }

        // after preprocessing - adjusted for classes, group etc. weights
        TMaybeData<TWeights<float>> GetWeights(const TString& name = "") const { // [objectIdx]
            return GetDataFromMap<TWeights<float>>(Data.Weights, name);
        }

        TMaybeData<TBaselineArrayRef> GetBaseline(const TString& name = "") const { // [approxIdx][objectIdx]
            const auto dataPtr = MapFindPtr(BaselineViews, name);
            if (dataPtr) {
                return TBaselineArrayRef(*dataPtr);
            }
            return Nothing();
        }

        TMaybeData<TConstArrayRef<TQueryInfo>> GetGroupInfo(const TString& name = "") const { // [objectIdx]
            return GetDataFromMap<TConstArrayRef<TQueryInfo>>(Data.GroupInfos, name);
        }


    protected:
        friend class TTargetSerialization;

    protected:
        void SaveDataNonSharedPart(IBinSaver* binSaver) const {
            Data.Save(binSaver);
        }

    private:
        TObjectsGroupingPtr ObjectsGrouping;
        TProcessedTargetData Data;

        // for returning from GetBaseline
        // [approxIdx][objectIdx], can be empty
        THashMap<TString, TVector<TConstArrayRef<float>>> BaselineViews;
    };

    using TTargetDataProviderPtr = TIntrusivePtr<TTargetDataProvider>;


    void GetGroupInfosSubset(
        TConstArrayRef<TQueryInfo> src,
        const TObjectsGroupingSubset& objectsGroupingSubset,
        NPar::TLocalExecutor* localExecutor,
        TVector<TQueryInfo>* dstSubset
    );

    /* temporary compatibility helper for support of old single TDataProvider/TDataset interface
     * will return empty result if weights are trivial
     */
    inline TConstArrayRef<float> GetWeights(const TTargetDataProvider& targetDataProvider) {
        auto maybeWeights = targetDataProvider.GetWeights();
        if (!maybeWeights || maybeWeights->IsTrivial()) {
            return TConstArrayRef<float>();
        }
        return maybeWeights->GetNonTrivialData();
    }


    // needed to make friends with TTargetDataProvider
    class TTargetSerialization {
    public:
        static void Load(
            TObjectsGroupingPtr objectsGrouping,
            IBinSaver* binSaver,
            TTargetDataProviderPtr* targetDataProvider
        );

        static void SaveNonSharedPart(const TTargetDataProvider& targetDataProvider, IBinSaver* binSaver);
    };
}
