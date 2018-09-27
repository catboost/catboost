#pragma once

#include "columns.h"
#include "features_layout.h"
#include "objects_grouping.h"
#include "quantized_features_info.h"
#include "util.h"

#include <catboost/libs/data_types/groupid.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/resource_holder.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/system/types.h>


namespace NCB {

    // if objectsGrouping is defined - check that groupIds correspond to it
    void CheckGroupIds(
        ui32 objectCount,
        TMaybeData<TConstArrayRef<TGroupId>> groupIds,
        TMaybe<TObjectsGroupingPtr> objectsGrouping = Nothing()
    );

    /* if groupIds is empty return trivial grouping
     *  checks that groupIds are consecutive
     */
    TObjectsGrouping CreateObjectsGroupingFromGroupIds(
        ui32 objectCount,
        TMaybeData<TConstArrayRef<TGroupId>> groupIds
    );

    // for use while building
    struct TCommonObjectsData {
    public:
        TVector<TIntrusivePtr<IResourceHolder>> ResourceHolders;

        /* this dataset can be a view from a bigger objects dataset
           this field provides this data to columns in derived classes
        */
        TAtomicSharedPtr<TArraySubsetIndexing<ui32>> SubsetIndexing;

        TMaybeData<TVector<TGroupId>> GroupIds; // [objectIdx]
        TMaybeData<TVector<TSubgroupId>> SubgroupIds; // [objectIdx]
        TMaybeData<TVector<ui64>> Timestamp; // [objectIdx]

    public:
        /* used in TObjectsDataProvider to avoid double checking
         * when ObjectsGrouping is created from GroupIds and GroupId's consistency is already checked
         * in this process
         */
        void CheckAllExceptGroupIds() const;

        // if objectsGrouping is defined and GroupIds are defined check that they are consistent
        void Check(TMaybe<TObjectsGroupingPtr> objectsGrouping) const;

        TCommonObjectsData GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            NPar::TLocalExecutor* localExecutor
        ) const;
    };


    class TObjectsDataProvider : public TThrRefBase {
    public:
        TObjectsDataProvider(
            // if not defined - call CreateObjectsGroupingFromGroupIds
            TMaybe<TObjectsGroupingPtr> objectsGrouping,
            TCommonObjectsData&& commonData,
            bool skipCheck
        );

        ui32 GetObjectCount() const {
            return ObjectsGrouping->GetObjectCount();
        }

        TObjectsGroupingPtr GetObjectsGrouping() const {
            return ObjectsGrouping;
        }

        virtual TIntrusivePtr<TObjectsDataProvider> GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            NPar::TLocalExecutor* localExecutor
        ) const = 0;

        /*
         * GetGroupIds, GetSubgroupIds are common for all implementations,
         *  so they're in this base class
         */

        TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const { // [objectIdx]
            return CommonData.GroupIds;
        }

        TMaybeData<TConstArrayRef<TSubgroupId>> GetSubgroupIds() const { // [objectIdx]
            return CommonData.SubgroupIds;
        }

        TMaybeData<TConstArrayRef<ui64>> GetTimestamp() const { // [objectIdx]
            return CommonData.Timestamp;
        }

    protected:
        TObjectsGroupingPtr ObjectsGrouping;
        TCommonObjectsData CommonData;
    };


    // for use while building and storing this part in TRawObjectsDataProvider
    struct TRawObjectsData {
    public:
        /* some feature holders can contain nullptr
         *  (ignored or this data provider contains only subset of features)
         */
        TVector<THolder<TFloatValuesHolder>> FloatFeatures; // [floatFeatureIdx]
        TVector<THolder<THashedCatValuesHolder>> CatFeatures; // [catFeatureIdx]

        // can be empty if there's no cat features
        TAtomicSharedPtr<TVector<THashMap<ui32, TString>>> CatFeaturesHashToString; // [catFeatureIdx]

    public:
        // TODO(akhropov): Is cat features hashes check too expensive/should be optional for release?
        void Check(
            ui32 objectCount,
            const TFeaturesLayout& featuresLayout,
            NPar::TLocalExecutor* localExecutor
        ) const;
    };

    class TRawObjectsDataProvider : public TObjectsDataProvider {
    public:
        TRawObjectsDataProvider(
            TMaybe<TObjectsGroupingPtr> objectsGrouping, // if not defined - init from groupId
            TCommonObjectsData&& commonData,
            TRawObjectsData&& data,

            bool skipCheck,

            // needed for check parallelization, can pass Nothing() if skipCheck is true
            TMaybe<const TFeaturesLayout*> featuresLayout,
            TMaybe<NPar::TLocalExecutor*> localExecutor
        )
            : TObjectsDataProvider(std::move(objectsGrouping), std::move(commonData), skipCheck)
        {
            if (!skipCheck) {
                data.Check(GetObjectCount(), **featuresLayout, *localExecutor);
            }
            Data = std::move(data);
        }

        TIntrusivePtr<TObjectsDataProvider> GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            NPar::TLocalExecutor* localExecutor
        ) const override;

        /* can return nullptr if this feature is unavailable
         * (ignored or this data provider contains only subset of features)
         */
        TMaybeData<const TFloatValuesHolder*> GetFloatFeature(ui32 floatFeatureIdx) const {
            return MakeMaybeData<const TFloatValuesHolder>(Data.FloatFeatures[floatFeatureIdx]);
        }

        /* can return nullptr if this feature is unavailable
         * (ignored or this data provider contains only subset of features)
         */
        TMaybeData<const THashedCatValuesHolder*> GetCatFeature(ui32 catFeatureIdx) const {
            return MakeMaybeData<const THashedCatValuesHolder>(Data.CatFeatures[catFeatureIdx]);
        }

        const THashMap<ui32, TString>& GetCatFeaturesHashToString(ui32 catFeatureIdx) const {
            return (*Data.CatFeaturesHashToString)[catFeatureIdx];
        }

        /* set functions are needed for current python mutable Pool interface
           builders should prefer to set fields directly to avoid unnecessary data copying
        */

        void SetGroupIds(TConstArrayRef<TStringBuf> groupStringIds);
        void SetSubgroupIds(TConstArrayRef<TStringBuf> subgroupStringIds);

    private:
        TRawObjectsData Data;
    };


    // for use while building and storing this part in TRawObjectsDataProvider
    struct TQuantizedObjectsData {
    public:
        /* some feature holders can contain nullptr
         *  (ignored or this data provider contains only subset of features)
         */
        TVector<THolder<IQuantizedFloatValuesHolder>> FloatFeatures; // [floatFeatureIdx]
        TVector<THolder<IQuantizedCatValuesHolder>> CatFeatures; // [catFeatureIdx]

        TIntrusivePtr<TQuantizedFeaturesInfo> QuantizedFeaturesInfo;

    public:
        void Check(ui32 objectCount, const TFeaturesLayout& featuresLayout) const;

        // subsetComposition passed by pointer, because pointers are used in columns, avoid temporaries
        TQuantizedObjectsData GetSubset(const TArraySubsetIndexing<ui32>* subsetComposition) const;
    };


    class TQuantizedObjectsDataProvider : public TObjectsDataProvider {
    public:
        TQuantizedObjectsDataProvider(
            TMaybe<TObjectsGroupingPtr> objectsGrouping, // if not defined - init from groupId
            TCommonObjectsData&& commonData,
            TQuantizedObjectsData&& data,
            bool skipCheck,

            // needed for check, can pass Nothing() if skipCheck is true
            TMaybe<const TFeaturesLayout*> featuresLayout
        )
            : TObjectsDataProvider(std::move(objectsGrouping), std::move(commonData), skipCheck)
        {
            if (!skipCheck) {
                data.Check(GetObjectCount(), **featuresLayout);
            }
            Data = std::move(data);
        }

        TIntrusivePtr<TObjectsDataProvider> GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            NPar::TLocalExecutor* localExecutor
        ) const override {
            return GetSubsetImpl<TQuantizedObjectsDataProvider>(objectsGroupingSubset, localExecutor);
        }

        /* can return nullptr if this feature is unavailable
         * (ignored or this data provider contains only subset of features)
         */
        TMaybeData<const IQuantizedFloatValuesHolder*> GetFloatFeature(ui32 floatFeatureIdx) const {
            return MakeMaybeData<const IQuantizedFloatValuesHolder>(Data.FloatFeatures[floatFeatureIdx]);
        }

        /* can return nullptr if this feature is unavailable
         * (ignored or this data provider contains only subset of features)
         */
        TMaybeData<const IQuantizedCatValuesHolder*> GetCatFeature(ui32 catFeatureIdx) const {
            return MakeMaybeData<const IQuantizedCatValuesHolder>(Data.CatFeatures[catFeatureIdx]);
        }

        TIntrusivePtr<TQuantizedFeaturesInfo> GetQuantizedFeaturesInfo() const {
            return Data.QuantizedFeaturesInfo;
        }

    protected:

        // for common implementation in TQuantizedObjectsDataProvider & TQuantizedObjectsForCPUDataProvider
        template <class TTQuantizedObjectsDataProvider>
        TIntrusivePtr<TQuantizedObjectsDataProvider> GetSubsetImpl(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            NPar::TLocalExecutor* localExecutor
        ) const {
            TCommonObjectsData subsetCommonData = CommonData.GetSubset(
                objectsGroupingSubset,
                localExecutor
            );
            TQuantizedObjectsData subsetData = Data.GetSubset(subsetCommonData.SubsetIndexing.Get());

            return MakeIntrusive<TTQuantizedObjectsDataProvider>(
                objectsGroupingSubset.GetSubsetGrouping(),
                std::move(subsetCommonData),
                std::move(subsetData),
                true,
                Nothing()
            );
        }

    protected:
        TQuantizedObjectsData Data;
    };

    class TQuantizedForCPUObjectsDataProvider : public TQuantizedObjectsDataProvider {
    public:
        TQuantizedForCPUObjectsDataProvider(
            TMaybe<TObjectsGroupingPtr> objectsGrouping, // if not defined - init from groupId
            TCommonObjectsData&& commonData,
            TQuantizedObjectsData&& data,
            bool skipCheck,

            // needed for check, can pass Nothing() if skipCheck is true
            TMaybe<const TFeaturesLayout*> featuresLayout
        );

        TQuantizedForCPUObjectsDataProvider(
            TQuantizedObjectsDataProvider&& arg
        );

        TIntrusivePtr<TObjectsDataProvider> GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            NPar::TLocalExecutor* localExecutor
        ) const override {
            return GetSubsetImpl<TQuantizedForCPUObjectsDataProvider>(
                objectsGroupingSubset,
                localExecutor
            );
        }

        /* overrides base class implementation with more restricted type
         * (more efficient for CPU score calculation)
         * features guaranteed to be stored as an array of ui8
         */
        TMaybeData<const TQuantizedFloatValuesHolder*> GetFloatFeature(ui32 floatFeatureIdx) const {
            return MakeMaybeData(
                // already checked in ctor that this cast is safe
                static_cast<const TQuantizedFloatValuesHolder*>(
                    Data.FloatFeatures[floatFeatureIdx].Get()
                )
            );
        }

        /* overrides base class implementation with more restricted type
         * (more efficient for CPU score calculation)
         * features guaranteed to be stored as an array of ui32
         */
        TMaybeData<const TQuantizedCatValuesHolder*> GetCatFeature(ui32 catFeatureIdx) const {
            return MakeMaybeData(
                // already checked in ctor that this cast is safe
                static_cast<const TQuantizedCatValuesHolder*>(
                    Data.CatFeatures[catFeatureIdx].Get()
                )
            );
        }

        ui32 GetCatFeatureUniqueValuesCount(ui32 catFeatureIdx) const {
            return CatFeatureUniqueValuesCount[catFeatureIdx];
        }

    private:
        // check that additional CPU-specific constraints are respected
        void Check() const;

    private:
        // store directly instead of looking up in Data.QuantizedFeaturesInfo for runtime efficiency
        TVector<ui32> CatFeatureUniqueValuesCount; // [catFeatureIdx]
    };

}
