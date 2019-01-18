#pragma once

#include "columns.h"
#include "features_layout.h"
#include "meta_info.h"
#include "objects_grouping.h"
#include "order.h"
#include "quantized_features_info.h"
#include "util.h"

#include <catboost/libs/data_types/groupid.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/resource_holder.h>
#include <catboost/libs/helpers/serialization.h>

#include <catboost/libs/options/binarization_options.h>

#include <library/binsaver/bin_saver.h>
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

        TFeaturesLayoutPtr FeaturesLayout;

        /* this dataset can be a view from a bigger objects dataset
           this field provides this data to columns in derived classes
        */
        TAtomicSharedPtr<TArraySubsetIndexing<ui32>> SubsetIndexing;

        EObjectsOrder Order = EObjectsOrder::Undefined;

        TMaybeData<TVector<TGroupId>> GroupIds; // [objectIdx]
        TMaybeData<TVector<TSubgroupId>> SubgroupIds; // [objectIdx]
        TMaybeData<TVector<ui64>> Timestamp; // [objectIdx]

        /* can be empty if there's no cat features
           elements can be empty, it is allowed for some cat features to have hashed representation only
        */
        TAtomicSharedPtr<TVector<THashMap<ui32, TString>>> CatFeaturesHashToString; // [catFeatureIdx]

    public:
        bool operator==(const TCommonObjectsData& rhs) const;

        // not a constructor to enable reuse of allocated data
        void PrepareForInitialization(const TDataMetaInfo& metaInfo, ui32 objectCount, ui32 prevTailCount);

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

        void Load(TFeaturesLayoutPtr featuresLayout, ui32 objectCount, IBinSaver* binSaver);
        void SaveNonSharedPart(IBinSaver* binSaver) const;
    };


    class TObjectsDataProvider : public TThrRefBase {
    public:
        TObjectsDataProvider(
            // if not defined - call CreateObjectsGroupingFromGroupIds
            TMaybe<TObjectsGroupingPtr> objectsGrouping,
            TCommonObjectsData&& commonData,
            bool skipCheck
        );

        virtual bool operator==(const TObjectsDataProvider& rhs) const {
            return (*ObjectsGrouping == *rhs.ObjectsGrouping) && (CommonData == rhs.CommonData);
        }

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
         * GetFeaturesLayout, GetGroupIds, GetSubgroupIds are common for all implementations,
         *  so they're in this base class
         */

        TFeaturesLayoutPtr GetFeaturesLayout() const {
            return CommonData.FeaturesLayout;
        }

        EObjectsOrder GetOrder() const {
            return CommonData.Order;
        }

        TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const { // [objectIdx]
            return CommonData.GroupIds;
        }

        TMaybeData<TConstArrayRef<TSubgroupId>> GetSubgroupIds() const { // [objectIdx]
            return CommonData.SubgroupIds;
        }

        TMaybeData<TConstArrayRef<ui64>> GetTimestamp() const { // [objectIdx]
            return CommonData.Timestamp;
        }

        const THashMap<ui32, TString>& GetCatFeaturesHashToString(ui32 catFeatureIdx) const {
            return (*CommonData.CatFeaturesHashToString)[catFeatureIdx];
        }

        /* set functions are needed for current python mutable Pool interface
           builders should prefer to set fields directly to avoid unnecessary data copying
        */

        // updates ObjectsGrouping
        void SetGroupIds(TConstArrayRef<TGroupId> groupIds); // [objectIdx]

        void SetSubgroupIds(TConstArrayRef<TSubgroupId> subgroupIds); // [objectIdx]


        void SaveCommonDataNonSharedPart(IBinSaver* binSaver) const {
            CommonData.SaveNonSharedPart(binSaver);
        }

    private:
        friend class TQuantizationImpl;
        friend class TRawBuilderDataHelper;

    protected:
        TObjectsGroupingPtr ObjectsGrouping;
        TCommonObjectsData CommonData;
    };

    using TObjectsDataProviderPtr = TIntrusivePtr<TObjectsDataProvider>;


    // for use while building and storing this part in TRawObjectsDataProvider
    struct TRawObjectsData {
    public:
        /* some feature holders can contain nullptr
         *  (ignored or this data provider contains only subset of features)
         */
        TVector<THolder<TFloatValuesHolder>> FloatFeatures; // [floatFeatureIdx]
        TVector<THolder<THashedCatValuesHolder>> CatFeatures; // [catFeatureIdx]

    public:
        bool operator==(const TRawObjectsData& rhs) const;

        // not a constructor to enable reuse of allocated data
        void PrepareForInitialization(const TDataMetaInfo& metaInfo);

        // TODO(akhropov): Is cat features hashes check too expensive/should be optional for release?
        void Check(
            ui32 objectCount,
            const TFeaturesLayout& featuresLayout,

            // can be nullptr is there's no categorical features
            const TVector<THashMap<ui32, TString>>* catFeaturesHashToString,
            NPar::TLocalExecutor* localExecutor
        ) const;
    };

    class TRawObjectsDataProvider : public TObjectsDataProvider {
    public:
        using TData = TRawObjectsData;

    public:
        TRawObjectsDataProvider(
            TMaybe<TObjectsGroupingPtr> objectsGrouping, // if not defined - init from groupId
            TCommonObjectsData&& commonData,
            TRawObjectsData&& data,

            bool skipCheck,

            // needed for check parallelization, can pass Nothing() if skipCheck is true
            TMaybe<NPar::TLocalExecutor*> localExecutor
        )
            : TObjectsDataProvider(std::move(objectsGrouping), std::move(commonData), skipCheck)
        {
            if (!skipCheck) {
                data.Check(
                    GetObjectCount(),
                    *GetFeaturesLayout(),
                    CommonData.CatFeaturesHashToString.Get(),
                    *localExecutor
                );
            }
            Data = std::move(data);
        }

        bool operator==(const TObjectsDataProvider& rhs) const override {
            const auto* rhsRawObjectsData = dynamic_cast<const TRawObjectsDataProvider*>(&rhs);
            if (!rhsRawObjectsData) {
                return false;
            }
            return TObjectsDataProvider::operator==(rhs) && (Data == rhsRawObjectsData->Data);
        }

        TObjectsDataProviderPtr GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            NPar::TLocalExecutor* localExecutor
        ) const override;

        // needed for low-level optimizations in CPU applying code
        const TFeaturesArraySubsetIndexing& GetFeaturesArraySubsetIndexing() const {
            return *CommonData.SubsetIndexing;
        }

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

        /* set functions are needed for current python mutable Pool interface
           builders should prefer to set fields directly to avoid unnecessary data copying
        */

        void SetGroupIds(TConstArrayRef<TStringBuf> groupStringIds);
        void SetSubgroupIds(TConstArrayRef<TStringBuf> subgroupStringIds);

        /* old TPool format (for cat feature  categ features hashes reinterpreted as float,
         * for compatibility with old code only
         */
        TVector<float> GetFeatureDataOldFormat(ui32 flatFeatureIdx) const;

    private:
        friend class TQuantizationImpl;
        friend class TRawBuilderDataHelper;

    private:
        TRawObjectsData Data;
    };

    // for use while building and storing this part in TQuantizedObjectsDataProvider
    struct TQuantizedObjectsData {
    public:
        /* some feature holders can contain nullptr
         *  (ignored or this data provider contains only subset of features)
         */
        TVector<THolder<IQuantizedFloatValuesHolder>> FloatFeatures; // [floatFeatureIdx]
        TVector<THolder<IQuantizedCatValuesHolder>> CatFeatures; // [catFeatureIdx]

        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;

    public:
        bool operator==(const TQuantizedObjectsData& rhs) const;

        // not a constructor to enable reuse of allocated data
        void PrepareForInitialization(
            const TDataMetaInfo& metaInfo,
            const NCatboostOptions::TBinarizationOptions& binarizationOptions
        );

        void Check(
            ui32 objectCount,
            const TFeaturesLayout& featuresLayout,
            NPar::TLocalExecutor* localExecutor
        ) const;

        // subsetComposition passed by pointer, because pointers are used in columns, avoid temporaries
        TQuantizedObjectsData GetSubset(const TArraySubsetIndexing<ui32>* subsetComposition) const;

        void Load(
            const TArraySubsetIndexing<ui32>* subsetIndexing,
            TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
            IBinSaver* binSaver
        );
        void SaveNonSharedPart(IBinSaver* binSaver) const;
    };

    using TRawObjectsDataProviderPtr = TIntrusivePtr<TRawObjectsDataProvider>;


    class TQuantizedObjectsDataProvider : public TObjectsDataProvider {
    public:
        using TData = TQuantizedObjectsData;

    public:
        TQuantizedObjectsDataProvider(
            TMaybe<TObjectsGroupingPtr> objectsGrouping, // if not defined - init from groupId
            TCommonObjectsData&& commonData,
            TQuantizedObjectsData&& data,
            bool skipCheck,

            // needed for check, can pass Nothing() if skipCheck is true
            TMaybe<NPar::TLocalExecutor*> localExecutor
        )
            : TObjectsDataProvider(std::move(objectsGrouping), std::move(commonData), skipCheck)
        {
            if (!skipCheck) {
                data.Check(GetObjectCount(), *GetFeaturesLayout(), *localExecutor);
            }
            Data = std::move(data);
        }

        bool operator==(const TObjectsDataProvider& rhs) const override {
            const auto* rhsQuantizedObjectsData = dynamic_cast<const TQuantizedObjectsDataProvider*>(&rhs);
            if (!rhsQuantizedObjectsData) {
                return false;
            }
            return TObjectsDataProvider::operator==(rhs) && (Data == rhsQuantizedObjectsData->Data);
        }

        TObjectsDataProviderPtr GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            NPar::TLocalExecutor* localExecutor
        ) const override {
            return GetSubsetImpl<TQuantizedObjectsDataProvider>(
                objectsGroupingSubset,
                localExecutor
            );
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

        TQuantizedFeaturesInfoPtr GetQuantizedFeaturesInfo() const {
            return Data.QuantizedFeaturesInfo;
        }

        ui32 CalcFeaturesCheckSum(NPar::TLocalExecutor* localExecutor) const;

    protected:
        friend class TObjectsSerialization;

    protected:
        void SaveDataNonSharedPart(IBinSaver* binSaver) const {
            Data.SaveNonSharedPart(binSaver);
        }

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

    using TQuantizedObjectsDataProviderPtr = TIntrusivePtr<TQuantizedObjectsDataProvider>;


    class TQuantizedForCPUObjectsDataProvider : public TQuantizedObjectsDataProvider {
    public:
        TQuantizedForCPUObjectsDataProvider(
            TMaybe<TObjectsGroupingPtr> objectsGrouping, // if not defined - init from groupId
            TCommonObjectsData&& commonData,
            TQuantizedObjectsData&& data,
            bool skipCheck,

            // needed for check, can pass Nothing() if skipCheck is true
            TMaybe<NPar::TLocalExecutor*> localExecutor
        );

        TQuantizedForCPUObjectsDataProvider(
            TQuantizedObjectsDataProvider&& arg
        );

        TObjectsDataProviderPtr GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            NPar::TLocalExecutor* localExecutor
        ) const override {
            return GetSubsetImpl<TQuantizedForCPUObjectsDataProvider>(
                objectsGroupingSubset,
                localExecutor
            );
        }

        // needed for effective calculation with Permutation blocks on CPU
        void EnsureConsecutiveFeaturesData(NPar::TLocalExecutor* localExecutor);

        // needed for low-level optimizations in CPU training code
        const TFeaturesArraySubsetIndexing& GetFeaturesArraySubsetIndexing() const {
            return *CommonData.SubsetIndexing;
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

        // low-level function, data is without subset indexing, apply external subset indexing!
        const ui8* GetFloatFeatureRawSrcData(ui32 floatFeatureIdx) const {
            return *((*GetFloatFeature(floatFeatureIdx))->GetArrayData().GetSrc());
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

        // low-level function, data is without subset indexing, apply external subset indexing!
        const ui32* GetCatFeatureRawSrcData(ui32 catFeatureIdx) const {
            return *((*GetCatFeature(catFeatureIdx))->GetArrayData().GetSrc());
        }

        TCatFeatureUniqueValuesCounts GetCatFeatureUniqueValuesCounts(ui32 catFeatureIdx) const {
            return CatFeatureUniqueValuesCounts[catFeatureIdx];
        }

    private:
        // check that additional CPU-specific constraints are respected
        void Check() const;

    private:
        // store directly instead of looking up in Data.QuantizedFeaturesInfo for runtime efficiency
        TVector<TCatFeatureUniqueValuesCounts> CatFeatureUniqueValuesCounts; // [catFeatureIdx]
    };


    // util function for commonly used functionality
    template <class TBase>
    inline TConstPtrArraySubset<typename TBase::TValueType> SubsetWithAlternativeIndexing(
        TMaybeData<const TCompressedValuesHolderImpl<TBase>*> featureData,
        const TFeaturesArraySubsetIndexing* alternativeIndexing
    ) {
        return TConstPtrArraySubset<typename TBase::TValueType>(
            (*featureData)->GetArrayData().GetSrc(),
            alternativeIndexing
        );
    }


    // needed to make friends with TObjectsDataProvider s
    class TObjectsSerialization {
    public:
        template <class TObjectsDataProviderType>
        static void Load(
            TFeaturesLayoutPtr featuresLayout,
            TObjectsGroupingPtr objectsGrouping,
            IBinSaver* binSaver,
            TIntrusivePtr<TObjectsDataProviderType>* objectsData
        ) {
            TCommonObjectsData commonObjectsData;
            commonObjectsData.Load(featuresLayout, objectsGrouping->GetObjectCount(), binSaver);
            TQuantizedFeaturesInfoPtr quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *featuresLayout,
                TConstArrayRef<ui32>(),
                NCatboostOptions::TBinarizationOptions(),
                false
            );
            quantizedFeaturesInfo->LoadNonSharedPart(binSaver);
            TQuantizedObjectsData quantizedObjectsData;
            quantizedObjectsData.Load(
                commonObjectsData.SubsetIndexing.Get(),
                quantizedFeaturesInfo,
                binSaver
            );
            *objectsData = MakeIntrusive<TObjectsDataProviderType>(
                objectsGrouping,
                std::move(commonObjectsData),
                std::move(quantizedObjectsData),
                true,
                Nothing()
            );
        }

        template <class TObjectsDataProviderType>
        static void SaveNonSharedPart(const TObjectsDataProviderType& objectsData, IBinSaver* binSaver) {
            objectsData.SaveCommonDataNonSharedPart(binSaver);
            objectsData.GetQuantizedFeaturesInfo()->SaveNonSharedPart(binSaver);
            objectsData.SaveDataNonSharedPart(binSaver);
        }
    };

    THashMap<ui32, TString> MergeCatFeaturesHashToString(const TObjectsDataProvider& objectsData);

}
