#pragma once

#include "columns.h"
#include "composite_columns.h"
#include "exclusive_feature_bundling.h"
#include "feature_grouping.h"
#include "feature_index.h"
#include "features_layout.h"
#include "meta_info.h"
#include "objects_grouping.h"
#include "order.h"
#include "quantized_features_info.h"
#include "util.h"

#include <catboost/private/libs/data_types/groupid.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/compression.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>
#include <catboost/libs/helpers/resource_constrained_executor.h>
#include <catboost/libs/helpers/resource_holder.h>
#include <catboost/libs/helpers/serialization.h>

#include <catboost/private/libs/options/binarization_options.h>

#include <library/cpp/binsaver/bin_saver.h>
#include <library/cpp/dbg_output/dump.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/variant.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/system/types.h>

#include <utility>


class IOutputStream;


namespace NCB {

    /* if groupIds is empty return trivial grouping
     *  checks that groupIds are consecutive
     */
    template <class TGroupIdClass>
    TObjectsGrouping CreateObjectsGroupingFromGroupIds(
        ui32 objectCount,
        TMaybeData<TConstArrayRef<TGroupIdClass>> groupIds
    ) {
        if (!groupIds) {
            return TObjectsGrouping(objectCount);
        }
        auto groupIdsData = *groupIds;

        CheckDataSize(groupIdsData.size(), (size_t)objectCount, "group Ids", false);

        return TObjectsGrouping(GroupSamples<TGroupIdClass>(groupIdsData), true);
    }

    template <class TId>
    class TMaybeStringOrNumIdColumn {
    public:
        TMaybeStringOrNumIdColumn() {}

        TMaybeStringOrNumIdColumn(TVector<TId>&& data)
            : Data(std::move(data))
            , StoreStringColumns(false)
        {}

        TMaybeStringOrNumIdColumn(TVector<TString>&& data)
            : Data(std::move(data))
            , StoreStringColumns(true)
        {}

        bool operator==(const TMaybeStringOrNumIdColumn& rhs) const {
            return (Data == rhs.Data) && (StoreStringColumns == rhs.StoreStringColumns);
        }

        void SetStoreStringColumns(bool storeStringColumns) {
            StoreStringColumns = storeStringColumns;
            if (storeStringColumns) {
                Data = TMaybeData<TVector<TString>>();
            } else {
                Data = TMaybeData<TVector<TId>>();
            }
        }

        void SetStoreStringColumnsVal(bool storeStringColumns) {
            StoreStringColumns = storeStringColumns;
        }

        size_t GetSize() const {
            if (Data.index() == 1) {
                return std::get<TMaybeData<TVector<TString>>>(Data)->size();
            } else {
                return std::get<TMaybeData<TVector<TId>>>(Data)->size();
            }
        }

        bool IsDefined() const {
            if (StoreStringColumns) {
                return GetMaybeStringData().Defined();
            } else {
                return GetMaybeNumData().Defined();
            }
        }

        const TMaybeData<TVector<TId>>& GetMaybeNumData() const {
            CB_ENSURE(!StoreStringColumns);
            return std::get<TMaybeData<TVector<TId>>>(Data);
        }

        TMaybeData<TVector<TId>>& GetMaybeNumData() {
            CB_ENSURE(!StoreStringColumns);
            return std::get<TMaybeData<TVector<TId>>>(Data);
        }

        const TMaybeData<TVector<TString>>& GetMaybeStringData() const {
            CB_ENSURE(StoreStringColumns);
            return std::get<TMaybeData<TVector<TString>>>(Data);
        }

        TMaybeData<TVector<TString>>& GetMaybeStringData() {
            CB_ENSURE(StoreStringColumns);
            return std::get<TMaybeData<TVector<TString>>>(Data);
        }

        bool StoreString() const {
            return StoreStringColumns;
        }

        SAVELOAD(Data, StoreStringColumns);

    private:
        std::variant<TMaybeData<TVector<TId>>, TMaybeData<TVector<TString>>> Data;
        bool StoreStringColumns = false;
    };

    template <class TId>
    TMaybeStringOrNumIdColumn<TId> GetSubsetFromMaybeStringOrNumIdColumn(
        const TMaybeStringOrNumIdColumn<TId>& source,
        const TArraySubsetIndexing<ui32>& subsetIndexing,
        TMaybe<NPar::ILocalExecutor*> localExecutor
    ) {
        TMaybeStringOrNumIdColumn<TId> result;
        result.SetStoreStringColumns(source.StoreString());
        if (source.StoreString()) {
            result.GetMaybeStringData() = GetSubsetOfMaybeEmpty<TString>(
                (TMaybeData<TConstArrayRef<TString>>) source.GetMaybeStringData(),
                subsetIndexing,
                localExecutor
            );
        } else {
            result.GetMaybeNumData() = GetSubsetOfMaybeEmpty<TId>(
                (TMaybeData<TConstArrayRef<TId>>) source.GetMaybeNumData(),
                subsetIndexing,
                localExecutor
            );
        }
        return result;
    }

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

        bool StoreStringColumns = false;
        TMaybeData<TVector<TString>> SampleId;
        TMaybeStringOrNumIdColumn<TSubgroupId> SubgroupIds; // [objectIdx]
        TMaybeStringOrNumIdColumn<TGroupId> GroupIds; // [objectIdx]
        TMaybeData<TVector<ui64>> Timestamp; // [objectIdx]

        /* can be empty if there's no cat features
           elements can be empty, it is allowed for some cat features to have hashed representation only
        */
        TAtomicSharedPtr<TVector<THashMap<ui32, TString>>> CatFeaturesHashToString; // [catFeatureIdx]

    public:
        void SetStoreStringColumns(bool storeStringColumns);
        bool EqualTo(const TCommonObjectsData& rhs, bool ignoreSparsity = false) const;

        // not a constructor to enable reuse of allocated data
        void PrepareForInitialization(const TDataMetaInfo& metaInfo, ui32 objectCount, ui32 prevTailCount);

        // used in DataProviderBuilder
        void SetBuildersArrayRef(
            const TDataMetaInfo& metaInfo,
            TArrayRef<TGroupId>* numGroupIdsRefPtr,
            TArrayRef<TString>* stringGroupIdsRefPtr,
            TArrayRef<TSubgroupId>* numSubgroupIdsRefPtr,
            TArrayRef<TString>* stringSubgroupIdsRefPtr
        );

        /* used in TObjectsDataProvider to avoid double checking
         * when ObjectsGrouping is created from GroupIds and GroupId's consistency is already checked
         * in this process
         */
        void CheckAllExceptGroupIds() const;

        // if objectsGrouping is defined and GroupIds are defined check that they are consistent
        void Check(TMaybe<TObjectsGroupingPtr> objectsGrouping) const;

        TCommonObjectsData GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            NPar::ILocalExecutor* localExecutor
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

        /*
         * ignoreSparsity means don't take into account whether columns are marked as either sparse or dense
         *  - only compare values
         */
        virtual bool EqualTo(const TObjectsDataProvider& rhs, bool ignoreSparsity = false) const {
            return (*ObjectsGrouping == *rhs.ObjectsGrouping) &&
                CommonData.EqualTo(rhs.CommonData, ignoreSparsity);
        }

        bool operator==(const TObjectsDataProvider& rhs) const {
            return EqualTo(rhs);
        }

        ui32 GetObjectCount() const noexcept {
            return ObjectsGrouping->GetObjectCount();
        }

        TObjectsGroupingPtr GetObjectsGrouping() const {
            return ObjectsGrouping;
        }

        /* Note that this function checks that some data is really stored as dense columns, not if
         * some features are potentially Sparse (this information is stored in FeaturesLayout)
         */
        virtual bool HasDenseData() const = 0;

        /* Note that this function checks that some data is really stored as sparse columns, not if
         * some features are potentially Sparse (this information is stored in FeaturesLayout)
         */
        virtual bool HasSparseData() const = 0;

        virtual TIntrusivePtr<TObjectsDataProvider> GetSubsetImpl(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            TMaybe<TConstArrayRef<ui32>> ignoredFeatures,
            ui64 cpuRamLimit,
            NPar::ILocalExecutor* localExecutor
        ) const = 0;

        TIntrusivePtr<TObjectsDataProvider> GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            ui64 cpuRamLimit,
            NPar::ILocalExecutor* localExecutor
        ) const {
            return GetSubsetImpl(
                objectsGroupingSubset,
                /*ignoredFeatures*/ Nothing(),
                cpuRamLimit,
                localExecutor
            );
        }

        TIntrusivePtr<TObjectsDataProvider> GetFeaturesSubset(
            const TVector<ui32>& ignoredFeatures,
            NPar::ILocalExecutor* localExecutor
        ) const;

        TIntrusivePtr<TObjectsDataProvider> Clone(NPar::ILocalExecutor* localExecutor) const;


        // The following Get* functions are common for all implementations, so they're in this base class

        TFeaturesLayoutPtr GetFeaturesLayout() const noexcept {
            return CommonData.FeaturesLayout;
        }

        EObjectsOrder GetOrder() const {
            return CommonData.Order;
        }

        TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const { // [objectIdx]
            return CommonData.GroupIds.GetMaybeNumData();
        }

        TMaybeData<TConstArrayRef<TString>> GetStringGroupIds() const { // [objectIdx]
            return CommonData.GroupIds.GetMaybeStringData();
        }

        TMaybeData<TConstArrayRef<TSubgroupId>> GetSubgroupIds() const { // [objectIdx]
            return CommonData.SubgroupIds.GetMaybeNumData();
        }

        TMaybeData<TConstArrayRef<TString>> GetStringSubgroupIds() const { // [objectIdx]
            return CommonData.SubgroupIds.GetMaybeStringData();
        }

        TMaybeData<TConstArrayRef<TString>> GetSampleIds() const noexcept { // [objectIdx]
            return CommonData.SampleId;
        }

        TMaybeData<TConstArrayRef<ui64>> GetTimestamp() const noexcept { // [objectIdx]
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

        void SetTimestamps(TConstArrayRef<ui64> timestamps); // [objectIdx]


        void SaveCommonDataNonSharedPart(IBinSaver* binSaver) const {
            CommonData.SaveNonSharedPart(binSaver);
        }

        virtual TQuantizedFeaturesInfoPtr GetQuantizedFeaturesInfo() const {
            return nullptr;
        }

    private:
        friend class TQuantizationImpl;
        template <class TTObjectsDataProvider>
        friend class TBuilderDataHelper;

    protected:
        TObjectsGroupingPtr ObjectsGrouping;
        TCommonObjectsData CommonData;
    };

    using TObjectsDataProviderPtr = TIntrusivePtr<TObjectsDataProvider>;
    using TObjectsDataProviderConstPtr = TIntrusiveConstPtr<TObjectsDataProvider>;


    // for use while building and storing this part in TRawObjectsDataProvider
    struct TRawObjectsData {
    public:
        /* some feature holders can contain nullptr
         *  (ignored or this data provider contains only subset of features)
         */
        TVector<THolder<TFloatValuesHolder>> FloatFeatures; // [floatFeatureIdx]
        TVector<THolder<THashedCatValuesHolder>> CatFeatures; // [catFeatureIdx]
        TVector<THolder<TStringTextValuesHolder>> TextFeatures; // [textFeatureIdx]
        TVector<THolder<TEmbeddingValuesHolder>> EmbeddingFeatures; // [embeddingFeatureIdx]

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
            NPar::ILocalExecutor* localExecutor
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
            TMaybe<NPar::ILocalExecutor*> localExecutor
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

        bool EqualTo(const TObjectsDataProvider& rhs, bool ignoreSparsity = false) const override {
            const auto* rhsRawObjectsData = dynamic_cast<const TRawObjectsDataProvider*>(&rhs);
            if (!rhsRawObjectsData) {
                return false;
            }
            return TObjectsDataProvider::EqualTo(rhs, ignoreSparsity) && (Data == rhsRawObjectsData->Data);
        }

        TObjectsDataProviderPtr GetSubsetImpl(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            TMaybe<TConstArrayRef<ui32>> ignoredFeatures,
            ui64 cpuRamLimit,
            NPar::ILocalExecutor* localExecutor
        ) const override;

        bool HasDenseData() const override;
        bool HasSparseData() const override;


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

        void SetFloatFeature(ui32 floatFeatureIdx, THolder<TFloatValuesHolder>&& newFeatureValues) {
            Data.FloatFeatures[floatFeatureIdx] = std::move(newFeatureValues);
        }

        /* can return nullptr if this feature is unavailable
         * (ignored or this data provider contains only subset of features)
         */
        TMaybeData<const THashedCatValuesHolder*> GetCatFeature(ui32 catFeatureIdx) const {
            return MakeMaybeData<const THashedCatValuesHolder>(Data.CatFeatures[catFeatureIdx]);
        }

        void SetCatFeature(ui32 catFeatureIdx, THolder<THashedCatValuesHolder>&& newFeatureValues) {
            Data.CatFeatures[catFeatureIdx] = std::move(newFeatureValues);
        }

        /* can return nullptr if this feature is unavailable
         * (ignored or this data provider contains only subset of features)
         */
        TMaybeData<const TStringTextValuesHolder*> GetTextFeature(ui32 textFeatureIdx) const {
            return MakeMaybeData<const TStringTextValuesHolder>(Data.TextFeatures[textFeatureIdx]);
        }

        void SetTextFeature(ui32 textFeatureIdx, THolder<TStringTextValuesHolder>&& newFeatureValues) {
            Data.TextFeatures[textFeatureIdx] = std::move(newFeatureValues);
        }

        /* can return nullptr if this feature is unavailable
         * (ignored or this data provider contains only subset of features)
         */
        TMaybeData<const TEmbeddingValuesHolder*> GetEmbeddingFeature(ui32 embeddingFeatureIdx) const {
            return MakeMaybeData<const TEmbeddingValuesHolder>(Data.EmbeddingFeatures[embeddingFeatureIdx]);
        }

        void SetEmbeddingFeature(ui32 embeddingFeatureIdx, THolder<TEmbeddingValuesHolder>&& newFeatureValues) {
            Data.EmbeddingFeatures[embeddingFeatureIdx] = std::move(newFeatureValues);
        }

        /* set functions are needed for current python mutable Pool interface
           builders should prefer to set fields directly to avoid unnecessary data copying
        */

        void SetGroupIds(TConstArrayRef<TStringBuf> groupStringIds);
        void SetSubgroupIds(TConstArrayRef<TStringBuf> subgroupStringIds);

    private:
        friend class TQuantizationImpl;
        template <class TTObjectsDataProvider>
        friend class TBuilderDataHelper;

    private:
        TData ExtractObjectData() {
            return std::move(Data);
        }

    private:
        TRawObjectsData Data;
    };

    struct TExclusiveFeatureBundlesData {
        // lookups
        TVector<TMaybe<TExclusiveBundleIndex>> FlatFeatureIndexToBundlePart; // [flatFeatureIdx]

        TVector<TExclusiveFeaturesBundle> MetaData; // [bundleIdx]

        /* supported TExclusiveFeatureBundleArrayHolder types are TExclusiveFeatureBundleArrayHolder
         * and TExclusiveFeatureBundleSparseArrayHolder
         */
        TVector<THolder<IExclusiveFeatureBundleArray>> SrcData; // [bundleIdx]

    public:
        TExclusiveFeatureBundlesData() = default;

        // SrcData is not initialized here - create separately
        TExclusiveFeatureBundlesData(
            const NCB::TFeaturesLayout& featuresLayout,
            TVector<TExclusiveFeaturesBundle>&& metaData
        );

        // Execute tasks added to resourceConstrainedExecutor to complete GetSubset operation
        void GetSubsetWithScheduling(
            const TFeaturesArraySubsetIndexing* subsetIndexing,

            // needed only for sparse features
            const TMaybe<TFeaturesArraySubsetInvertedIndexing>& subsetInvertedIndexing,
            TResourceConstrainedExecutor* resourceConstrainedExecutor,
            TExclusiveFeatureBundlesData* subsetData
        ) const;

        void Save(NPar::ILocalExecutor* localExecutor, IBinSaver* binSaver) const;
        void Load(const TArraySubsetIndexing<ui32>* subsetIndexing, IBinSaver* binSaver);
    };


    struct TFeatureGroupsData {
        TVector<TMaybe<TFeaturesGroupIndex>> FlatFeatureIndexToGroupPart;
        TVector<TFeaturesGroup> MetaData;
        TVector<THolder<IFeaturesGroupArray>> SrcData; // [groupIdx]

    public:
        TFeatureGroupsData() = default;

        // SrcData is not initialized here - create separately
        TFeatureGroupsData(
            const NCB::TFeaturesLayout& featuresLayout,
            TVector<TFeaturesGroup>&& metaData
        );

        // Execute tasks added to resourceConstrainedExecutor to complete GetSubset operation
        void GetSubsetWithScheduling(
            const TFeaturesArraySubsetIndexing* subsetIndexing,

            // needed only for sparse features
            const TMaybe<TFeaturesArraySubsetInvertedIndexing>& subsetInvertedIndexing,
            TResourceConstrainedExecutor* resourceConstrainedExecutor,
            TFeatureGroupsData* subsetData
        ) const;

        void Save(NPar::ILocalExecutor* localExecutor, IBinSaver* binSaver) const;
        void Load(const TArraySubsetIndexing<ui32>* subsetIndexing, IBinSaver* binSaver);
    };


    struct TPackedBinaryFeaturesData {
        // lookups
        TVector<TMaybe<TPackedBinaryIndex>> FlatFeatureIndexToPackedBinaryIndex; // [flatFeatureIdx]
        TVector<TFeatureIdxWithType> PackedBinaryToSrcIndex; // [linearPackedBinaryIndex]

        TVector<THolder<IBinaryPacksArray>> SrcData; // [packIdx][objectIdx][bitIdx]

    public:
        TPackedBinaryFeaturesData() = default;

        // does not init data in SrcData elements, it has to be filled later if necessary
        TPackedBinaryFeaturesData(
            const TFeaturesLayout& featuresLayout,
            const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
            const TExclusiveFeatureBundlesData& exclusiveFeatureBundlesData,
            bool dontPack = false // set true to disable binary features packing
        );

        // Execute tasks added to resourceConstrainedExecutor to complete GetSubset operation
        void GetSubsetWithScheduling(
            const TFeaturesArraySubsetIndexing* subsetIndexing,

            // needed only for sparse features
            const TMaybe<TFeaturesArraySubsetInvertedIndexing>& subsetInvertedIndexing,
            TResourceConstrainedExecutor* resourceConstrainedExecutor,
            TPackedBinaryFeaturesData* subsetData
        ) const;

        void Save(NPar::ILocalExecutor* localExecutor, IBinSaver* binSaver) const;
        void Load(const TArraySubsetIndexing<ui32>* subsetIndexing, IBinSaver* binSaver);
    };

    // for use while building and storing this part in TQuantizedObjectsDataProvider
    struct TQuantizedObjectsData : public TMoveOnly {
    public:
        /* some feature holders can contain nullptr
         *  (ignored or this data provider contains only subset of features)
         */
        TVector<THolder<IQuantizedFloatValuesHolder>> FloatFeatures; // [floatFeatureIdx]
        TVector<THolder<IQuantizedCatValuesHolder>> CatFeatures; // [catFeatureIdx]
        TVector<THolder<TTokenizedTextValuesHolder>> TextFeatures; // [textFeatureIdx]
        TVector<THolder<TEmbeddingValuesHolder>> EmbeddingFeatures; // [EmbeddingFeatureIdx]

        TPackedBinaryFeaturesData PackedBinaryFeaturesData;
        TExclusiveFeatureBundlesData ExclusiveFeatureBundlesData;
        TFeatureGroupsData FeaturesGroupsData;

        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;

        mutable TMaybe<ui32> CachedFeaturesCheckSum;

    public:
        // ignores QuantizedFeaturesInfo, compares only features data
        bool operator==(const TQuantizedObjectsData& rhs) const;

        // not a constructor to enable reuse of allocated data
        void PrepareForInitialization(
            const TDataMetaInfo& metaInfo,
            const NCatboostOptions::TBinarizationOptions& binarizationOptions,
            const TMap<ui32, NCatboostOptions::TBinarizationOptions>& perFloatFeatureQuantization
        );

        void Check(
            ui32 objectCount,
            const TFeaturesLayout& featuresLayout,
            NPar::ILocalExecutor* localExecutor
        ) const;

        void Load(
            const TArraySubsetIndexing<ui32>* subsetIndexing,
            const TFeaturesLayout& featuresLayout,
            TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
            IBinSaver* binSaver
        );
        void SaveNonSharedPart(const TFeaturesLayout& featuresLayout, IBinSaver* binSaver) const;
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
            TMaybe<NPar::ILocalExecutor*> localExecutor
        )
            : TObjectsDataProvider(std::move(objectsGrouping), std::move(commonData), skipCheck)
        {
            if (!skipCheck) {
                data.Check(GetObjectCount(), *GetFeaturesLayout(), *localExecutor);
            }
            Data = std::move(data);
            CatFeatureUniqueValuesCounts.yresize(Data.CatFeatures.size());
            for (auto catFeatureIdx : xrange(Data.CatFeatures.size())) {
                CatFeatureUniqueValuesCounts[catFeatureIdx] =
                    Data.QuantizedFeaturesInfo->GetUniqueValuesCounts(TCatFeatureIdx(catFeatureIdx));
            }
        }

        bool EqualTo(const TObjectsDataProvider& rhs, bool ignoreSparsity = false) const override {
            const auto* rhsQuantizedObjectsData = dynamic_cast<const TQuantizedObjectsDataProvider*>(&rhs);
            if (!rhsQuantizedObjectsData) {
                return false;
            }
            return TObjectsDataProvider::EqualTo(rhs, ignoreSparsity) &&
                (Data == rhsQuantizedObjectsData->Data);
        }

        TObjectsDataProviderPtr GetSubsetImpl(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            TMaybe<TConstArrayRef<ui32>> ignoredFeatures,
            ui64 cpuRamLimit,
            NPar::ILocalExecutor* localExecutor
        ) const override;

        bool HasDenseData() const override;
        bool HasSparseData() const override;

        const TFeaturesArraySubsetIndexing& GetFeaturesArraySubsetIndexing() const {
            return *CommonData.SubsetIndexing;
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

        /* can return nullptr if this feature is unavailable
         * (ignored or this data provider contains only subset of features)
         */
        TMaybeData<const TTokenizedTextValuesHolder*> GetTextFeature(ui32 textFeatureIdx) const {
            return MakeMaybeData<const TTokenizedTextValuesHolder>(Data.TextFeatures[textFeatureIdx]);
        }

        TMaybeData<const TEmbeddingValuesHolder*> GetEmbeddingFeature(ui32 embeddingFeatureIdx) const {
            return MakeMaybeData<const TEmbeddingValuesHolder>(Data.EmbeddingFeatures[embeddingFeatureIdx]);
        }

        TQuantizedFeaturesInfoPtr GetQuantizedFeaturesInfo() const override {
            return Data.QuantizedFeaturesInfo;
        }

        // result is TQuantizedFloatValuesHolder or TQuantizedFloatSparseValuesHolder
        TMaybeData<const IQuantizedFloatValuesHolder*> GetNonPackedFloatFeature(ui32 floatFeatureIdx) const {
            CheckFeatureIsNotInAggregated(EFeatureType::Float, "Float", floatFeatureIdx);
            return MakeMaybeData(Data.FloatFeatures[floatFeatureIdx].Get());
        }

        // result is TQuantizedCatValuesHolder or TQuantizedCatSparseValuesHolder
        TMaybeData<const IQuantizedCatValuesHolder*> GetNonPackedCatFeature(ui32 catFeatureIdx) const {
            CheckFeatureIsNotInAggregated(EFeatureType::Categorical, "Cat", catFeatureIdx);
            return MakeMaybeData(Data.CatFeatures[catFeatureIdx].Get());
        }

        TCatFeatureUniqueValuesCounts GetCatFeatureUniqueValuesCounts(ui32 catFeatureIdx) const {
            return CatFeatureUniqueValuesCounts[catFeatureIdx];
        }


        size_t GetPackedBinaryFeaturesSize() const {
            return Data.PackedBinaryFeaturesData.PackedBinaryToSrcIndex.size();
        }

        size_t GetBinaryFeaturesPacksSize() const {
            return Data.PackedBinaryFeaturesData.SrcData.size();
        }

        const IBinaryPacksArray& GetBinaryFeaturesPack(ui32 packIdx) const {
            return *(Data.PackedBinaryFeaturesData.SrcData[packIdx]);
        }

        TMaybe<TPackedBinaryIndex> GetFloatFeatureToPackedBinaryIndex(TFloatFeatureIdx floatFeatureIdx) const {
            return GetFeatureToPackedBinaryIndex(floatFeatureIdx);
        }

        TMaybe<TPackedBinaryIndex> GetCatFeatureToPackedBinaryIndex(TCatFeatureIdx catFeatureIdx) const {
            return GetFeatureToPackedBinaryIndex(catFeatureIdx);
        }

        template <EFeatureType FeatureType>
        inline TMaybe<TPackedBinaryIndex> GetFeatureToPackedBinaryIndex(TFeatureIdx<FeatureType> featureIdx) const {
            const ui32 flatFeatureIdx = GetFeaturesLayout()->GetExternalFeatureIdx(*featureIdx, FeatureType);
            return Data.PackedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex[flatFeatureIdx];
        }

        template <EFeatureType FeatureType>
        bool IsFeaturePackedBinary(TFeatureIdx<FeatureType> featureIdx) const {
            return GetFeatureToPackedBinaryIndex(featureIdx).Defined();
        }

        TFeatureIdxWithType GetPackedBinaryFeatureSrcIndex(
            TPackedBinaryIndex packedBinaryIndex
        ) const {
            return Data.PackedBinaryFeaturesData.PackedBinaryToSrcIndex[packedBinaryIndex.GetLinearIdx()];
        }

        ui32 CalcFeaturesCheckSum(NPar::ILocalExecutor* localExecutor) const;

        void CheckCPUTrainCompatibility() const;

        /* needed for effective calculation with Permutation blocks on CPU
         * sparse data is unaffected
         */
        void EnsureConsecutiveIfDenseFeaturesData(NPar::ILocalExecutor* localExecutor);

        size_t GetExclusiveFeatureBundlesSize() const {
            return Data.ExclusiveFeatureBundlesData.MetaData.size();
        }

        TConstArrayRef<TExclusiveFeaturesBundle> GetExclusiveFeatureBundlesMetaData() const {
            return Data.ExclusiveFeatureBundlesData.MetaData;
        }

        const IExclusiveFeatureBundleArray& GetExclusiveFeaturesBundle(ui32 bundleIdx) const {
            return *Data.ExclusiveFeatureBundlesData.SrcData[bundleIdx];
        }

        TMaybe<TExclusiveBundleIndex> GetFloatFeatureToExclusiveBundleIndex(
            TFloatFeatureIdx floatFeatureIdx
        ) const {
            return GetFeatureToExclusiveBundleIndex(floatFeatureIdx);
        }

        TMaybe<TExclusiveBundleIndex> GetCatFeatureToExclusiveBundleIndex(TCatFeatureIdx catFeatureIdx) const {
            return GetFeatureToExclusiveBundleIndex(catFeatureIdx);
        }

        template <EFeatureType FeatureType>
        inline TMaybe<TExclusiveBundleIndex> GetFeatureToExclusiveBundleIndex(
            TFeatureIdx<FeatureType> featureIdx
        ) const {
            const ui32 flatFeatureIdx = GetFeaturesLayout()->GetExternalFeatureIdx(*featureIdx, FeatureType);
            return Data.ExclusiveFeatureBundlesData.FlatFeatureIndexToBundlePart[flatFeatureIdx];
        }

        template <EFeatureType FeatureType>
        bool IsFeatureInExclusiveBundle(TFeatureIdx<FeatureType> featureIdx) const {
            return GetFeatureToExclusiveBundleIndex(featureIdx).Defined();
        }

        size_t GetFeaturesGroupsSize() const {
            return Data.FeaturesGroupsData.MetaData.size();
        }

        TConstArrayRef<TFeaturesGroup> GetFeaturesGroupsMetaData() const {
            return Data.FeaturesGroupsData.MetaData;
        }

        const TFeaturesGroup& GetFeaturesGroupMetaData(ui32 groupIdx) const {
            return Data.FeaturesGroupsData.MetaData[groupIdx];
        }

        const IFeaturesGroupArray& GetFeaturesGroup(ui32 groupIdx) const {
            return *Data.FeaturesGroupsData.SrcData[groupIdx];
        }

        TMaybe<TFeaturesGroupIndex> GetFloatFeatureToFeaturesGroupIndex(TFloatFeatureIdx floatFeatureIdx) const {
            return GetFeatureToFeaturesGroupIndex(floatFeatureIdx);
        }

        TMaybe<TFeaturesGroupIndex> GetCatFeatureToFeaturesGroupIndex(TCatFeatureIdx catFeatureIdx) const {
            return GetFeatureToFeaturesGroupIndex(catFeatureIdx);
        }

        template <EFeatureType FeatureType>
        inline TMaybe<TFeaturesGroupIndex> GetFeatureToFeaturesGroupIndex(TFeatureIdx<FeatureType> featureIdx) const {
            const ui32 flatFeatureIdx = GetFeaturesLayout()->GetExternalFeatureIdx(*featureIdx, FeatureType);
            return Data.FeaturesGroupsData.FlatFeatureIndexToGroupPart[flatFeatureIdx];
        }

    protected:
        friend class TObjectsSerialization;
        template <class TTObjectsDataProvider>
        friend class TBuilderDataHelper;

    protected:
        void SaveDataNonSharedPart(IBinSaver* binSaver) const {
            Data.SaveNonSharedPart(*GetFeaturesLayout(), binSaver);
        }

        TQuantizedObjectsData ExtractObjectData() {
            return std::move(Data);
        }

    private:
        void CheckFeatureIsNotInAggregated(
            EFeatureType featureType,
            const TStringBuf featureTypeName,
            ui32 perTypeFeatureIdx
        ) const;

    private:
        TQuantizedObjectsData Data;

        // store directly instead of looking up in Data.QuantizedFeaturesInfo for runtime efficiency
        TVector<TCatFeatureUniqueValuesCounts> CatFeatureUniqueValuesCounts; // [catFeatureIdx]
    };


    void DbgDumpQuantizedFeatures(
        const TQuantizedObjectsDataProvider& quantizedObjectsDataProvider,
        IOutputStream* out
    );

}

template <>
struct TDumper<TMaybe<NCB::TPackedBinaryIndex>> {
    template <class S>
    static inline void Dump(S& s, const TMaybe<NCB::TPackedBinaryIndex>& maybePackedBinaryIndex) {
        if (maybePackedBinaryIndex) {
            s << DbgDump(*maybePackedBinaryIndex);
        } else {
            s << '-';
        }
    }
};


namespace NCB {
    TString DbgDumpMetaData(const TPackedBinaryFeaturesData& packedBinaryFeaturesData);

    using TQuantizedObjectsDataProviderPtr = TIntrusivePtr<TQuantizedObjectsDataProvider>;


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

            TQuantizedFeaturesInfoPtr quantizedFeaturesInfo;
            AddWithShared(binSaver, &quantizedFeaturesInfo);

            typename TObjectsDataProviderType::TData quantizedObjectsData;
            quantizedObjectsData.Load(
                commonObjectsData.SubsetIndexing.Get(),
                *featuresLayout,
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

            TQuantizedFeaturesInfoPtr quantizedFeaturesInfo = objectsData.GetQuantizedFeaturesInfo();
            AddWithShared(binSaver, &quantizedFeaturesInfo);

            objectsData.SaveDataNonSharedPart(binSaver);
        }
    };

    THashMap<ui32, TString> MergeCatFeaturesHashToString(const TObjectsDataProvider& objectsData);

}
