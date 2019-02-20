#pragma once

#include "meta_info.h"
#include "objects_grouping.h"
#include "objects.h"
#include "target.h"

#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/libs/helpers/serialization.h>

#include <library/binsaver/bin_saver.h>
#include <library/dbg_output/dump.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/system/yassert.h>


namespace NCB {

    template <class TObjectsData>
    struct TBuilderData {
        TDataMetaInfo MetaInfo;
        TRawTargetData TargetData;
        TCommonObjectsData CommonObjectsData;
        TObjectsData ObjectsData;
    };

    using TRawBuilderData = TBuilderData<TRawObjectsData>;
    using TQuantizedBuilderData = TBuilderData<TQuantizedObjectsData>;
    using TQuantizedForCPUBuilderData = TBuilderData<TQuantizedForCPUObjectsData>;

    TQuantizedBuilderData CastToBase(TQuantizedForCPUBuilderData&& builderData);


    template <class TTObjectsDataProvider>
    class TDataProviderTemplate : public TThrRefBase {
    public:
        TDataMetaInfo MetaInfo;
        TIntrusivePtr<TTObjectsDataProvider> ObjectsData;
        TObjectsGroupingPtr ObjectsGrouping;
        TRawTargetDataProvider RawTargetData;

    public:
        TDataProviderTemplate(
            TDataMetaInfo&& metaInfo,
            TIntrusivePtr<TTObjectsDataProvider> objectsData,
            TObjectsGroupingPtr objectsGrouping,
            TRawTargetDataProvider&& rawTargetData
        )
            : MetaInfo(std::move(metaInfo))
            , ObjectsData(std::move(objectsData))
            , ObjectsGrouping(objectsGrouping)
            , RawTargetData(std::move(rawTargetData))
        {}

        bool operator==(const TDataProviderTemplate& rhs) const {
            return (MetaInfo == rhs.MetaInfo) && (*ObjectsData == *rhs.ObjectsData) &&
                (*ObjectsGrouping == *rhs.ObjectsGrouping) && (RawTargetData == rhs.RawTargetData);
        }


        /*
         * if can be cast as TDataProviderTemplate<TNewObjectsDataProvider> move to object of that type and return
         * (but only if not shared - check it before calling this function
         *  or call immediately after creation)
         */
        template <class TNewObjectsDataProvider>
        TIntrusivePtr<TDataProviderTemplate<TNewObjectsDataProvider>> CastMoveTo() {
            TNewObjectsDataProvider* newObjectsDataProvider = dynamic_cast<TNewObjectsDataProvider*>(
                ObjectsData.Get()
            );
            if (!newObjectsDataProvider) {
                return nullptr;
            } else {
                CB_ENSURE_INTERNAL(RefCount() == 1, "Can't move from shared object");
                return MakeIntrusive<TDataProviderTemplate<TNewObjectsDataProvider>>(
                    std::move(MetaInfo),
                    TIntrusivePtr<TNewObjectsDataProvider>(newObjectsDataProvider),
                    ObjectsGrouping,
                    std::move(RawTargetData)
                );
            }
        }


        TIntrusivePtr<TDataProviderTemplate> GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            NPar::TLocalExecutor* localExecutor
        ) const {
            TVector<std::function<void()>> tasks;

            TIntrusivePtr<TTObjectsDataProvider> objectsDataSubset;

            tasks.emplace_back(
                [&, this]() {
                    auto baseObjectsDataSubset = ObjectsData->GetSubset(
                        objectsGroupingSubset,
                        localExecutor
                    );
                    objectsDataSubset = dynamic_cast<TTObjectsDataProvider*>(baseObjectsDataSubset.Get());
                    Y_VERIFY(objectsDataSubset);
                }
            );

            // TMaybe - to defer initialization to async task
            TMaybe<TRawTargetDataProvider> rawTargetDataSubset;

            tasks.emplace_back(
                [&, this]() {
                    rawTargetDataSubset = MakeMaybe<TRawTargetDataProvider>(
                        RawTargetData.GetSubset(
                            objectsGroupingSubset,
                            localExecutor
                        )
                    );
                }
            );

            ExecuteTasksInParallel(&tasks, localExecutor);


            return MakeIntrusive<TDataProviderTemplate>(
                TDataMetaInfo(MetaInfo), // assuming copying it is not very expensive
                objectsDataSubset,
                objectsDataSubset->GetObjectsGrouping(),
                std::move(*rawTargetDataSubset)
            );
        }

        // for Cython
        TIntrusivePtr<TDataProviderTemplate> GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            int threadCount
        ) const {
            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(threadCount);
            return GetSubset(objectsGroupingSubset, &localExecutor);
        }

        // ObjectsGrouping->GetObjectCount() used a lot, so make it a member here
        ui32 GetObjectCount() const {
            return ObjectsGrouping->GetObjectCount();
        }

        // Set* methods needed for python-package

        void SetBaseline(TBaselineArrayRef baseline) { // [approxIdx][objectIdx]
            RawTargetData.SetBaseline(baseline);
            MetaInfo.BaselineCount = baseline.size();
        }

        void SetGroupIds(TConstArrayRef<TGroupId> groupIds) {
            ObjectsData->SetGroupIds(groupIds);
            ObjectsGrouping = ObjectsData->GetObjectsGrouping();
            RawTargetData.SetObjectsGrouping(ObjectsGrouping);
            MetaInfo.HasGroupId = true;
        }

        void SetGroupWeights(TConstArrayRef<float> groupWeights) { // [objectIdx]
            RawTargetData.SetGroupWeights(groupWeights);
            MetaInfo.HasGroupWeight = true;
        }

        void SetPairs(TConstArrayRef<TPair> pairs) {
            RawTargetData.SetPairs(pairs);
            MetaInfo.HasPairs = true;
        }

        void SetSubgroupIds(TConstArrayRef<TSubgroupId> subgroupIds) { // [objectIdx]
            ObjectsData->SetSubgroupIds(subgroupIds);
            MetaInfo.HasSubgroupIds = true;
        }

        void SetWeights(TConstArrayRef<float> weights) { // [objectIdx]
            RawTargetData.SetWeights(weights);
            MetaInfo.HasWeights = true;
        }
    };

    using TDataProvider = TDataProviderTemplate<TObjectsDataProvider>;
    using TDataProviderPtr = TIntrusivePtr<TDataProvider>;
    using TConstDataProviderPtr = TIntrusivePtr<const TDataProvider>;

    using TRawDataProvider = TDataProviderTemplate<TRawObjectsDataProvider>;
    using TRawDataProviderPtr = TIntrusivePtr<TRawDataProvider>;
    using TConstRawDataProviderPtr = TIntrusivePtr<const TRawDataProvider>;

    using TQuantizedDataProvider = TDataProviderTemplate<TQuantizedObjectsDataProvider>;
    using TQuantizedDataProviderPtr = TIntrusivePtr<TQuantizedDataProvider>;
    using TConstQuantizedDataProviderPtr = TIntrusivePtr<const TQuantizedDataProvider>;

    using TQuantizedForCPUDataProvider = TDataProviderTemplate<TQuantizedForCPUObjectsDataProvider>;
    using TQuantizedForCPUDataProviderPtr = TIntrusivePtr<TQuantizedForCPUDataProvider>;
    using TConstQuantizedForCPUDataProviderPtr = TIntrusivePtr<const TQuantizedForCPUDataProvider>;


    /*
     * TDataProviderTemplate can be either TRawObjectsDataProvider or TQuantized(ForCPU)ObjectsDataProvider
     *  had to make this method instead of TDataProviderTemplate constructor because it
     *  won't work for TDataProviderTemplate=TTObjectsDataProvider (kind of base class)
     */
    template <class TTObjectsDataProvider>
    TIntrusivePtr<TDataProviderTemplate<TTObjectsDataProvider>> MakeDataProvider(
        TMaybe<TObjectsGroupingPtr> objectsGrouping, // if undefined ObjectsGrouping created from data
        TBuilderData<typename TTObjectsDataProvider::TData>&& builderData,
        bool skipCheck,
        NPar::TLocalExecutor* localExecutor
    ) {
        if (!skipCheck) {
            /* most likely have been already checked, but call it here for consistency with
             * overall checking logic
             */
            builderData.MetaInfo.Validate();
        }

        TIntrusivePtr<TTObjectsDataProvider> objectsData;

        auto makeObjectsDataProvider = [&] () {
            objectsData = MakeIntrusive<TTObjectsDataProvider>(
                objectsGrouping,
                std::move(builderData.CommonObjectsData),
                std::move(builderData.ObjectsData),
                skipCheck,
                localExecutor
            );
        };

        TVector<std::function<void()>> tasks;

        /* if objectsGrouping is defined we can run objectsData and target data creation in parallel
         * otherwise create grouping in objectsDataProvider first and then pass it to target constructor
         */
        if (objectsGrouping) {
            tasks.emplace_back(makeObjectsDataProvider);
        } else {
            makeObjectsDataProvider();
            objectsGrouping = objectsData->GetObjectsGrouping();
        }

        // TMaybe - to defer initialization to async task
        TMaybe<TRawTargetDataProvider> rawTargetData;

        tasks.emplace_back(
            [&] () {
                rawTargetData = MakeMaybe<TRawTargetDataProvider>(
                    *objectsGrouping,
                    std::move(builderData.TargetData),
                    skipCheck,
                    localExecutor
                );
            }
        );

        ExecuteTasksInParallel(&tasks, localExecutor);


        return MakeIntrusive<TDataProviderTemplate<TTObjectsDataProvider>>(
            std::move(builderData.MetaInfo), // assuming copying it is not very expensive
            objectsData,
            *objectsGrouping,
            std::move(*rawTargetData)
        );
    }


    template <class TTObjectsDataProvider>
    struct TDataProvidersTemplate {
        using TDataProviderTemplatePtr = TIntrusivePtr<TDataProviderTemplate<TTObjectsDataProvider>>;
        using TDataPtr = TDataProviderTemplatePtr;

        TDataProviderTemplatePtr Learn;
        TVector<TDataProviderTemplatePtr> Test;
    };

    using TDataProviders = TDataProvidersTemplate<TObjectsDataProvider>;
    using TRawDataProviders = TDataProvidersTemplate<TRawObjectsDataProvider>;
    using TQuantizedDataProviders = TDataProvidersTemplate<TQuantizedObjectsDataProvider>;
    using TQuantizedForCPUDataProviders = TDataProvidersTemplate<TQuantizedForCPUObjectsDataProvider>;


    template <class TTObjectsDataProvider>
    class TProcessedDataProviderTemplate : public TThrRefBase {
    public:
        TDataMetaInfo MetaInfo;
        TObjectsGroupingPtr ObjectsGrouping;
        TIntrusivePtr<TTObjectsDataProvider> ObjectsData;
        TTargetDataProviders TargetData;

    public:
        TProcessedDataProviderTemplate() = default;

        TProcessedDataProviderTemplate(
            TDataMetaInfo&& metaInfo,
            TObjectsGroupingPtr objectsGrouping,
            TIntrusivePtr<TTObjectsDataProvider> objectsData,
            TTargetDataProviders&& targetData
        )
            : MetaInfo(std::move(metaInfo))
            , ObjectsGrouping(objectsGrouping)
            , ObjectsData(std::move(objectsData))
            , TargetData(std::move(targetData))
        {}

        /* does not share serialized data with external structures
         *
         *  order of serialization is the following
         *  [MetaInfo]
         *  [ObjectsGrouping]
         *  [ObjectsData.CommonData.NonSharedPart]
         *  [QuantizedFeaturesInfo.NonSharedPart]
         *  [ObjectsData.Data.NonSharedPart]
         *  [TargetData] (w/o ObjectsGrouping)
         */
        int operator&(IBinSaver& binSaver);

        // ObjectsGrouping->GetObjectCount() used a lot, so make it a member here
        ui32 GetObjectCount() const {
            return ObjectsGrouping->GetObjectCount();
        }

        TIntrusivePtr<TProcessedDataProviderTemplate> GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            NPar::TLocalExecutor* localExecutor
        ) const {
            TVector<std::function<void()>> tasks;

            TIntrusivePtr<TTObjectsDataProvider> objectsDataSubset;

            tasks.emplace_back(
                [&, this]() {
                    auto baseObjectsDataSubset = ObjectsData->GetSubset(
                        objectsGroupingSubset,
                        localExecutor
                    );
                    objectsDataSubset = dynamic_cast<TTObjectsDataProvider*>(baseObjectsDataSubset.Get());
                    Y_VERIFY(objectsDataSubset);
                }
            );

            TTargetDataProviders targetDataSubset;

            tasks.emplace_back(
                [&, this]() {
                    targetDataSubset = NCB::GetSubsets(TargetData, objectsGroupingSubset, localExecutor);
                }
            );

            ExecuteTasksInParallel(&tasks, localExecutor);


            return MakeIntrusive<TProcessedDataProviderTemplate>(
                TDataMetaInfo(MetaInfo), // assuming copying it is not very expensive
                objectsDataSubset->GetObjectsGrouping(),
                std::move(objectsDataSubset),
                std::move(targetDataSubset)
            );
        }

        template <class TNewObjectsDataProvider>
        TProcessedDataProviderTemplate<TNewObjectsDataProvider> Cast() {
            TProcessedDataProviderTemplate<TNewObjectsDataProvider> newDataProvider;
            auto* newObjectsDataProvider = dynamic_cast<TNewObjectsDataProvider*>(ObjectsData.Get());
            CB_ENSURE_INTERNAL(newObjectsDataProvider, "Cannot cast to requested objects type");
            newDataProvider.MetaInfo = MetaInfo;
            newDataProvider.ObjectsGrouping = ObjectsGrouping;
            newDataProvider.ObjectsData = newObjectsDataProvider;
            newDataProvider.TargetData = TargetData;
            return newDataProvider;
        }
    };


    template <class TTObjectsDataProvider>
    int TProcessedDataProviderTemplate<TTObjectsDataProvider>::operator&(IBinSaver& binSaver) {
        AddWithShared(&binSaver, &MetaInfo);
        AddWithShared(&binSaver, &ObjectsGrouping);
        if (binSaver.IsReading()) {
            TObjectsSerialization::Load<TTObjectsDataProvider>(
                MetaInfo.FeaturesLayout,
                ObjectsGrouping,
                &binSaver,
                &ObjectsData
            );
            TTargetSerialization::Load(ObjectsGrouping, &binSaver, &TargetData);
        } else {
            TObjectsSerialization::SaveNonSharedPart<TTObjectsDataProvider>(*ObjectsData, &binSaver);
            TTargetSerialization::SaveNonSharedPart(TargetData, &binSaver);
        }
        return 0;
    }

    using TProcessedDataProvider = TProcessedDataProviderTemplate<TObjectsDataProvider>;
    using TProcessedDataProviderPtr = TIntrusivePtr<TProcessedDataProvider>;

    using TTrainingDataProvider = TProcessedDataProviderTemplate<TQuantizedObjectsDataProvider>;
    using TTrainingDataProviderPtr = TIntrusivePtr<TTrainingDataProvider>;
    using TConstTrainingDataProviderPtr = TIntrusivePtr<const TTrainingDataProviderPtr>;

    using TTrainingForCPUDataProvider = TProcessedDataProviderTemplate<TQuantizedForCPUObjectsDataProvider>;
    using TTrainingForCPUDataProviderPtr = TIntrusivePtr<TTrainingForCPUDataProvider>;
    using TConstTrainingForCPUDataProviderPtr = TIntrusivePtr<const TTrainingForCPUDataProvider>;


    template <class TTObjectsDataProvider>
    inline ui32 GetObjectCount(
        TConstArrayRef<TIntrusivePtr<TProcessedDataProviderTemplate<TTObjectsDataProvider>>> data
    ) {
        ui32 totalCount = 0;
        for (const auto& dataPart : data) {
            totalCount += dataPart->GetObjectCount();
        }
        return totalCount;
    }

    template <class TTObjectsDataProvider>
    inline TVector<size_t> CalcTestOffsets(
        ui32 learnObjectCount,
        TConstArrayRef<TIntrusivePtr<TProcessedDataProviderTemplate<TTObjectsDataProvider>>> testData
    ) {
        TVector<size_t> testOffsets(testData.size() + 1);
        testOffsets[0] = learnObjectCount;
        for (auto testIdx : xrange(testData.size())) {
            testOffsets[testIdx + 1] = testOffsets[testIdx] + testData[testIdx]->GetObjectCount();
        }
        return testOffsets;
    }


    template <class TTObjectsDataProvider>
    class TTrainingDataProvidersTemplate {
    public:
        using TTrainingDataProviderTemplatePtr =
            TIntrusivePtr<TProcessedDataProviderTemplate<TTObjectsDataProvider>>;
        using TDataPtr = TTrainingDataProviderTemplatePtr;

        TTrainingDataProviderTemplatePtr Learn;
        TVector<TTrainingDataProviderTemplatePtr> Test;

    public:
        SAVELOAD_WITH_SHARED(Learn, Test)

        ui32 GetTestSampleCount() const {
            return NCB::GetObjectCount<TTObjectsDataProvider>(Test);
        }

        TVector<size_t> CalcTestOffsets() const {
            return NCB::CalcTestOffsets<TTObjectsDataProvider>(Learn->GetObjectCount(), Test);
        }

        TFeaturesLayoutPtr GetFeaturesLayout() const {
            return Learn->MetaInfo.GetFeaturesLayout();
        }

        template <class TNewObjectsDataProvider>
        TTrainingDataProvidersTemplate<TNewObjectsDataProvider> Cast() {
            using TNewData = TProcessedDataProviderTemplate<TNewObjectsDataProvider>;

            TTrainingDataProvidersTemplate<TNewObjectsDataProvider> newData;
            newData.Learn = MakeIntrusive<TNewData>(Learn->template Cast<TNewObjectsDataProvider>());
            for (auto& testData : Test) {
                newData.Test.emplace_back(
                    MakeIntrusive<TNewData>(testData->template Cast<TNewObjectsDataProvider>())
                );
            }
            return newData;
        }
    };

    using TTrainingDataProviders = TTrainingDataProvidersTemplate<TQuantizedObjectsDataProvider>;
    using TTrainingForCPUDataProviders = TTrainingDataProvidersTemplate<TQuantizedForCPUObjectsDataProvider>;

}
