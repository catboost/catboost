#pragma once

#include "feature_estimators.h"
#include "meta_info.h"
#include "objects_grouping.h"
#include "objects.h"
#include "target.h"

#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/libs/helpers/serialization.h>

#include <library/cpp/binsaver/bin_saver.h>
#include <library/cpp/dbg_output/dump.h>
#include <library/cpp/threading/local_executor/local_executor.h>

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

        bool EqualTo(const TDataProviderTemplate& rhs, bool ignoreSparsity = false) const {
            return MetaInfo.EqualTo(rhs.MetaInfo, ignoreSparsity) &&
                ObjectsData->EqualTo(*rhs.ObjectsData, ignoreSparsity) &&
                (*ObjectsGrouping == *rhs.ObjectsGrouping) && (RawTargetData == rhs.RawTargetData);
        }

        bool operator==(const TDataProviderTemplate& rhs) const {
            return EqualTo(rhs);
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
            ui64 cpuUsedRamLimit,
            NPar::ILocalExecutor* localExecutor
        ) const {
            TVector<std::function<void()>> tasks;

            TIntrusivePtr<TTObjectsDataProvider> objectsDataSubset;

            tasks.emplace_back(
                [&, this]() {
                    auto baseObjectsDataSubset = ObjectsData->GetSubset(
                        objectsGroupingSubset,
                        cpuUsedRamLimit,
                        localExecutor
                    );
                    objectsDataSubset = dynamic_cast<TTObjectsDataProvider*>(baseObjectsDataSubset.Get());
                    CB_ENSURE(objectsDataSubset, "Unexpected type of data provider");
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
            ui64 cpuUsedRamLimit,
            int threadCount
        ) const {
            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(threadCount);
            return GetSubset(objectsGroupingSubset, cpuUsedRamLimit, &localExecutor);
        }

        TIntrusivePtr<TDataProviderTemplate> Clone(
            ui64 cpuUsedRamLimit,
            NPar::ILocalExecutor* localExecutor
        ) const {
            return GetSubset(
                GetGroupingSubsetFromObjectsSubset(
                    ObjectsGrouping,
                    TArraySubsetIndexing(TFullSubset<ui32>(GetObjectCount())),
                    EObjectsOrder::Ordered),
                cpuUsedRamLimit,
                localExecutor
            );
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

        void SetTimestamps(TConstArrayRef<ui64> timestamps) { // [objectIdx]
            ObjectsData->SetTimestamps(timestamps);
            MetaInfo.HasTimestamp = true;
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
    using TConstQuantizedDataProviderPtr = TIntrusivePtr<const TQuantizedObjectsDataProvider>;

    /*
     * TDataProviderTemplate can be either TRawObjectsDataProvider or TQuantizedObjectsDataProvider
     *  had to make this method instead of TDataProviderTemplate constructor because it
     *  won't work for TDataProviderTemplate=TTObjectsDataProvider (kind of base class)
     */
    template <class TTObjectsDataProvider>
    TIntrusivePtr<TDataProviderTemplate<TTObjectsDataProvider>> MakeDataProvider(
        TMaybe<TObjectsGroupingPtr> objectsGrouping, // if undefined ObjectsGrouping created from data
        TBuilderData<typename TTObjectsDataProvider::TData>&& builderData,
        bool skipCheck,
        bool forceUnitAutoPairWeights,
        NPar::ILocalExecutor* localExecutor
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
                    forceUnitAutoPairWeights,
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


    template <class TTObjectsDataProvider>
    class TProcessedDataProviderTemplate : public TThrRefBase {
    public:
        TFeaturesLayoutPtr OriginalFeaturesLayout; // because Quantize can be destructive now
        TDataMetaInfo MetaInfo;
        TObjectsGroupingPtr ObjectsGrouping;
        TIntrusivePtr<TTObjectsDataProvider> ObjectsData;
        TTargetDataProviderPtr TargetData;

    public:
        TProcessedDataProviderTemplate() = default;

        TProcessedDataProviderTemplate(
            TFeaturesLayoutPtr originalFeaturesLayout,
            TDataMetaInfo&& metaInfo,
            TObjectsGroupingPtr objectsGrouping,
            TIntrusivePtr<TTObjectsDataProvider> objectsData,
            TTargetDataProviderPtr targetData
        )
            : OriginalFeaturesLayout(std::move(originalFeaturesLayout))
            , MetaInfo(std::move(metaInfo))
            , ObjectsGrouping(objectsGrouping)
            , ObjectsData(std::move(objectsData))
            , TargetData(std::move(targetData))
        {}

        /* does not share serialized data with external structures
         *
         *  order of serialization is the following
         *  [OriginalFeaturesLayout]
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
            ui64 cpuUsedRamLimit,
            NPar::ILocalExecutor* localExecutor
        ) const {
            TVector<std::function<void()>> tasks;

            TIntrusivePtr<TTObjectsDataProvider> objectsDataSubset;

            tasks.emplace_back(
                [&, this]() {
                    auto baseObjectsDataSubset = ObjectsData->GetSubset(
                        objectsGroupingSubset,
                        cpuUsedRamLimit,
                        localExecutor
                    );
                    objectsDataSubset = dynamic_cast<TTObjectsDataProvider*>(baseObjectsDataSubset.Get());
                    CB_ENSURE(objectsDataSubset, "Unexpected type of data provider");
                }
            );

            TTargetDataProviderPtr targetDataSubset;

            tasks.emplace_back(
                [&, this]() {
                    targetDataSubset = TargetData->GetSubset(objectsGroupingSubset, localExecutor);
                }
            );

            ExecuteTasksInParallel(&tasks, localExecutor);

            auto subset = MakeIntrusive<TProcessedDataProviderTemplate>(
                OriginalFeaturesLayout,
                TDataMetaInfo(MetaInfo), // assuming copying it is not very expensive
                objectsDataSubset->GetObjectsGrouping(),
                std::move(objectsDataSubset),
                std::move(targetDataSubset)
            );
            subset->UpdateMetaInfo();

            return subset;
        }

        void UpdateMetaInfo() {
            MetaInfo.ObjectCount = GetObjectCount();
            if (ObjectsData->GetQuantizedFeaturesInfo()) {
                MetaInfo.MaxCatFeaturesUniqValuesOnLearn =
                    ObjectsData->GetQuantizedFeaturesInfo()->CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn();
            }

            const auto& targets = TargetData->GetTarget();
            if (targets.Defined() && MetaInfo.ObjectCount > 0 && targets->size() == 1) {
                auto targetBounds = CalcMinMax(MakeConstArrayRef(targets->front()));
                MetaInfo.TargetStats = {targetBounds.Min, targetBounds.Max};
            }
        }
    };


    template <class TTObjectsDataProvider>
    int TProcessedDataProviderTemplate<TTObjectsDataProvider>::operator&(IBinSaver& binSaver) {
        AddWithShared(&binSaver, &OriginalFeaturesLayout);
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
            TTargetSerialization::SaveNonSharedPart(*TargetData, &binSaver);
        }
        return 0;
    }

    using TProcessedDataProvider = TProcessedDataProviderTemplate<TObjectsDataProvider>;
    using TProcessedDataProviderPtr = TIntrusivePtr<TProcessedDataProvider>;

    using TTrainingDataProvider = TProcessedDataProviderTemplate<TQuantizedObjectsDataProvider>;
    using TTrainingDataProviderPtr = TIntrusivePtr<TTrainingDataProvider>;

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

    template <class TDataProvidersTemplate>
    TDataProvidersTemplate CreateTrainTestSubsets(
        typename TDataProvidersTemplate::TDataPtr srcData,
        NCB::TArraySubsetIndexing<ui32>&& trainIndices,
        NCB::TArraySubsetIndexing<ui32>&& testIndices,
        ui64 cpuUsedRamLimit,
        NPar::ILocalExecutor* localExecutor
    ) {
        const ui64 perTaskCpuUsedRamLimit = cpuUsedRamLimit / 2;

        const NCB::EObjectsOrder objectsOrder = NCB::EObjectsOrder::Ordered;
        TDataProvidersTemplate result;

        TVector<std::function<void()>> tasks;
        tasks.emplace_back(
            [&]() {
                result.Learn = srcData->GetSubset(
                    GetSubset(
                        srcData->ObjectsGrouping,
                        std::move(trainIndices),
                        objectsOrder
                    ),
                    perTaskCpuUsedRamLimit,
                    localExecutor
                );
            }
        );
        tasks.emplace_back(
            [&]() {
                result.Test.emplace_back(
                    srcData->GetSubset(
                        GetSubset(
                            srcData->ObjectsGrouping,
                            std::move(testIndices),
                            objectsOrder
                        ),
                        perTaskCpuUsedRamLimit,
                        localExecutor
                    )
                );
            }
        );
        NCB::ExecuteTasksInParallel(&tasks, localExecutor);
        return result;
    }


    struct TQuantizedEstimatedFeaturesInfo {
    public:
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
        TVector<TEstimatedFeatureId> Layout; // [objectDataProvider FeatureIdx]

    public:
         inline int operator&(IBinSaver& binSaver) {
             AddWithSharedMulti(&binSaver, QuantizedFeaturesInfo);
             binSaver.Add(0, &Layout);
             return 0;
         }
    };

    template <class TTObjectsDataProvider>
    void SerializeNonShared(IBinSaver* binSaver, TIntrusivePtr<TTObjectsDataProvider>* objectsData) {
        bool nonEmpty;
        TFeaturesLayoutPtr featuresLayout;
        TObjectsGroupingPtr objectsGrouping;

        if (binSaver->IsReading()) {
            binSaver->Add(0, &nonEmpty);
            if (!nonEmpty) {
                objectsData->Reset();
                return;
            }

            AddWithSharedMulti(binSaver, featuresLayout, objectsGrouping);

            TObjectsSerialization::Load<TTObjectsDataProvider>(
                std::move(featuresLayout),
                std::move(objectsGrouping),
                binSaver,
                objectsData
            );
        } else {
            nonEmpty = objectsData->Get() != nullptr;
            binSaver->Add(0, &nonEmpty);
            if (!nonEmpty) {
                return;
            }
            featuresLayout = (*objectsData)->GetFeaturesLayout();
            objectsGrouping = (*objectsData)->GetObjectsGrouping();
            AddWithSharedMulti(binSaver, featuresLayout, objectsGrouping);

            TObjectsSerialization::SaveNonSharedPart<TTObjectsDataProvider>(**objectsData, binSaver);
        }
    }

    class TEstimatedForCPUObjectsDataProviders {
    public:
        using TTObjectsDataProvider = TQuantizedObjectsDataProvider;
        using TTObjectsDataProviderPtr = TIntrusivePtr<TTObjectsDataProvider>;

        TTObjectsDataProviderPtr Learn; // can be nullptr
        TVector<TTObjectsDataProviderPtr> Test;

        TFeatureEstimatorsPtr FeatureEstimators;

        TQuantizedEstimatedFeaturesInfo QuantizedEstimatedFeaturesInfo;

    public:
        // FeatureEstimators is non-serializable
        inline int operator&(IBinSaver& binSaver) {
             SerializeNonShared(&binSaver, &Learn);
             size_t testCount;
             if (!binSaver.IsReading()) {
                 testCount = Test.size();
             }
             binSaver.Add(0, &testCount);
             if (binSaver.IsReading()) {
                 Test.resize(testCount);
             }
             for (auto& testPart : Test) {
                 SerializeNonShared(&binSaver, &testPart);
             }
             binSaver.Add(0, &QuantizedEstimatedFeaturesInfo);
             return 0;
         }

        ui32 GetFeatureCount() const {
            return QuantizedEstimatedFeaturesInfo.Layout.size();
        }

        TQuantizedFeaturesInfoPtr GetQuantizedFeaturesInfo() const {
            return QuantizedEstimatedFeaturesInfo.QuantizedFeaturesInfo;
        }

        ui32 CalcFeaturesCheckSum(NPar::ILocalExecutor* localExecutor) const {
            ui32 checkSum = 0;
            if (Learn) {
                checkSum += Learn->CalcFeaturesCheckSum(localExecutor);
            }
            for (const auto& testData : Test) {
                checkSum += testData->CalcFeaturesCheckSum(localExecutor);
            }
            return checkSum;
        }
    };

    class TTrainingDataProviders {
    public:
        using TTObjectsDataProvider = TQuantizedObjectsDataProvider;
        using TTrainingDataProviderTemplatePtr =
            TIntrusivePtr<TProcessedDataProviderTemplate<TTObjectsDataProvider>>;
        using TDataPtr = TTrainingDataProviderTemplatePtr;

        TTrainingDataProviderTemplatePtr Learn;
        TVector<TTrainingDataProviderTemplatePtr> Test;

        TFeatureEstimatorsPtr FeatureEstimators = MakeIntrusive<TFeatureEstimators>();

        // not filled for GPU to save memory
        TEstimatedForCPUObjectsDataProviders EstimatedObjectsData;

    public:
        // FeatureEstimators is non-serializable
        inline int operator&(IBinSaver& binSaver) {
             AddWithSharedMulti(&binSaver, Learn, Test);
             binSaver.Add(0, &EstimatedObjectsData);
             return 0;
        }

        ui32 GetTestSampleCount() const {
            return NCB::GetObjectCount<TTObjectsDataProvider>(Test);
        }

        TVector<size_t> CalcTestOffsets() const {
            return NCB::CalcTestOffsets<TTObjectsDataProvider>(Learn->GetObjectCount(), Test);
        }

        TFeaturesLayoutPtr GetFeaturesLayout() const {
            return Learn->MetaInfo.FeaturesLayout;
        }

        ui32 CalcFeaturesCheckSum(NPar::ILocalExecutor* localExecutor) const {
            ui32 checkSum = Learn->ObjectsData->CalcFeaturesCheckSum(localExecutor);
            for (const auto& testData : Test) {
                checkSum += testData->ObjectsData->CalcFeaturesCheckSum(localExecutor);
            }
            checkSum += EstimatedObjectsData.CalcFeaturesCheckSum(localExecutor);
            return checkSum;
        }
    };

}
