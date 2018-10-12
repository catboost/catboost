#pragma once

#include "meta_info.h"
#include "objects_grouping.h"
#include "objects.h"
#include "target.h"

#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/libs/helpers/serialization.h>

#include <library/binsaver/bin_saver.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/ptr.h>
#include <util/generic/vector.h>


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
                ;
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
                    objectsDataSubset = ObjectsData->GetSubset(
                        objectsGroupingSubset,
                        localExecutor
                    );
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

        TDataProviderTemplatePtr Learn;
        TVector<TDataProviderTemplatePtr> Test;
    };

    using TDataProviders = TDataProvidersTemplate<TObjectsDataProvider>;
    using TRawDataProviders = TDataProvidersTemplate<TRawObjectsDataProvider>;
    using TQuantizedDataProviders = TDataProvidersTemplate<TQuantizedObjectsDataProvider>;
    using TQuantizedForCPUDataProviders = TDataProvidersTemplate<TQuantizedForCPUObjectsDataProvider>;


    template <class TTObjectsDataProvider>
    class TTrainingDataProviderTemplate : TThrRefBase {
    public:
        TDataMetaInfo MetaInfo;
        TObjectsGroupingPtr ObjectsGrouping;
        TIntrusivePtr<TTObjectsDataProvider> ObjectsData;
        TTargetDataProviders TargetData;

    public:
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
    };


    template <class TTObjectsDataProvider>
    int TTrainingDataProviderTemplate<TTObjectsDataProvider>::operator&(IBinSaver& binSaver) {
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


    using TTrainingDataProvider = TTrainingDataProviderTemplate<TQuantizedObjectsDataProvider>;
    using TTrainingDataProviderPtr = TIntrusivePtr<TQuantizedObjectsDataProvider>;
    using TConstTrainingDataProviderPtr = TIntrusivePtr<const TQuantizedObjectsDataProvider>;

    using TTrainingForCPUDataProvider = TTrainingDataProviderTemplate<TQuantizedForCPUObjectsDataProvider>;
    using TTrainingForCPUDataProviderPtr = TIntrusivePtr<TQuantizedForCPUObjectsDataProvider>;
    using TConstTrainingForCPUDataProviderPtr = TIntrusivePtr<const TQuantizedForCPUObjectsDataProvider>;


    template <class TTObjectsDataProvider>
    class TTrainingDataProvidersTemplate {
    public:
        using TTrainingDataProviderTemplatePtr =
            TIntrusivePtr<TTrainingDataProviderTemplate<TTObjectsDataProvider>>;

        TTrainingDataProviderTemplatePtr Learn;
        TVector<TTrainingDataProviderTemplatePtr> Test;

    public:
        SAVELOAD_WITH_SHARED(Learn, Test)
    };

    using TTrainingDataProviders = TTrainingDataProvidersTemplate<TQuantizedObjectsDataProvider>;
    using TTrainingForCPUDataProviders = TTrainingDataProvidersTemplate<TQuantizedForCPUObjectsDataProvider>;

}
