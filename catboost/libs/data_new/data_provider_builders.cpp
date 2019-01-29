#include "data_provider_builders.h"

#include "data_provider.h"
#include "feature_index.h"
#include "objects.h"
#include "target.h"
#include "util.h"
#include "visitor.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/compression.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/resource_holder.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/options/restrictions.h>
#include <catboost/libs/quantization/utils.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/xrange.h>
#include <util/generic/ylimits.h>
#include <util/generic/ymath.h>
#include <util/stream/labeled.h>
#include <util/system/yassert.h>

#include <algorithm>
#include <array>


namespace NCB {

    // hack to extract private data from inside providers
    class TRawBuilderDataHelper {
    public:
        static TRawBuilderData Extract(TRawDataProvider&& rawDataProvider) {
            TRawBuilderData data;
            data.MetaInfo = std::move(rawDataProvider.MetaInfo);
            data.TargetData = std::move(rawDataProvider.RawTargetData.Data);
            data.CommonObjectsData = std::move(rawDataProvider.ObjectsData->CommonData);
            data.ObjectsData = std::move(rawDataProvider.ObjectsData->Data);
            return data;
        }
    };


    class TRawObjectsOrderDataProviderBuilder : public IDataProviderBuilder,
                                                public IRawObjectsOrderDataVisitor
    {
    public:
        TRawObjectsOrderDataProviderBuilder(
            const TDataProviderBuilderOptions& options,
            NPar::TLocalExecutor* localExecutor
        )
            : InBlock(false)
            , ObjectCount(0)
            , CatFeatureCount(0)
            , Cursor(NotSet)
            , NextCursor(0)
            , Options(options)
            , LocalExecutor(localExecutor)
            , InProcess(false)
            , ResultTaken(false)
        {}

        void Start(
            bool inBlock,
            const TDataMetaInfo& metaInfo,
            ui32 objectCount,
            EObjectsOrder objectsOrder,

            // keep necessary resources for data to be available (memory mapping for a file for example)
            TVector<TIntrusivePtr<IResourceHolder>> resourceHolders
        ) override {
            CB_ENSURE(!InProcess, "Attempt to start new processing without finishing the last");
            InProcess = true;
            ResultTaken = false;

            InBlock = inBlock;

            ui32 prevTailSize = 0;
            if (InBlock) {
                CB_ENSURE(!metaInfo.HasPairs, "Pairs are not supported in block processing");

                prevTailSize = (NextCursor < ObjectCount) ? (ObjectCount - NextCursor) : 0;
                NextCursor = prevTailSize;
            } else {
                NextCursor = 0;
            }
            ObjectCount = objectCount + prevTailSize;
            CatFeatureCount = metaInfo.FeaturesLayout->GetCatFeatureCount();

            Cursor = NotSet;

            Data.MetaInfo = metaInfo;
            Data.TargetData.PrepareForInitialization(metaInfo, ObjectCount, prevTailSize);
            Data.CommonObjectsData.PrepareForInitialization(metaInfo, ObjectCount, prevTailSize);
            Data.ObjectsData.PrepareForInitialization(metaInfo);

            Data.CommonObjectsData.ResourceHolders = std::move(resourceHolders);
            Data.CommonObjectsData.Order = objectsOrder;

            FloatFeaturesStorage.PrepareForInitialization(*metaInfo.FeaturesLayout, ObjectCount, prevTailSize);
            CatFeaturesStorage.PrepareForInitialization(*metaInfo.FeaturesLayout, ObjectCount, prevTailSize);

            if (metaInfo.HasWeights) {
                PrepareForInitialization(ObjectCount, prevTailSize, &WeightsBuffer);
            }
            if (metaInfo.HasGroupWeight) {
                PrepareForInitialization(ObjectCount, prevTailSize, &GroupWeightsBuffer);
            }
        }

        void StartNextBlock(ui32 blockSize) override {
            Cursor = NextCursor;
            NextCursor = Cursor + blockSize;
        }

        // TCommonObjectsData
        void AddGroupId(ui32 localObjectIdx, TGroupId value) override {
            (*Data.CommonObjectsData.GroupIds)[Cursor + localObjectIdx] = value;
        }

        void AddSubgroupId(ui32 localObjectIdx, TSubgroupId value) override {
            (*Data.CommonObjectsData.SubgroupIds)[Cursor + localObjectIdx] = value;
        }

        void AddTimestamp(ui32 localObjectIdx, ui64 value) override {
            (*Data.CommonObjectsData.Timestamp)[Cursor + localObjectIdx] = value;
        }

        // TRawObjectsData
        void AddFloatFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, float feature) override {
            FloatFeaturesStorage.Set(
                GetInternalFeatureIdx<EFeatureType::Float>(flatFeatureIdx),
                Cursor + localObjectIdx,
                feature
            );
        }

        void AddAllFloatFeatures(ui32 localObjectIdx, TConstArrayRef<float> features) override {
            auto objectIdx = Cursor + localObjectIdx;
            for (auto perTypeFeatureIdx : xrange(features.size())) {
                FloatFeaturesStorage.Set(
                    TFloatFeatureIdx(perTypeFeatureIdx),
                    objectIdx,
                    features[perTypeFeatureIdx]
                );
            }
        }

        ui32 GetCatFeatureValue(ui32 flatFeatureIdx, TStringBuf feature) override {
            auto catFeatureIdx = GetInternalFeatureIdx<EFeatureType::Categorical>(flatFeatureIdx);
            ui32 hashVal = CalcCatFeatureHash(feature);
            int hashPartIdx = LocalExecutor->GetWorkerThreadId();
            CB_ENSURE(hashPartIdx < CB_THREAD_LIMIT, "Internal error: thread ID exceeds CB_THREAD_LIMIT");
            auto& catFeatureHashes = HashMapParts[hashPartIdx].CatFeatureHashes;
            catFeatureHashes.resize(CatFeatureCount);
            auto& catFeatureHash = catFeatureHashes[*catFeatureIdx];

            THashMap<ui32, TString>::insert_ctx insertCtx;
            if (!catFeatureHash.contains(hashVal, insertCtx)) {
                catFeatureHash.emplace_direct(insertCtx, hashVal, feature);
            }
            return hashVal;
        }
        void AddCatFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) override {
            auto catFeatureIdx = GetInternalFeatureIdx<EFeatureType::Categorical>(flatFeatureIdx);
            CatFeaturesStorage.Set(
                catFeatureIdx,
                Cursor + localObjectIdx,
                GetCatFeatureValue(flatFeatureIdx, feature)
            );
        }
        void AddAllCatFeatures(ui32 localObjectIdx, TConstArrayRef<ui32> features) override {
            auto objectIdx = Cursor + localObjectIdx;
            for (auto perTypeFeatureIdx : xrange(features.size())) {
                CatFeaturesStorage.Set(
                    TCatFeatureIdx(perTypeFeatureIdx),
                    objectIdx,
                    features[perTypeFeatureIdx]
                );
            }
        }


        // TRawTargetData

        /* if raw data contains target data as strings (label is a synonym for target)
            prefer passing it as TString to avoid unnecessary memory copies
            even if these strings represent float or ints
        */
        void AddTarget(ui32 localObjectIdx, const TString& value) override {
            (*Data.TargetData.Target)[Cursor + localObjectIdx] = value;
        }
        void AddTarget(ui32 localObjectIdx, float value) override {
            (*Data.TargetData.Target)[Cursor + localObjectIdx] = ToString(value);
        }
        void AddBaseline(ui32 localObjectIdx, ui32 baselineIdx, float value) override {
            Data.TargetData.Baseline[baselineIdx][Cursor + localObjectIdx] = value;
        }
        void AddWeight(ui32 localObjectIdx, float value) override {
            WeightsBuffer[Cursor + localObjectIdx] = value;
        }
        void AddGroupWeight(ui32 localObjectIdx, float value) override {
            GroupWeightsBuffer[Cursor + localObjectIdx] = value;
        }

        // separate method because they can be loaded from a separate data source
        void SetGroupWeights(TVector<float>&& groupWeights) override {
            CheckDataSize(groupWeights.size(), (size_t)ObjectCount, "groupWeights");
            GroupWeightsBuffer = std::move(groupWeights);
        }

        void SetPairs(TVector<TPair>&& pairs) override {
            Data.TargetData.Pairs = std::move(pairs);
        }

        // needed for checking groupWeights consistency while loading from separate file
        TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const override {
            return Data.CommonObjectsData.GroupIds;
        }

        void Finish() override {
            CB_ENSURE(InProcess, "Attempt to Finish without starting processing");
            CB_ENSURE(
                NextCursor >= ObjectCount,
                "processed object count is less than than specified in metadata"
            );

            if (ObjectCount != 0) {
                CATBOOST_INFO_LOG << "Object info sizes: " << ObjectCount << " "
                    << Data.MetaInfo.FeaturesLayout->GetExternalFeatureCount() << Endl;
            } else {
                // should this be an error?
                CATBOOST_ERROR_LOG << "No objects info loaded" << Endl;
            }

            // if data has groups and we're in block mode, skip last group data
            if (InBlock && Data.MetaInfo.HasGroupId) {
                RollbackNextCursorToLastGroupStart();
            }

            InProcess = false;
        }

        TDataProviderPtr GetResult() override {
            CB_ENSURE_INTERNAL(!InProcess, "Attempt to GetResult before finishing processing");
            CB_ENSURE_INTERNAL(!ResultTaken, "Attempt to GetResult several times");

            if (InBlock && Data.MetaInfo.HasGroupId) {
                // copy, not move weights buffers

                if (Data.MetaInfo.HasWeights) {
                    Data.TargetData.Weights = TWeights<float>(TVector<float>(WeightsBuffer));
                }
                if (Data.MetaInfo.HasGroupWeight) {
                    Data.TargetData.GroupWeights = TWeights<float>(TVector<float>(GroupWeightsBuffer));
                }
            } else {
                if (Data.MetaInfo.HasWeights) {
                    Data.TargetData.Weights = TWeights<float>(std::move(WeightsBuffer));
                }
                if (Data.MetaInfo.HasGroupWeight) {
                    Data.TargetData.GroupWeights = TWeights<float>(std::move(GroupWeightsBuffer));
                }
            }

            Data.CommonObjectsData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(ObjectCount)
            );

            FloatFeaturesStorage.GetResult(
                *Data.MetaInfo.FeaturesLayout,
                Data.CommonObjectsData.SubsetIndexing.Get(),
                &Data.ObjectsData.FloatFeatures
            );

            CatFeaturesStorage.GetResult(
                *Data.MetaInfo.FeaturesLayout,
                Data.CommonObjectsData.SubsetIndexing.Get(),
                &Data.ObjectsData.CatFeatures
            );

            if (CatFeatureCount) {
                auto& catFeaturesHashToString = *Data.CommonObjectsData.CatFeaturesHashToString;
                catFeaturesHashToString.resize(CatFeatureCount);
                for (const auto& part : HashMapParts) {
                    if (part.CatFeatureHashes.empty()) {
                        continue;
                    }
                    for (auto catFeatureIdx : xrange(CatFeatureCount)) {
                        catFeaturesHashToString[catFeatureIdx].insert(
                            part.CatFeatureHashes[catFeatureIdx].begin(),
                            part.CatFeatureHashes[catFeatureIdx].end()
                        );
                    }
                }
            }

            ResultTaken = true;

            if (InBlock && Data.MetaInfo.HasGroupId) {
                auto fullData = MakeDataProvider<TRawObjectsDataProvider>(
                    /*objectsGrouping*/ Nothing(), // will init from data
                    std::move(Data),
                    Options.SkipCheck,
                    LocalExecutor
                );

                ui32 groupCount = fullData->ObjectsGrouping->GetGroupCount();
                CB_ENSURE(groupCount != 1, "blocks must be big enough to contain more than a single group");
                TVector<TSubsetBlock<ui32>> subsetBlocks = {TSubsetBlock<ui32>{{ui32(0), groupCount - 1}, 0}};
                auto result = fullData->GetSubset(
                    GetSubset(
                        fullData->ObjectsGrouping,
                        TArraySubsetIndexing<ui32>(
                            TRangesSubset<ui32>(groupCount - 1, std::move(subsetBlocks))
                        ),
                        EObjectsOrder::Ordered
                    ),
                    LocalExecutor
                )->CastMoveTo<TObjectsDataProvider>();

                // save Data parts - it still contains last group data
                Data = TRawBuilderDataHelper::Extract(std::move(*fullData));

                return result;
            } else {
                return MakeDataProvider<TRawObjectsDataProvider>(
                    /*objectsGrouping*/ Nothing(), // will init from data
                    std::move(Data),
                    Options.SkipCheck,
                    LocalExecutor
                )->CastMoveTo<TObjectsDataProvider>();
            }
        }

        TDataProviderPtr GetLastResult() override {
            CB_ENSURE_INTERNAL(!InProcess, "Attempt to GetLastResult before finishing processing");
            CB_ENSURE_INTERNAL(ResultTaken, "Attempt to call GetLastResult before GetResult");

            if (!InBlock || !Data.MetaInfo.HasGroupId) {
                return nullptr;
            }

            auto fullData = MakeDataProvider<TRawObjectsDataProvider>(
                /*objectsGrouping*/ Nothing(), // will init from data
                std::move(Data),
                Options.SkipCheck,
                LocalExecutor
            );

            ui32 groupCount = fullData->ObjectsGrouping->GetGroupCount();
            Y_VERIFY(groupCount != 1);
            TVector<TSubsetBlock<ui32>> subsetBlocks = {TSubsetBlock<ui32>{{groupCount - 1, groupCount}, 0}};
            return fullData->GetSubset(
                GetSubset(
                    fullData->ObjectsGrouping,
                    TArraySubsetIndexing<ui32>(TRangesSubset<ui32>(1, std::move(subsetBlocks))),
                    EObjectsOrder::Ordered
                ),
                LocalExecutor
            )->CastMoveTo<TObjectsDataProvider>();
        }

    private:
        void RollbackNextCursorToLastGroupStart() {
            const auto& groupIds = *Data.CommonObjectsData.GroupIds;
            if (ObjectCount == 0) {
                return;
            }
            auto rit = groupIds.rbegin();
            TGroupId lastGroupId = *rit;
            for (++rit; rit != groupIds.rend(); ++rit) {
                if (*rit != lastGroupId) {
                    break;
                }
            }
            NextCursor = ObjectCount - (rit - groupIds.rbegin());
        }

        template <EFeatureType FeatureType>
        TFeatureIdx<FeatureType> GetInternalFeatureIdx(ui32 flatFeatureIdx) const {
            return TFeatureIdx<FeatureType>(
                Data.MetaInfo.FeaturesLayout->GetInternalFeatureIdx(flatFeatureIdx)
            );
        }

    private:
        struct THashPart {
            TVector<THashMap<ui32, TString>> CatFeatureHashes;
        };


        template <EFeatureType FeatureType, class T>
        struct TFeaturesStorage {
            /* shared between this builder and created data provider for efficiency
             * (can be reused for cache here if there's no more references to data
             *  (data provider was freed))
             */

            TVector<TIntrusivePtr<TVectorHolder<T>>> Storage; // [perTypeFeatureIdx]

            // view into storage for faster access
            TVector<TArrayRef<T>> DstView; // [perTypeFeatureIdx]

            // copy from Data.MetaInfo.FeaturesLayout for fast access
            TVector<bool> IsAvailable; // [perTypeFeatureIdx]

        public:
            void PrepareForInitialization(
                const TFeaturesLayout& featuresLayout,
                ui32 objectCount,
                ui32 prevTailSize
            ) {
                const size_t featureCount = (size_t) featuresLayout.GetFeatureCount(FeatureType);
                Storage.resize(featureCount);
                DstView.resize(featureCount);
                IsAvailable.yresize(featureCount);
                for (auto perTypeFeatureIdx : xrange(featureCount)) {
                    if (featuresLayout.GetInternalFeatureMetaInfo(perTypeFeatureIdx, FeatureType).IsAvailable)
                    {
                        auto& maybeSharedStoragePtr = Storage[perTypeFeatureIdx];

                        if (!maybeSharedStoragePtr) {
                            Y_VERIFY(!prevTailSize);
                            maybeSharedStoragePtr = MakeIntrusive<TVectorHolder<T>>();
                            maybeSharedStoragePtr->Data.yresize(objectCount);
                        } else {
                            Y_VERIFY(prevTailSize <= maybeSharedStoragePtr->Data.size());
                            auto newMaybeSharedStoragePtr = MakeIntrusive<TVectorHolder<T>>();
                            newMaybeSharedStoragePtr->Data.yresize(objectCount);
                            if (prevTailSize) {
                                std::copy(
                                    maybeSharedStoragePtr->Data.end() - prevTailSize,
                                    maybeSharedStoragePtr->Data.end(),
                                    newMaybeSharedStoragePtr->Data.begin()
                                );
                            }
                            maybeSharedStoragePtr = std::move(newMaybeSharedStoragePtr);
                        }
                        DstView[perTypeFeatureIdx] = maybeSharedStoragePtr->Data;
                        IsAvailable[perTypeFeatureIdx] = true;
                    } else {
                        Storage[perTypeFeatureIdx] = nullptr;
                        DstView[perTypeFeatureIdx] = TArrayRef<T>();
                        IsAvailable[perTypeFeatureIdx] = false;
                    }
                }
            }

            void Set(TFeatureIdx<FeatureType> perTypeFeatureIdx, ui32 objectIdx, T value) {
                if (IsAvailable[*perTypeFeatureIdx]) {
                    DstView[*perTypeFeatureIdx][objectIdx] = value;
                }
            }

            template <EFeatureValuesType ColumnType>
            void GetResult(
                const TFeaturesLayout& featuresLayout,
                const TFeaturesArraySubsetIndexing* subsetIndexing,
                TVector<THolder<TArrayValuesHolder<T, ColumnType>>>* result
            ) {
                CB_ENSURE_INTERNAL(Storage.size() == DstView.size(), "Storage is inconsistent with DstView");

                const size_t featureCount = (size_t)featuresLayout.GetFeatureCount(FeatureType);

                CB_ENSURE_INTERNAL(
                    Storage.size() == featureCount,
                    "Storage is inconsistent with feature Layout"
                );

                result->clear();
                result->reserve(featureCount);
                for (auto perTypeFeatureIdx : xrange(featureCount)) {
                    if (IsAvailable[perTypeFeatureIdx]) {
                        result->push_back(
                            MakeHolder<TArrayValuesHolder<T, ColumnType>>(
                                /* featureId */ (ui32)featuresLayout.GetExternalFeatureIdx(
                                    perTypeFeatureIdx, FeatureType
                                ),
                                TMaybeOwningConstArrayHolder<T>::CreateOwning(
                                    DstView[perTypeFeatureIdx],
                                    Storage[perTypeFeatureIdx]
                                ),
                                subsetIndexing
                            )
                        );
                    } else {
                        result->push_back(nullptr);
                    }
                }
            }
        };

    private:
        bool InBlock;

        ui32 ObjectCount;
        ui32 CatFeatureCount;

        TRawBuilderData Data;

        // will be moved to Data.RawTargetData at GetResult
        TVector<float> WeightsBuffer;
        TVector<float> GroupWeightsBuffer;

        TFeaturesStorage<EFeatureType::Float, float> FloatFeaturesStorage;
        TFeaturesStorage<EFeatureType::Categorical, ui32> CatFeaturesStorage;

        std::array<THashPart, CB_THREAD_LIMIT> HashMapParts;


        static constexpr const ui32 NotSet = Max<ui32>();
        ui32 Cursor;
        ui32 NextCursor;

        TDataProviderBuilderOptions Options;

        NPar::TLocalExecutor* LocalExecutor;

        bool InProcess;
        bool ResultTaken;
    };


    class TRawFeaturesOrderDataProviderBuilder : public IDataProviderBuilder,
                                                 public IRawFeaturesOrderDataVisitor
    {
    public:
        static constexpr int OBJECT_CALC_BLOCK_SIZE = 10000;

        TRawFeaturesOrderDataProviderBuilder(
            const TDataProviderBuilderOptions& options,
            NPar::TLocalExecutor* localExecutor
        )
            : ObjectCount(0)
            , Options(options)
            , LocalExecutor(localExecutor)
            , InProcess(false)
            , ResultTaken(false)
        {}

        void Start(
            const TDataMetaInfo& metaInfo,
            ui32 objectCount,
            EObjectsOrder objectsOrder,

            // keep necessary resources for data to be available (memory mapping for a file for example)
            TVector<TIntrusivePtr<IResourceHolder>> resourceHolders
        ) override {
            CB_ENSURE(!InProcess, "Attempt to start new processing without finishing the last");
            InProcess = true;
            ResultTaken = false;

            ObjectCount = objectCount;


            ObjectCalcParams.Reset(
                new NPar::TLocalExecutor::TExecRangeParams(0, SafeIntegerCast<int>(ObjectCount))
            );
            ObjectCalcParams->SetBlockSize(OBJECT_CALC_BLOCK_SIZE);

            Data.MetaInfo = metaInfo;
            Data.TargetData.PrepareForInitialization(metaInfo, ObjectCount, 0);
            Data.CommonObjectsData.PrepareForInitialization(metaInfo, ObjectCount, 0);
            Data.ObjectsData.PrepareForInitialization(metaInfo);

            Data.CommonObjectsData.ResourceHolders = std::move(resourceHolders);
            Data.CommonObjectsData.Order = objectsOrder;

            Data.CommonObjectsData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(ObjectCount)
            );
        }

        // TCommonObjectsData
        void AddGroupId(ui32 objectIdx, TGroupId value) override {
            (*Data.CommonObjectsData.GroupIds)[objectIdx] = value;
        }

        void AddSubgroupId(ui32 objectIdx, TSubgroupId value) override {
            (*Data.CommonObjectsData.SubgroupIds)[objectIdx] = value;
        }

        void AddTimestamp(ui32 objectIdx, ui64 value) override {
            (*Data.CommonObjectsData.Timestamp)[objectIdx] = value;
        }

        // TRawObjectsData
        void AddFloatFeature(ui32 flatFeatureIdx, TMaybeOwningConstArrayHolder<float> features) override {
            auto floatFeatureIdx = GetInternalFeatureIdx<EFeatureType::Float>(flatFeatureIdx);
            Data.ObjectsData.FloatFeatures[*floatFeatureIdx] = MakeHolder<TFloatValuesHolder>(
                flatFeatureIdx,
                std::move(features),
                Data.CommonObjectsData.SubsetIndexing.Get()
            );
        }

        void AddCatFeature(ui32 flatFeatureIdx, TConstArrayRef<TString> feature) override {
            AddCatFeatureImpl(flatFeatureIdx, feature);
        }
        void AddCatFeature(ui32 flatFeatureIdx, TConstArrayRef<TStringBuf> feature) override {
            AddCatFeatureImpl(flatFeatureIdx, feature);
        }

        void AddCatFeature(ui32 flatFeatureIdx, TMaybeOwningConstArrayHolder<ui32> features) override {
            auto catFeatureIdx = GetInternalFeatureIdx<EFeatureType::Categorical>(flatFeatureIdx);
            Data.ObjectsData.CatFeatures[*catFeatureIdx] = MakeHolder<THashedCatValuesHolder>(
                flatFeatureIdx,
                std::move(features),
                Data.CommonObjectsData.SubsetIndexing.Get()
            );
        }


        // TRawTargetData

        void AddTarget(TConstArrayRef<TString> value) override {
            Data.TargetData.Target->assign(value.begin(), value.end());
        }
        void AddTarget(TConstArrayRef<float> value) override {
            TArrayRef<TString> target = *Data.TargetData.Target;

            LocalExecutor->ExecRange(
                [&](int objectIdx) {
                    target[objectIdx] = ToString(value[objectIdx]);
                },
                *ObjectCalcParams,
                NPar::TLocalExecutor::WAIT_COMPLETE
            );

        }
        void AddBaseline(ui32 baselineIdx, TConstArrayRef<float> value) override {
            Data.TargetData.Baseline[baselineIdx].assign(value.begin(), value.end());
        }
        void AddWeights(TConstArrayRef<float> value) override {
            Data.TargetData.Weights = TWeights<float>(TVector<float>(value.begin(), value.end()));
        }
        void AddGroupWeights(TConstArrayRef<float> value) override {
            Data.TargetData.GroupWeights = TWeights<float>(
                TVector<float>(value.begin(), value.end()),
                "GroupWeights"
            );
        }

        // separate method because they can be loaded from a separate data source
        void SetGroupWeights(TVector<float>&& groupWeights) override {
            CheckDataSize(groupWeights.size(), (size_t)ObjectCount, "groupWeights");
            Data.TargetData.GroupWeights = TWeights<float>(std::move(groupWeights), "GroupWeights");
        }

        void SetPairs(TVector<TPair>&& pairs) override {
            Data.TargetData.Pairs = std::move(pairs);
        }

        // needed for checking groupWeights consistency while loading from separate file
        TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const override {
            return Data.CommonObjectsData.GroupIds;
        }

        void Finish() override {
            CB_ENSURE(InProcess, "Attempt to Finish without starting processing");
            InProcess = false;
        }

        TDataProviderPtr GetResult() override {
            CB_ENSURE_INTERNAL(!InProcess, "Attempt to GetResult before finishing processing");
            CB_ENSURE_INTERNAL(!ResultTaken, "Attempt to GetResult several times");

            ResultTaken = true;

            return MakeDataProvider<TRawObjectsDataProvider>(
                /*objectsGrouping*/ Nothing(), // will init from data
                std::move(Data),
                Options.SkipCheck,
                LocalExecutor
            )->CastMoveTo<TObjectsDataProvider>();
        }

    private:
        template <EFeatureType FeatureType>
        TFeatureIdx<FeatureType> GetInternalFeatureIdx(ui32 flatFeatureIdx) const {
            return TFeatureIdx<FeatureType>(
                Data.MetaInfo.FeaturesLayout->GetInternalFeatureIdx(flatFeatureIdx)
            );
        }

        template <class TStringLike>
        void AddCatFeatureImpl(ui32 flatFeatureIdx, TConstArrayRef<TStringLike> feature) {
            auto catFeatureIdx = GetInternalFeatureIdx<EFeatureType::Categorical>(flatFeatureIdx);

            TVector<ui32> hashedCatValues;
            hashedCatValues.yresize(ObjectCount);

            LocalExecutor->ExecRange(
                [&](int objectIdx) {
                    hashedCatValues[objectIdx] = CalcCatFeatureHash(feature[objectIdx]);
                },
                *ObjectCalcParams,
                NPar::TLocalExecutor::WAIT_COMPLETE
            );

            auto& catFeatureHash = (*Data.CommonObjectsData.CatFeaturesHashToString)[*catFeatureIdx];

            for (auto objectIdx : xrange(ObjectCount)) {
                const ui32 hashedValue = hashedCatValues[objectIdx];
                THashMap<ui32, TString>::insert_ctx insertCtx;
                if (!catFeatureHash.contains(hashedValue, insertCtx)) {
                    catFeatureHash.emplace_direct(insertCtx, hashedValue, feature[objectIdx]);
                }
            }

            Data.ObjectsData.CatFeatures[*catFeatureIdx] = MakeHolder<THashedCatValuesHolder>(
                flatFeatureIdx,
                TMaybeOwningConstArrayHolder<ui32>::CreateOwning(std::move(hashedCatValues)),
                Data.CommonObjectsData.SubsetIndexing.Get()
            );
        }


    private:
        ui32 ObjectCount;

        TRawBuilderData Data;

        TDataProviderBuilderOptions Options;

        NPar::TLocalExecutor* LocalExecutor;

        // have to make it THolder because NPar::TLocalExecutor::TExecRangeParams is unassignable/unmoveable
        THolder<NPar::TLocalExecutor::TExecRangeParams> ObjectCalcParams;

        bool InProcess;
        bool ResultTaken;
    };


    class TQuantizedFeaturesDataProviderBuilder : public IDataProviderBuilder,
                                                  public IQuantizedFeaturesDataVisitor
    {
    public:
        TQuantizedFeaturesDataProviderBuilder(
            const TDataProviderBuilderOptions& options,
            NPar::TLocalExecutor* localExecutor
        )
            : ObjectCount(0)
            , Options(options)
            , LocalExecutor(localExecutor)
            , InProcess(false)
            , ResultTaken(false)
        {}

        void Start(
            const TDataMetaInfo& metaInfo,
            ui32 objectCount,
            EObjectsOrder objectsOrder,

            // keep necessary resources for data to be available (memory mapping for a file for example)
            TVector<TIntrusivePtr<IResourceHolder>> resourceHolders,

            const NCB::TPoolQuantizationSchema& poolQuantizationSchema
        ) override {
            CB_ENSURE(!InProcess, "Attempt to start new processing without finishing the last");

            CB_ENSURE(!poolQuantizationSchema.FeatureIndices.empty(), "No features in quantized pool!");

            InProcess = true;
            ResultTaken = false;

            ObjectCount = objectCount;

            ClassNames = poolQuantizationSchema.ClassNames;

            Data.MetaInfo = metaInfo;
            Data.TargetData.PrepareForInitialization(metaInfo, objectCount, 0);
            Data.CommonObjectsData.PrepareForInitialization(metaInfo, objectCount, 0);
            Data.ObjectsData.PrepareForInitialization(
                metaInfo,

                // TODO(akhropov): get from quantized pool meta info when it will be available: MLTOOLS-2392.
                NCatboostOptions::TBinarizationOptions(
                    EBorderSelectionType::GreedyLogSum, // default value
                    SafeIntegerCast<ui32>(poolQuantizationSchema.Borders[0].size()),
                    ENanMode::Forbidden // default value
                )
            );

            FillQuantizedFeaturesInfo(
                poolQuantizationSchema,
                Data.ObjectsData.QuantizedFeaturesInfo.Get());

            Data.CommonObjectsData.ResourceHolders = std::move(resourceHolders);
            Data.CommonObjectsData.Order = objectsOrder;

            FloatFeaturesStorage.PrepareForInitialization(
                *metaInfo.FeaturesLayout,
                objectCount,
                poolQuantizationSchema
            );

            if (metaInfo.HasWeights) {
                WeightsBuffer.yresize(objectCount);
            }
            if (metaInfo.HasGroupWeight) {
                GroupWeightsBuffer.yresize(objectCount);
            }
        }

        template <class T>
        static void CopyPart(ui32 objectOffset, TUnalignedArrayBuf<T> srcPart, TVector<T>* dstData) {
            CB_ENSURE_INTERNAL(
                objectOffset < dstData->size(),
                LabeledOutput(objectOffset, srcPart.GetSize(), dstData->size()));
            CB_ENSURE_INTERNAL(
                objectOffset + srcPart.GetSize() <= dstData->size(),
                LabeledOutput(objectOffset, srcPart.GetSize(), dstData->size()));

            TArrayRef<T> dstArrayRef(dstData->data() + objectOffset, srcPart.GetSize());
            srcPart.WriteTo(&dstArrayRef);
        }

        // TCommonObjectsData
        void AddGroupIdPart(ui32 objectOffset, TUnalignedArrayBuf<TGroupId> groupIdPart) override {
            CopyPart(objectOffset, groupIdPart, &(*Data.CommonObjectsData.GroupIds));
        }

        void AddSubgroupIdPart(ui32 objectOffset, TUnalignedArrayBuf<TSubgroupId> subgroupIdPart) override {
            CopyPart(objectOffset, subgroupIdPart, &(*Data.CommonObjectsData.SubgroupIds));
        }

        void AddTimestampPart(ui32 objectOffset, TUnalignedArrayBuf<ui64> timestampPart) override {
            CopyPart(objectOffset, timestampPart, &(*Data.CommonObjectsData.Timestamp));
        }

        void AddFloatFeaturePart(
            ui32 flatFeatureIdx,
            ui32 objectOffset,
            TMaybeOwningConstArrayHolder<ui8> featuresPart // per-feature data size depends on BitsPerKey
        ) override {
            FloatFeaturesStorage.Set(
                GetInternalFeatureIdx<EFeatureType::Float>(flatFeatureIdx),
                objectOffset,
                *featuresPart
            );
        }

        void AddCatFeaturePart(
            ui32 flatFeatureIdx,
            ui32 objectOffset,
            TMaybeOwningConstArrayHolder<ui8> featuresPart // per-feature data size depends on BitsPerKey
        ) override {
            Y_UNUSED(flatFeatureIdx);
            Y_UNUSED(objectOffset);
            Y_UNUSED(featuresPart);
            CB_ENSURE(false, "Categorical features are not yet supported in serialized quantized pools");
        }

        // TRawTargetData

        void AddTargetPart(ui32 objectOffset, TUnalignedArrayBuf<float> targetPart) override {
            auto& target = *Data.TargetData.Target;
            for (auto it = targetPart.GetIterator(); !it.AtEnd(); it.Next(), ++objectOffset) {
                target[objectOffset] = ToString(it.Cur());
            }
        }

        void AddTargetPart(ui32 objectOffset, TMaybeOwningConstArrayHolder<TString> targetPart) override {
            Copy(
                (*targetPart).begin(),
                (*targetPart).end(),
                std::next((*Data.TargetData.Target).begin(), objectOffset)
            );
        }

        void AddBaselinePart(
            ui32 objectOffset,
            ui32 baselineIdx,
            TUnalignedArrayBuf<float> baselinePart
        ) override {
            CopyPart(objectOffset, baselinePart, &Data.TargetData.Baseline[baselineIdx]);
        }

        void AddWeightPart(ui32 objectOffset, TUnalignedArrayBuf<float> weightPart) override {
            CopyPart(objectOffset, weightPart, &WeightsBuffer);
        }

        void AddGroupWeightPart(ui32 objectOffset, TUnalignedArrayBuf<float> groupWeightPart) override {
            CopyPart(objectOffset, groupWeightPart, &GroupWeightsBuffer);
        }

        // separate method because they can be loaded from a separate data source
        void SetGroupWeights(TVector<float>&& groupWeights) override {
            CheckDataSize(groupWeights.size(), (size_t)ObjectCount, "groupWeights");
            GroupWeightsBuffer = std::move(groupWeights);
        }

        void SetPairs(TVector<TPair>&& pairs) override {
            Data.TargetData.Pairs = std::move(pairs);
        }

        // needed for checking groupWeights consistency while loading from separate file
        TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const override {
            return Data.CommonObjectsData.GroupIds;
        }

        void Finish() override {
            CB_ENSURE(InProcess, "Attempt to Finish without starting processing");

            if (ObjectCount != 0) {
                CATBOOST_INFO_LOG << "Object info sizes: " << ObjectCount << " "
                    << Data.MetaInfo.FeaturesLayout->GetExternalFeatureCount() << Endl;
            } else {
                // should this be an error?
                CATBOOST_ERROR_LOG << "No objects info loaded" << Endl;
            }
            InProcess = false;
        }

        TDataProviderPtr GetResult() override {
            CB_ENSURE_INTERNAL(!InProcess, "Attempt to GetResult before finishing processing");
            CB_ENSURE_INTERNAL(!ResultTaken, "Attempt to GetResult several times");

            if (Data.MetaInfo.HasWeights) {
                Data.TargetData.Weights = TWeights<float>(std::move(WeightsBuffer));
            }
            if (Data.MetaInfo.HasGroupWeight) {
                Data.TargetData.GroupWeights = TWeights<float>(std::move(GroupWeightsBuffer));
            }

            Data.CommonObjectsData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(ObjectCount)
            );

            FloatFeaturesStorage.GetResult(
                ObjectCount,
                *Data.MetaInfo.FeaturesLayout,
                Data.CommonObjectsData.SubsetIndexing.Get(),
                &Data.ObjectsData.FloatFeatures
            );

            ResultTaken = true;

            if (Options.CpuCompatibleFormat) {
                return MakeDataProvider<TQuantizedForCPUObjectsDataProvider>(
                    /*objectsGrouping*/ Nothing(), // will init from data
                    std::move(Data),
                    Options.SkipCheck,
                    LocalExecutor
                )->CastMoveTo<TObjectsDataProvider>();
            } else {
                return MakeDataProvider<TQuantizedObjectsDataProvider>(
                    /*objectsGrouping*/ Nothing(), // will init from data
                    std::move(Data),
                    Options.SkipCheck,
                    LocalExecutor
                )->CastMoveTo<TObjectsDataProvider>();
            }
        }

    private:
        template <EFeatureType FeatureType>
        TFeatureIdx<FeatureType> GetInternalFeatureIdx(ui32 flatFeatureIdx) const {
            return TFeatureIdx<FeatureType>(
                Data.MetaInfo.FeaturesLayout->GetInternalFeatureIdx(flatFeatureIdx)
            );
        }

        static TVector<float> Slice(
            const TVector<float>& v,
            const size_t offset,
            const size_t size)
        {
            CB_ENSURE_INTERNAL(offset < v.size(), LabeledOutput(v.size(), offset));

            TVector<float> slice;
            slice.reserve(size);
            for (size_t i = 0; i < size && offset + i < v.size(); ++i) {
                slice.push_back(v[offset + i]);
            }
            return slice;
        }

        static void FillQuantizedFeaturesInfo(
            const NCB::TPoolQuantizationSchema& schema,
            TQuantizedFeaturesInfo* info
        ) {
            const auto& featuresLayout = *info->GetFeaturesLayout();
            const auto metaInfos = featuresLayout.GetExternalFeaturesMetaInfo();
            for (size_t i = 0, iEnd = schema.FeatureIndices.size(); i < iEnd; ++i) {
                const auto flatFeatureIdx = schema.FeatureIndices[i];
                const auto nanMode = schema.NanModes[i];
                const auto& metaInfo = metaInfos[flatFeatureIdx];
                CB_ENSURE(
                    metaInfo.Type == EFeatureType::Float,
                    "quantization schema's feature type for feature " LabeledOutput(flatFeatureIdx)
                    << " (float) is inconsistent with features layout");
                if (!metaInfo.IsAvailable) {
                    continue;
                }

                const auto typedFeatureIdx = featuresLayout.GetInternalFeatureIdx<EFeatureType::Float>(
                    flatFeatureIdx);
                const auto remapper = featuresLayout.GetQuantizedFeatureIndexRemapper(flatFeatureIdx);

                if (!remapper.HasArtificialFeatures()) {
                    info->SetBorders(typedFeatureIdx, TVector<float>(schema.Borders[i]));
                    info->SetNanMode(typedFeatureIdx, nanMode);
                    continue;
                }

                const auto bordersPerArtificialFeature = remapper.GetBinsPerArtificialFeature() - 1;
                info->SetBorders(typedFeatureIdx, Slice(schema.Borders[i], 0, bordersPerArtificialFeature));
                if (ENanMode::Min == nanMode) {
                    info->SetNanMode(typedFeatureIdx, ENanMode::Min);
                } else {
                    info->SetNanMode(typedFeatureIdx, nanMode);
                }

                for (ui32 j = 0, jEnd = remapper.GetArtificialFeatureCount(); j < jEnd; ++j) {
                    const auto typedArtificialFeatureIdx =
                        featuresLayout.GetInternalFeatureIdx<EFeatureType::Float>(remapper.GetIdxOffset() + j);
                    info->SetBorders(typedArtificialFeatureIdx, Slice(
                            schema.Borders[i],
                            (i + 1) * bordersPerArtificialFeature,
                            bordersPerArtificialFeature));
                    info->SetNanMode(typedArtificialFeatureIdx, ENanMode::Max);
                }
            }
        }

    private:
        template <EFeatureType FeatureType>
        class TFeaturesStorage {
        private:
            static_assert(FeatureType == EFeatureType::Float, "Only float features are currently supported");

            /* shared between this builder and created data provider for efficiency
             * (can be reused for cache here if there's no more references to data
             *  (data provider was freed))
             */

            TVector<TIntrusivePtr<TVectorHolder<ui64>>> Storage; // [perTypeFeatureIdx]

            // view into storage for faster access
            TVector<TArrayRef<ui64>> DstView; // [perTypeFeatureIdx]

            // copy from Data.MetaInfo.FeaturesLayout for fast access
            TVector<bool> IsAvailable; // [perTypeFeatureIdx]

            TVector<TIndexHelper<ui64>> IndexHelpers; // [perTypeFeatureIdx]

        public:
            void PrepareForInitialization(
                const TFeaturesLayout& featuresLayout,
                ui32 objectCount,
                const TPoolQuantizationSchema& quantizationSchema
            ) {
                Y_UNUSED(quantizationSchema);
                const size_t perTypeFeatureCount = (size_t)featuresLayout.GetFeatureCount(FeatureType);
                Storage.resize(perTypeFeatureCount);
                DstView.resize(perTypeFeatureCount);
                IsAvailable.resize(perTypeFeatureCount, false); // filled from quantization Schema, then checked
                IndexHelpers.resize(perTypeFeatureCount, TIndexHelper<ui64>(8));

                const auto metaInfos = featuresLayout.GetExternalFeaturesMetaInfo();
                for (size_t flatFeatureIdx = 0; flatFeatureIdx < metaInfos.size(); ++flatFeatureIdx) {
                    if (!metaInfos[flatFeatureIdx].IsAvailable) {
                        continue;
                    }

                    const auto typedFeatureIdx = featuresLayout.GetInternalFeatureIdx<FeatureType>(
                        flatFeatureIdx);

                    IsAvailable[typedFeatureIdx.Idx] = true;
                    IndexHelpers[typedFeatureIdx.Idx] = TIndexHelper<ui64>(8);
                }

                for (auto perTypeFeatureIdx : xrange(perTypeFeatureCount)) {
                    if (featuresLayout.GetInternalFeatureMetaInfo(
                            perTypeFeatureIdx,
                            FeatureType
                        ).IsAvailable)
                    {
                        CB_ENSURE(
                            IsAvailable[perTypeFeatureIdx],
                            FeatureType << " feature #" << perTypeFeatureIdx
                            << " has no data in quantized pool"
                        );

                        auto& maybeSharedStoragePtr = Storage[perTypeFeatureIdx];
                        if (!maybeSharedStoragePtr || (maybeSharedStoragePtr->RefCount() > 1)) {
                            /* storage is either uninited or shared with some other references
                             * so it has to be reset to be reused
                             */
                            Storage[perTypeFeatureIdx] = MakeIntrusive<TVectorHolder<ui64>>();
                        }
                        maybeSharedStoragePtr->Data.yresize(
                            IndexHelpers[perTypeFeatureIdx].CompressedSize(objectCount)
                        );
                        DstView[perTypeFeatureIdx] =  maybeSharedStoragePtr->Data;
                    } else {
                        Storage[perTypeFeatureIdx] = nullptr;
                        DstView[perTypeFeatureIdx] = TArrayRef<ui64>();
                    }
                }
            }

            void Set(
                TFeatureIdx<FeatureType> perTypeFeatureIdx,
                ui32 objectOffset,
                TConstArrayRef<ui8> featuresPart
            ) {
                if (!IsAvailable[*perTypeFeatureIdx]) {
                    return;
                }

                const auto dstCapacityInBytes =
                    DstView[*perTypeFeatureIdx].size() *
                    sizeof(decltype(*DstView[*perTypeFeatureIdx].data()));
                const auto objectOffsetInBytes = objectOffset * sizeof(ui8);

                CB_ENSURE_INTERNAL(
                    objectOffsetInBytes < dstCapacityInBytes,
                    LabeledOutput(perTypeFeatureIdx, objectOffset, objectOffsetInBytes, featuresPart.size(), dstCapacityInBytes));
                CB_ENSURE_INTERNAL(
                    objectOffsetInBytes + featuresPart.size() <= dstCapacityInBytes,
                    LabeledOutput(perTypeFeatureIdx, objectOffset, objectOffsetInBytes, featuresPart.size(), dstCapacityInBytes));

                memcpy(
                    ((ui8*)DstView[*perTypeFeatureIdx].data()) + objectOffset,
                    featuresPart.data(),
                    featuresPart.size());
            }

            template <class IColumnType>
            void GetResult(
                ui32 objectCount,
                const TFeaturesLayout& featuresLayout,
                const TFeaturesArraySubsetIndexing* subsetIndexing,
                TVector<THolder<IColumnType>>* result
            ) {
                CB_ENSURE_INTERNAL(
                    Storage.size() == DstView.size(),
                    "Storage is inconsistent with DstView; "
                    LabeledOutput(Storage.size(), DstView.size()));

                const size_t featureCount = (size_t)featuresLayout.GetFeatureCount(FeatureType);

                CB_ENSURE_INTERNAL(
                    Storage.size() == featureCount,
                    "Storage is inconsistent with feature Layout; "
                    LabeledOutput(Storage.size(), featureCount));

                result->clear();
                result->reserve(featureCount);
                for (auto perTypeFeatureIdx : xrange(featureCount)) {
                    if (IsAvailable[perTypeFeatureIdx]) {
                        result->push_back(
                            MakeHolder<TCompressedValuesHolderImpl<IColumnType>>(
                                /* featureId */ featuresLayout.GetExternalFeatureIdx(
                                    perTypeFeatureIdx, FeatureType
                                ),
                                TCompressedArray(
                                    objectCount,
                                    IndexHelpers[perTypeFeatureIdx].GetBitsPerKey(),
                                    TMaybeOwningArrayHolder<ui64>::CreateOwning(
                                        DstView[perTypeFeatureIdx],
                                        Storage[perTypeFeatureIdx]
                                    )
                                ),
                                subsetIndexing
                            )
                        );
                    } else {
                        result->push_back(nullptr);
                    }
                }
            }
        };

    private:
        ui32 ObjectCount;

        /* for conversion back to string representation
         * if empty - no conversion
         *
         *  TODO(akhropov): conversion back will be removed in MLTOOLS-2393
         */
        TVector<TString> ClassNames;

        TQuantizedBuilderData Data;

        // will be moved to Data.RawTargetData at GetResult
        TVector<float> WeightsBuffer;
        TVector<float> GroupWeightsBuffer;

        TFeaturesStorage<EFeatureType::Float> FloatFeaturesStorage;

        TDataProviderBuilderOptions Options;

        NPar::TLocalExecutor* LocalExecutor;

        bool InProcess;
        bool ResultTaken;
    };


    THolder<IDataProviderBuilder> CreateDataProviderBuilder(
        EDatasetVisitorType visitorType,
        const TDataProviderBuilderOptions& options,
        NPar::TLocalExecutor* localExecutor
    ) {
        switch (visitorType) {
            case EDatasetVisitorType::RawObjectsOrder:
                return MakeHolder<TRawObjectsOrderDataProviderBuilder>(options, localExecutor);
            case EDatasetVisitorType::RawFeaturesOrder:
                return MakeHolder<TRawFeaturesOrderDataProviderBuilder>(options, localExecutor);
            case EDatasetVisitorType::QuantizedFeatures:
                return MakeHolder<TQuantizedFeaturesDataProviderBuilder>(options, localExecutor);
            default:
                return nullptr;
        }
    }


    TDataProviderPtr CreateDataProviderFromFeaturesOrderData(TVector<TVector<float>>&& floatFeatures) {
        const ui32 objectsCount = floatFeatures.empty() ? ui32(0) : (ui32)floatFeatures[0].size();

        TDataProviderBuilderOptions options;
        options.SkipCheck = true;
        return NCB::CreateDataProvider<IRawFeaturesOrderDataVisitor>(
            [&] (IRawFeaturesOrderDataVisitor* visitor) {
                TDataMetaInfo metaInfo;
                metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                    SafeIntegerCast<ui32>(floatFeatures.size())
                );

                visitor->Start(metaInfo, objectsCount, NCB::EObjectsOrder::Undefined, {});

                for (auto featureIdx : xrange(floatFeatures.size())) {
                    visitor->AddFloatFeature(
                        featureIdx,
                        TMaybeOwningConstArrayHolder<float>::CreateOwning(std::move(floatFeatures[featureIdx]))
                    );
                }

                visitor->Finish();
            },
            options
        );
    }


    TDataProviderPtr CreateDataProviderFromObjectsOrderData(TVector<TVector<float>>&& floatFeatures) {
        const ui32 featuresCount = floatFeatures.empty() ? ui32(0) : (ui32)floatFeatures[0].size();

        TDataProviderBuilderOptions options;
        options.SkipCheck = true;
        return NCB::CreateDataProvider<IRawObjectsOrderDataVisitor>(
            [&] (IRawObjectsOrderDataVisitor* visitor) {
                TDataMetaInfo metaInfo;
                metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(featuresCount);

                visitor->Start(false, metaInfo, (ui32)floatFeatures.size(), NCB::EObjectsOrder::Undefined, {});
                visitor->StartNextBlock((ui32)floatFeatures.size());

                for (auto objectIdx : xrange(floatFeatures.size())) {
                    visitor->AddAllFloatFeatures(objectIdx, floatFeatures[objectIdx]);
                }

                visitor->Finish();
            },
            options
        );
    }
}
