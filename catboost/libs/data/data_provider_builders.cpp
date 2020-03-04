#include "data_provider_builders.h"

#include "data_provider.h"
#include "feature_index.h"
#include "lazy_columns.h"
#include "objects.h"
#include "sparse_columns.h"
#include "target.h"
#include "util.h"
#include "visitor.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/compression.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/resource_holder.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/labels/helpers.h>
#include <catboost/private/libs/options/restrictions.h>
#include <catboost/private/libs/quantization/utils.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/xrange.h>
#include <util/generic/ylimits.h>
#include <util/generic/ymath.h>
#include <util/stream/labeled.h>
#include <util/string/join.h>
#include <util/system/yassert.h>

#include <algorithm>
#include <array>


namespace NCB {

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
            bool haveUnknownNumberOfSparseFeatures,
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
            if (haveUnknownNumberOfSparseFeatures) {
                // make a copy because it can be updated
                Data.MetaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(*metaInfo.FeaturesLayout);
            }

            Data.TargetData.PrepareForInitialization(Data.MetaInfo, ObjectCount, prevTailSize);
            Data.CommonObjectsData.PrepareForInitialization(Data.MetaInfo, ObjectCount, prevTailSize);
            Data.ObjectsData.PrepareForInitialization(Data.MetaInfo);

            Data.CommonObjectsData.ResourceHolders = std::move(resourceHolders);
            Data.CommonObjectsData.Order = objectsOrder;

            auto prepareFeaturesStorage = [&] (auto& featuresStorage) {
                featuresStorage.PrepareForInitialization(
                    *Data.MetaInfo.FeaturesLayout,
                    haveUnknownNumberOfSparseFeatures,
                    ObjectCount,
                    prevTailSize,
                    /*dataCanBeReusedForNextBlock*/ InBlock && Data.MetaInfo.HasGroupId,
                    LocalExecutor
                );
            };

            prepareFeaturesStorage(FloatFeaturesStorage);
            prepareFeaturesStorage(CatFeaturesStorage);
            prepareFeaturesStorage(TextFeaturesStorage);

            switch (Data.MetaInfo.TargetType) {
                case ERawTargetType::Float:
                case ERawTargetType::Integer:
                    PrepareForInitialization(
                        Data.MetaInfo.TargetCount,
                        ObjectCount,
                        prevTailSize,
                        &FloatTarget
                    );
                    break;
                case ERawTargetType::String:
                    PrepareForInitialization(
                        Data.MetaInfo.TargetCount,
                        ObjectCount,
                        prevTailSize,
                        &StringTarget
                    );
                    break;
                default:
                    ;
            }

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

        void AddAllFloatFeatures(ui32 localObjectIdx, TConstPolymorphicValuesSparseArray<float, ui32> features) override {
            auto objectIdx = Cursor + localObjectIdx;
            features.ForEachNonDefault(
                [&] (ui32 perTypeFeatureIdx, float value) {
                    FloatFeaturesStorage.Set(TFloatFeatureIdx(perTypeFeatureIdx), objectIdx, value);
                }
            );
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

        void AddAllCatFeatures(ui32 localObjectIdx, TConstPolymorphicValuesSparseArray<ui32, ui32> features) override {
            auto objectIdx = Cursor + localObjectIdx;
            features.ForEachNonDefault(
                [&] (ui32 perTypeFeatureIdx, ui32 value) {
                    CatFeaturesStorage.Set(TCatFeatureIdx(perTypeFeatureIdx), objectIdx, value);
                }
            );
        }

        void AddCatFeatureDefaultValue(ui32 flatFeatureIdx, TStringBuf feature) override {
            auto catFeatureIdx = GetInternalFeatureIdx<EFeatureType::Categorical>(flatFeatureIdx);
            CatFeaturesStorage.SetDefaultValue(catFeatureIdx, GetCatFeatureValue(flatFeatureIdx, feature));
        }

        void AddTextFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) override {
            auto textFeatureIdx = GetInternalFeatureIdx<EFeatureType::Text>(flatFeatureIdx);
            TextFeaturesStorage.Set(
                textFeatureIdx,
                Cursor + localObjectIdx,
                TString(feature)
            );
        }

        void AddTextFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, const TString& feature) override {
            auto textFeatureIdx = GetInternalFeatureIdx<EFeatureType::Text>(flatFeatureIdx);
            TextFeaturesStorage.Set(
                textFeatureIdx,
                Cursor + localObjectIdx,
                feature
            );
        }
        void AddAllTextFeatures(ui32 localObjectIdx, TConstArrayRef<TString> features) override {
            auto objectIdx = Cursor + localObjectIdx;
            for (auto perTypeFeatureIdx : xrange(features.size())) {
                TextFeaturesStorage.Set(
                    TTextFeatureIdx(perTypeFeatureIdx),
                    objectIdx,
                    features[perTypeFeatureIdx]
                );
            }
        }
        void AddAllTextFeatures(ui32 localObjectIdx, TConstPolymorphicValuesSparseArray<TString, ui32> features) override {
            auto objectIdx = Cursor + localObjectIdx;
            features.ForEachNonDefault(
                [&] (ui32 perTypeFeatureIdx, TString value) {
                    TextFeaturesStorage.Set(TTextFeatureIdx(perTypeFeatureIdx), objectIdx, std::move(value));
                }
            );
        }

        // TRawTargetData

        void AddTarget(ui32 localObjectIdx, const TString& value) override {
            AddTarget(0, localObjectIdx, value);
        }
        void AddTarget(ui32 localObjectIdx, float value) override {
            AddTarget(0, localObjectIdx, value);
        }
        void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, const TString& value) override {
            Y_ASSERT(Data.MetaInfo.TargetType == ERawTargetType::String);
            StringTarget[flatTargetIdx][Cursor + localObjectIdx] = value;
        }
        void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, float value) override {
            Y_ASSERT(
                (Data.MetaInfo.TargetType == ERawTargetType::Float) ||
                (Data.MetaInfo.TargetType == ERawTargetType::Integer)
            );
            FloatTarget[flatTargetIdx][Cursor + localObjectIdx] = value;
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

        // separate method because they can be loaded from a separate data source
        void SetBaseline(TVector<TVector<float>>&& baseline) override {
            Data.TargetData.Baseline = std::move(baseline);
        }

        void SetPairs(TVector<TPair>&& pairs) override {
            Data.TargetData.Pairs = std::move(pairs);
        }

        void SetTimestamps(TVector<ui64>&& timestamps) override {
            CheckDataSize(timestamps.size(), (size_t)ObjectCount, "timestamps");
            Data.CommonObjectsData.Timestamp = std::move(timestamps);
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
                // copy, not move target & weights buffers
                if (Data.MetaInfo.TargetType == ERawTargetType::String) {
                    for (auto targetIdx : xrange(Data.MetaInfo.TargetCount)) {
                        Data.TargetData.Target[targetIdx] = StringTarget[targetIdx];
                    }
                } else {
                    for (auto targetIdx : xrange(Data.MetaInfo.TargetCount)) {
                        Data.TargetData.Target[targetIdx] =
                            (ITypedSequencePtr<float>)MakeIntrusive<TTypeCastArrayHolder<float, float>>(
                                TVector<float>(FloatTarget[targetIdx])
                            );
                    }
                }
                if (Data.MetaInfo.HasWeights) {
                    Data.TargetData.Weights = TWeights<float>(
                        TVector<float>(WeightsBuffer),
                        AsStringBuf("Weights"),
                        /*allWeightsCanBeZero*/ true
                    );
                }
                if (Data.MetaInfo.HasGroupWeight) {
                    Data.TargetData.GroupWeights = TWeights<float>(
                        TVector<float>(GroupWeightsBuffer),
                        AsStringBuf("GroupWeights"),
                        /*allWeightsCanBeZero*/ true
                    );
                }
            } else {
                if (Data.MetaInfo.TargetType == ERawTargetType::String) {
                    for (auto targetIdx : xrange(Data.MetaInfo.TargetCount)) {
                        Data.TargetData.Target[targetIdx] = std::move(StringTarget[targetIdx]);
                    }
                } else {
                    for (auto targetIdx : xrange(Data.MetaInfo.TargetCount)) {
                        Data.TargetData.Target[targetIdx] =
                            (ITypedSequencePtr<float>)MakeIntrusive<TTypeCastArrayHolder<float, float>>(
                                std::move(FloatTarget[targetIdx])
                            );
                    }
                }
                if (Data.MetaInfo.HasWeights) {
                    Data.TargetData.Weights = TWeights<float>(
                        std::move(WeightsBuffer),
                        AsStringBuf("Weights"),
                        /*allWeightsCanBeZero*/ InBlock
                    );
                }
                if (Data.MetaInfo.HasGroupWeight) {
                    Data.TargetData.GroupWeights = TWeights<float>(
                        std::move(GroupWeightsBuffer),
                        AsStringBuf("GroupWeights"),
                        /*allWeightsCanBeZero*/ InBlock
                    );
                }
            }

            Data.CommonObjectsData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(ObjectCount)
            );

            auto getFeaturesResult = [&] (auto& featuresStorage, auto* dstFeatures) {
                featuresStorage.GetResult(
                    Data.CommonObjectsData.SubsetIndexing.Get(),
                    Options.SparseArrayIndexingType,
                    Data.MetaInfo.FeaturesLayout.Get(),
                    dstFeatures
                );
            };

            getFeaturesResult(FloatFeaturesStorage, &Data.ObjectsData.FloatFeatures);
            getFeaturesResult(CatFeaturesStorage, &Data.ObjectsData.CatFeatures);

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

            getFeaturesResult(TextFeaturesStorage, &Data.ObjectsData.TextFeatures);

            ResultTaken = true;

            if (InBlock && Data.MetaInfo.HasGroupId) {
                auto fullData = MakeDataProvider<TRawObjectsDataProvider>(
                    /*objectsGrouping*/ Nothing(), // will init from data
                    std::move(Data),
                    Options.SkipCheck,
                    LocalExecutor
                );

                TDataProviderPtr result;

                ui32 groupCount = fullData->ObjectsGrouping->GetGroupCount();
                if (groupCount != 1) {
                    TVector<TSubsetBlock<ui32>> subsetBlocks = {
                        TSubsetBlock<ui32>{{ui32(0), groupCount - 1}, 0}
                    };
                    result = fullData->GetSubset(
                        GetSubset(
                            fullData->ObjectsGrouping,
                            TArraySubsetIndexing<ui32>(
                                TRangesSubset<ui32>(groupCount - 1, std::move(subsetBlocks))
                            ),
                            EObjectsOrder::Ordered
                        ),
                        Options.MaxCpuRamUsage,
                        LocalExecutor
                    )->CastMoveTo<TObjectsDataProvider>();
                }

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
            if (groupCount == 1) {
                return fullData->CastMoveTo<TObjectsDataProvider>();
            } else {
                TVector<TSubsetBlock<ui32>> subsetBlocks = {
                    TSubsetBlock<ui32>{{groupCount - 1, groupCount}, 0}
                };
                return fullData->GetSubset(
                    GetSubset(
                        fullData->ObjectsGrouping,
                        TArraySubsetIndexing<ui32>(TRangesSubset<ui32>(1, std::move(subsetBlocks))),
                        EObjectsOrder::Ordered
                    ),
                    Options.MaxCpuRamUsage,
                    LocalExecutor
                )->CastMoveTo<TObjectsDataProvider>();
            }
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
            return Data.MetaInfo.FeaturesLayout->GetExpandingInternalFeatureIdx<FeatureType>(flatFeatureIdx);
        }

    private:
        struct THashPart {
            TVector<THashMap<ui32, TString>> CatFeatureHashes;
        };

        template <EFeatureType FeatureType, class T>
        class TFeaturesStorage {
            struct TSparseIndex2d {
                ui32 PerTypeFeatureIdx;
                ui32 ObjectIdx;
            };

            struct TSparsePart {
                TVector<TSparseIndex2d> Indices;
                TVector<T> Values;
            };

            struct TSparseDataForBuider {
                TVector<ui32> ObjectIndices;
                TVector<T> Values;
            };

            struct TPerFeatureData {
                /* shared between this builder and created data provider for efficiency
                 * (can be reused for cache here if there's no more references to data
                 *  (data provider was freed))
                 */
                TIntrusivePtr<TVectorHolder<T>> DenseDataStorage;

                TArrayRef<T> DenseDstView;

                T DefaultValue; // used only for sparse

                TFeatureMetaInfo MetaInfo;

            public:
                TPerFeatureData() = default;
                TPerFeatureData(TFeatureMetaInfo&& metaInfo)
                    : MetaInfo(std::move(metaInfo))
                {}
            };

            typedef void (*TSetCallback)(
                TFeatureIdx<FeatureType> perTypeFeatureIdx,
                ui32 objectIdx,
                T value,
                TFeaturesStorage* storage
            );

        private:
            ui32 ObjectCount = 0;
            bool HasSparseData = false;
            bool DataCanBeReusedForNextBlock = false;

            NPar::TLocalExecutor* LocalExecutor = nullptr;

            TVector<TPerFeatureData> PerFeatureData; // [perTypeFeatureIdx]

            std::array<TSparsePart, CB_THREAD_LIMIT> SparseDataParts; // [threadId]

            // [perTypeFeaturesIdx] + extra element for adding new features
            TVector<TSetCallback> PerFeatureCallbacks;

        private:
            // set callbacks

            static void SetDenseFeature(
                TFeatureIdx<FeatureType> perTypeFeatureIdx,
                ui32 objectIdx,
                T value,
                TFeaturesStorage* storage
            ) {
                storage->PerFeatureData[*perTypeFeatureIdx].DenseDstView[objectIdx] = value;
            }

            static void SetSparseFeature(
                TFeatureIdx<FeatureType> perTypeFeatureIdx,
                ui32 objectIdx,
                T value,
                TFeaturesStorage* storage
            ) {
                auto& sparseDataPart = storage->SparseDataParts[storage->LocalExecutor->GetWorkerThreadId()];
                sparseDataPart.Indices.emplace_back(TSparseIndex2d{*perTypeFeatureIdx, objectIdx});
                sparseDataPart.Values.emplace_back(value);
            }

            static void IgnoreFeature(
                TFeatureIdx<FeatureType> perTypeFeatureIdx,
                ui32 objectIdx,
                T value,
                TFeaturesStorage* storage
            ) {
                Y_UNUSED(perTypeFeatureIdx, objectIdx, value, storage);
            }

            void PrepareForInitializationSparseParts(ui32 prevObjectCount, ui32 prevTailSize) {
                ui32 objectIdxShift = prevObjectCount - prevTailSize;

                LocalExecutor->ExecRangeWithThrow(
                    [&, objectIdxShift, prevTailSize] (int partIdx) {
                        auto& sparseDataPart = SparseDataParts[partIdx];
                        size_t dstIdx = 0;
                        if (prevTailSize) {
                            for (auto i : xrange(sparseDataPart.Indices.size())) {
                                auto index2d = sparseDataPart.Indices[i];
                                if (index2d.ObjectIdx >= objectIdxShift) {
                                    sparseDataPart.Indices[dstIdx].PerTypeFeatureIdx
                                        = index2d.PerTypeFeatureIdx;
                                    sparseDataPart.Indices[dstIdx].ObjectIdx
                                        = index2d.ObjectIdx - objectIdxShift;
                                    sparseDataPart.Values[dstIdx] = std::move(sparseDataPart.Values[i]);
                                    ++dstIdx;
                                }
                            }
                        }
                        sparseDataPart.Indices.resize(dstIdx);
                        sparseDataPart.Values.resize(dstIdx);
                    },
                    0,
                    (int)SparseDataParts.size(),
                    NPar::TLocalExecutor::WAIT_COMPLETE
                );
            }


            TVector<TMaybe<TConstPolymorphicValuesSparseArray<T, ui32>>> CreateSparseArrays(
                ui32 objectCount,
                ESparseArrayIndexingType sparseArrayIndexingType,
                NPar::TLocalExecutor* localExecutor
            ) {
                TVector<TSparseDataForBuider> sparseDataForBuilders(PerFeatureData.size()); // [perTypeFeatureIdx]

                for (auto& sparseDataPart : SparseDataParts) {
                    for (auto i : xrange(sparseDataPart.Indices.size())) {
                        auto index2d = sparseDataPart.Indices[i];
                        if (index2d.PerTypeFeatureIdx >= sparseDataForBuilders.size()) {
                            // add previously unknown features
                            sparseDataForBuilders.resize(index2d.PerTypeFeatureIdx + 1);
                        }
                        auto& dataForBuilder = sparseDataForBuilders[index2d.PerTypeFeatureIdx];
                        dataForBuilder.ObjectIndices.push_back(index2d.ObjectIdx);
                        dataForBuilder.Values.push_back(std::move(sparseDataPart.Values[i]));
                    }
                    if (!DataCanBeReusedForNextBlock) {
                        sparseDataPart = TSparsePart();
                    }
                }

                TVector<TMaybe<TConstPolymorphicValuesSparseArray<T, ui32>>> result(
                    sparseDataForBuilders.size()
                );

                localExecutor->ExecRangeWithThrow(
                    [&] (int perTypeFeatureIdx) {
                        T defaultValue = T();
                        if ((size_t)perTypeFeatureIdx < PerFeatureData.size()) {
                            if (!PerFeatureData[perTypeFeatureIdx].MetaInfo.IsSparse) {
                                return;
                            }
                            defaultValue = PerFeatureData[perTypeFeatureIdx].DefaultValue;
                        }

                        std::function<TTypedSequenceContainer<T>(TVector<T>&&)> createNonDefaultValues
                            = [&] (TVector<T>&& values) {
                                return TTypedSequenceContainer<T>(
                                    MakeIntrusive<TTypeCastArrayHolder<T, T>>(
                                        TMaybeOwningConstArrayHolder<T>::CreateOwning(std::move(values))
                                    )
                                );
                            };

                        result[perTypeFeatureIdx].ConstructInPlace(
                            MakeSparseArrayBase<const T, TTypedSequenceContainer<T>, ui32>(
                                objectCount,
                                std::move(sparseDataForBuilders[perTypeFeatureIdx].ObjectIndices),
                                std::move(sparseDataForBuilders[perTypeFeatureIdx].Values),
                                std::move(createNonDefaultValues),
                                sparseArrayIndexingType,
                                /*ordered*/ false,
                                std::move(defaultValue)
                            )
                        );
                    },
                    0,
                    SafeIntegerCast<int>(sparseDataForBuilders.size()),
                    NPar::TLocalExecutor::WAIT_COMPLETE
                );

                return result;
            }

        public:
            void PrepareForInitialization(
                const TFeaturesLayout& featuresLayout,
                bool haveUnknownNumberOfSparseFeatures,
                ui32 objectCount,
                ui32 prevTailSize,
                bool dataCanBeReusedForNextBlock,
                NPar::TLocalExecutor* localExecutor
            ) {
                ui32 prevObjectCount = ObjectCount;
                ObjectCount = objectCount;

                DataCanBeReusedForNextBlock = dataCanBeReusedForNextBlock;

                LocalExecutor = localExecutor;

                HasSparseData = haveUnknownNumberOfSparseFeatures;


                const size_t featureCount = (size_t) featuresLayout.GetFeatureCount(FeatureType);
                PerFeatureData.resize(featureCount);
                PerFeatureCallbacks.resize(featureCount + 1);
                for (auto perTypeFeatureIdx : xrange(featureCount)) {
                    auto& perFeatureData = PerFeatureData[perTypeFeatureIdx];
                    perFeatureData.MetaInfo
                        = featuresLayout.GetInternalFeatureMetaInfo(perTypeFeatureIdx, FeatureType);
                    if (perFeatureData.MetaInfo.IsAvailable) {
                        if (perFeatureData.MetaInfo.IsSparse) {
                            HasSparseData = true;
                            PerFeatureCallbacks[perTypeFeatureIdx] = SetSparseFeature;
                        } else {
                            auto& maybeSharedStoragePtr = perFeatureData.DenseDataStorage;

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
                            perFeatureData.DenseDstView = maybeSharedStoragePtr->Data;

                            PerFeatureCallbacks[perTypeFeatureIdx] = SetDenseFeature;
                        }
                    } else {
                        PerFeatureCallbacks[perTypeFeatureIdx] = IgnoreFeature;
                    }
                }
                // for new sparse features
                PerFeatureCallbacks.back() = SetSparseFeature;

                if (HasSparseData) {
                    PrepareForInitializationSparseParts(prevObjectCount, prevTailSize);
                }
            }

            void Set(TFeatureIdx<FeatureType> perTypeFeatureIdx, ui32 objectIdx, T value) {
                PerFeatureCallbacks[Min(size_t(*perTypeFeatureIdx), PerFeatureCallbacks.size() - 1)](
                    perTypeFeatureIdx,
                    objectIdx,
                    value,
                    this
                );
            }

            void SetDefaultValue(TFeatureIdx<FeatureType> perTypeFeatureIdx, T value) {
                if (*perTypeFeatureIdx >= PerFeatureData.size()) {
                    for (auto perTypeFeatureIdx : xrange(PerFeatureData.size(), size_t(*perTypeFeatureIdx + 1))) {
                        Y_UNUSED(perTypeFeatureIdx);
                        PerFeatureData.emplace_back(
                            TFeatureMetaInfo(FeatureType, /*name*/ "", /*isSparse*/ true)
                        );
                    }
                }
                PerFeatureData[*perTypeFeatureIdx].DefaultValue = value;
            }

            template <EFeatureValuesType ColumnType>
            void GetResult(
                const TFeaturesArraySubsetIndexing* subsetIndexing,
                ESparseArrayIndexingType sparseArrayIndexingType,
                TFeaturesLayout* featuresLayout,
                TVector<THolder<TTypedFeatureValuesHolder<T, ColumnType>>>* result
            ) {
                size_t featureCount = (size_t)featuresLayout->GetFeatureCount(FeatureType);

                CB_ENSURE_INTERNAL(
                    PerFeatureData.size() >= featureCount,
                    "PerFeatureData is inconsistent with feature Layout"
                );

                TVector<TMaybe<TConstPolymorphicValuesSparseArray<T, ui32>>> sparseData = CreateSparseArrays(
                    subsetIndexing->Size(),
                    sparseArrayIndexingType,
                    LocalExecutor
                );

                // there are some new sparse features
                for (auto perTypeFeatureIdx : xrange(featureCount, sparseData.size())) {
                    Y_UNUSED(perTypeFeatureIdx);
                    featuresLayout->AddFeature(TFeatureMetaInfo(FeatureType, /*name*/ "", /*isSparse*/ true));
                }

                featureCount = (size_t)featuresLayout->GetFeatureCount(FeatureType);

                result->clear();
                result->reserve(featureCount);
                for (auto perTypeFeatureIdx : xrange(featureCount)) {
                    const ui32 flatFeatureIdx = (ui32)featuresLayout->GetExternalFeatureIdx(
                        perTypeFeatureIdx,
                        FeatureType
                    );
                    const auto& metaInfo = featuresLayout->GetExternalFeatureMetaInfo(flatFeatureIdx);
                    if (metaInfo.IsAvailable) {
                        if (metaInfo.IsSparse) {
                            result->push_back(
                                MakeHolder<TSparsePolymorphicArrayValuesHolder<T, ColumnType>>(
                                    /* featureId */ flatFeatureIdx,
                                    std::move(*(sparseData[perTypeFeatureIdx]))
                                )
                            );
                        } else {
                            result->push_back(
                                MakeHolder<TPolymorphicArrayValuesHolder<T, ColumnType>>(
                                    /* featureId */ flatFeatureIdx,
                                    TMaybeOwningConstArrayHolder<T>::CreateOwning(
                                        PerFeatureData[perTypeFeatureIdx].DenseDstView,
                                        PerFeatureData[perTypeFeatureIdx].DenseDataStorage
                                    ),
                                    subsetIndexing
                                )
                            );
                        }
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
        TVector<TVector<TString>> StringTarget;
        TVector<TVector<float>> FloatTarget;

        // will be moved to Data.RawTargetData at GetResult
        TVector<float> WeightsBuffer;
        TVector<float> GroupWeightsBuffer;

        TFeaturesStorage<EFeatureType::Float, float> FloatFeaturesStorage;
        TFeaturesStorage<EFeatureType::Categorical, ui32> CatFeaturesStorage;
        TFeaturesStorage<EFeatureType::Text, TString> TextFeaturesStorage;

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
        void AddFloatFeature(ui32 flatFeatureIdx, ITypedSequencePtr<float> features) override {
            auto floatFeatureIdx = GetInternalFeatureIdx<EFeatureType::Float>(flatFeatureIdx);
            Data.ObjectsData.FloatFeatures[*floatFeatureIdx] = MakeHolder<TFloatArrayValuesHolder>(
                flatFeatureIdx,
                features->GetSubset(Data.CommonObjectsData.SubsetIndexing.Get())
            );
        }

        void AddFloatFeature(ui32 flatFeatureIdx, TConstPolymorphicValuesSparseArray<float, ui32> features) override {
            auto floatFeatureIdx = GetInternalFeatureIdx<EFeatureType::Float>(flatFeatureIdx);
            Data.ObjectsData.FloatFeatures[*floatFeatureIdx] = MakeHolder<TFloatSparseValuesHolder>(
                flatFeatureIdx,
                std::move(features)
            );
        }

        ui32 GetCatFeatureValue(ui32 flatFeatureIdx, TStringBuf feature) override {
            auto catFeatureIdx = GetInternalFeatureIdx<EFeatureType::Categorical>(flatFeatureIdx);
            auto& catFeatureHash = (*Data.CommonObjectsData.CatFeaturesHashToString)[*catFeatureIdx];

            ui32 hashedValue = CalcCatFeatureHash(feature);
            THashMap<ui32, TString>::insert_ctx insertCtx;
            if (!catFeatureHash.contains(hashedValue, insertCtx)) {
                catFeatureHash.emplace_direct(insertCtx, hashedValue, TString(feature));
            }
            return hashedValue;
        }

        void AddCatFeature(ui32 flatFeatureIdx, TConstArrayRef<TString> feature) override {
            AddCatFeatureImpl(flatFeatureIdx, feature);
        }
        void AddCatFeature(ui32 flatFeatureIdx, TConstArrayRef<TStringBuf> feature) override {
            AddCatFeatureImpl(flatFeatureIdx, feature);
        }

        void AddCatFeature(ui32 flatFeatureIdx, TMaybeOwningConstArrayHolder<ui32> features) override {
            auto catFeatureIdx = GetInternalFeatureIdx<EFeatureType::Categorical>(flatFeatureIdx);
            Data.ObjectsData.CatFeatures[*catFeatureIdx] = MakeHolder<THashedCatArrayValuesHolder>(
                flatFeatureIdx,
                std::move(features),
                Data.CommonObjectsData.SubsetIndexing.Get()
            );
        }

        void AddCatFeature(ui32 flatFeatureIdx, TConstPolymorphicValuesSparseArray<TString, ui32> features) override {
            auto catFeatureIdx = GetInternalFeatureIdx<EFeatureType::Categorical>(flatFeatureIdx);
            Data.ObjectsData.CatFeatures[*catFeatureIdx] = MakeHolder<THashedCatSparseValuesHolder>(
                flatFeatureIdx,
                MakeConstPolymorphicValuesSparseArray(
                    features.GetIndexing(),
                    TMaybeOwningConstArrayHolder<ui32>::CreateOwning(
                        CreateHashedCatValues(catFeatureIdx, features.GetNonDefaultValues().GetImpl())
                    ),
                    GetCatFeatureValue(flatFeatureIdx, features.GetDefaultValue())
                )
            );
        }

        void AddTextFeature(ui32 flatFeatureIdx, TConstArrayRef<TString> features) override {
            auto textFeatureIdx = GetInternalFeatureIdx<EFeatureType::Text>(flatFeatureIdx);
            TVector<TString> featuresCopy(features.begin(), features.end());

            Data.ObjectsData.TextFeatures[*textFeatureIdx] = MakeHolder<TStringTextArrayValuesHolder>(
                flatFeatureIdx,
                TMaybeOwningConstArrayHolder<TString>::CreateOwning(std::move(featuresCopy)),
                Data.CommonObjectsData.SubsetIndexing.Get()
            );
        }

        void AddTextFeature(ui32 flatFeatureIdx, TMaybeOwningConstArrayHolder<TString> features) override {
            auto textFeatureIdx = GetInternalFeatureIdx<EFeatureType::Text>(flatFeatureIdx);
            Data.ObjectsData.TextFeatures[*textFeatureIdx] = MakeHolder<TStringTextArrayValuesHolder>(
                flatFeatureIdx,
                std::move(features),
                Data.CommonObjectsData.SubsetIndexing.Get()
            );
        }

        void AddTextFeature(ui32 flatFeatureIdx, TConstPolymorphicValuesSparseArray<TString, ui32> features) override {
            auto textFeatureIdx = GetInternalFeatureIdx<EFeatureType::Text>(flatFeatureIdx);
            Data.ObjectsData.TextFeatures[*textFeatureIdx] = MakeHolder<TStringTextSparseValuesHolder>(
                flatFeatureIdx,
                std::move(features)
            );
        }

        // TRawTargetData

        void AddTarget(TConstArrayRef<TString> value) override {
            AddTarget(0, value);
        }
        void AddTarget(ITypedSequencePtr<float> value) override {
            AddTarget(0, std::move(value));
        }
        void AddTarget(ui32 flatTargetIdx, TConstArrayRef<TString> value) override {
            Data.TargetData.Target[flatTargetIdx] = TVector<TString>(value.begin(), value.end());
        }
        void AddTarget(ui32 flatTargetIdx, ITypedSequencePtr<float> value) override {
            Data.TargetData.Target[flatTargetIdx] = std::move(value);
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

        // separate method because they can be loaded from a separate data source
        void SetBaseline(TVector<TVector<float>>&& baseline) override {
            Data.TargetData.Baseline = std::move(baseline);
        }

        void SetPairs(TVector<TPair>&& pairs) override {
            Data.TargetData.Pairs = std::move(pairs);
        }

        void SetTimestamps(TVector<ui64>&& timestamps) override {
            CheckDataSize(timestamps.size(), (size_t)ObjectCount, "timestamps");
            Data.CommonObjectsData.Timestamp = timestamps;
        }

        // needed for checking groupWeights consistency while loading from separate file
        TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const override {
            return Data.CommonObjectsData.GroupIds;
        }

        void Finish() override {
            CB_ENSURE(InProcess, "Attempt to Finish without starting processing");
            InProcess = false;

            if (ObjectCount != 0) {
                CATBOOST_INFO_LOG << "Object info sizes: " << ObjectCount << " "
                    << Data.MetaInfo.FeaturesLayout->GetExternalFeatureCount() << Endl;
            } else {
                // should this be an error?
                CATBOOST_ERROR_LOG << "No objects info loaded" << Endl;
            }
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
        TVector<ui32> CreateHashedCatValues(
            TCatFeatureIdx catFeatureIdx,
            const ITypedSequence<TStringLike>& stringValues
        ) {
            TVector<ui32> hashedCatValues;
            hashedCatValues.yresize(stringValues.GetSize());
            TArrayRef<ui32> hashedCatValuesRef = hashedCatValues;

            TSimpleIndexRangesGenerator<ui32> indexRanges(
                TIndexRange<ui32>(stringValues.GetSize()),
                (ui32)OBJECT_CALC_BLOCK_SIZE
            );

            LocalExecutor->ExecRange(
                [&, hashedCatValuesRef](int subRangeIdx) {
                    TIndexRange<ui32> subRange = indexRanges.GetRange((ui32)subRangeIdx);
                    auto blockIterator = stringValues.GetBlockIterator(subRange);
                    ui32 objectIdx = subRange.Begin;
                    while (auto block = blockIterator->Next()) {
                        for (auto stringValue : block) {
                            hashedCatValuesRef[objectIdx++] = CalcCatFeatureHash(stringValue);
                        }
                    }
                },
                0,
                SafeIntegerCast<int>(indexRanges.RangesCount()),
                NPar::TLocalExecutor::WAIT_COMPLETE
            );

            auto& catFeatureHash = (*Data.CommonObjectsData.CatFeaturesHashToString)[*catFeatureIdx];

            auto blockIterator = stringValues.GetBlockIterator();
            ui32 objectIdx = 0;
            while (auto block = blockIterator->Next()) {
                for (auto stringValue : block) {
                    const ui32 hashedValue = hashedCatValues[objectIdx];
                    THashMap<ui32, TString>::insert_ctx insertCtx;
                    if (!catFeatureHash.contains(hashedValue, insertCtx)) {
                        catFeatureHash.emplace_direct(insertCtx, hashedValue, stringValue);
                    }
                    ++objectIdx;
                }
            }

            return hashedCatValues;
        }

        template <class TStringLike>
        void AddCatFeatureImpl(ui32 flatFeatureIdx, TConstArrayRef<TStringLike> feature) {
            auto catFeatureIdx = GetInternalFeatureIdx<EFeatureType::Categorical>(flatFeatureIdx);

            TVector<ui32> hashedCatValues = CreateHashedCatValues(
                catFeatureIdx,
                TTypeCastArrayHolder<TStringLike, TStringLike>(
                    TMaybeOwningConstArrayHolder<TStringLike>::CreateNonOwning(feature)
                )
            );

            Data.ObjectsData.CatFeatures[*catFeatureIdx] = MakeHolder<THashedCatArrayValuesHolder>(
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
            TDatasetSubset loadSubset,
            NPar::TLocalExecutor* localExecutor
        )
            : ObjectCount(0)
            , Options(options)
            , DatasetSubset(loadSubset)
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

            TConstArrayRef<NJson::TJsonValue> schemaClassLabels = poolQuantizationSchema.ClassLabels;

            if (metaInfo.TargetType == ERawTargetType::String) {
                CB_ENSURE(
                    !schemaClassLabels.empty(),
                    "poolQuantizationSchema must have class labels when target data type is String"
                );
                CB_ENSURE(
                    schemaClassLabels[0].GetType() == NJson::JSON_STRING,
                    "poolQuantizationSchema must have string class labels when target data type is String"
                );
                StringClassLabels.reserve(schemaClassLabels.size());
                for (const NJson::TJsonValue& classLabel : schemaClassLabels) {
                    StringClassLabels.push_back(classLabel.GetString());
                }
            } else if ((metaInfo.TargetType != ERawTargetType::None) && !schemaClassLabels.empty()) {
                FloatClassLabels.reserve(schemaClassLabels.size());
                switch (schemaClassLabels[0].GetType()) {
                    case NJson::JSON_INTEGER:
                        CB_ENSURE_INTERNAL(
                            metaInfo.TargetType == ERawTargetType::Integer,
                            "metaInfo.TargetType (" << metaInfo.TargetType
                            << ") is inconsistent with schemaClassLabels type ("
                            << schemaClassLabels[0].GetType() << ')'
                        );
                        for (const NJson::TJsonValue& classLabel : schemaClassLabels) {
                            FloatClassLabels.push_back(static_cast<float>(classLabel.GetInteger()));
                        }
                        break;
                    case NJson::JSON_DOUBLE:
                        CB_ENSURE_INTERNAL(
                            metaInfo.TargetType == ERawTargetType::Float,
                            "metaInfo.TargetType (" << metaInfo.TargetType
                            << ") is inconsistent with schemaClassLabels type ("
                            << schemaClassLabels[0].GetType() << ')'
                        );
                        for (const NJson::TJsonValue& classLabel : schemaClassLabels) {
                            FloatClassLabels.push_back(static_cast<float>(classLabel.GetDouble()));
                        }
                        break;
                    default:
                        CB_ENSURE_INTERNAL(
                            false,
                            "Unexpected JSON label type for numeric target type : "
                            << schemaClassLabels[0].GetType()
                        );
                }
            }

            InProcess = true;
            ResultTaken = false;

            ObjectCount = objectCount;

            Data.MetaInfo = metaInfo;

            if (Data.MetaInfo.ClassLabels.empty()) {
                Data.MetaInfo.ClassLabels.assign(schemaClassLabels.begin(), schemaClassLabels.end());
            } else {
                size_t prefixLength = Min(Data.MetaInfo.ClassLabels.size(), schemaClassLabels.size());
                auto firstGivenLabels = TConstArrayRef<NJson::TJsonValue>(
                    Data.MetaInfo.ClassLabels.begin(),
                    Data.MetaInfo.ClassLabels.begin() + prefixLength
                );
                CB_ENSURE(firstGivenLabels == schemaClassLabels,
                          "Class-names incompatible with quantized pool, expected: " << JoinSeq(",", schemaClassLabels));
            }

            Data.TargetData.PrepareForInitialization(metaInfo, objectCount, 0);
            Data.CommonObjectsData.PrepareForInitialization(metaInfo, objectCount, 0);
            Data.ObjectsData.Data.PrepareForInitialization(
                metaInfo,

                // TODO(akhropov): get from quantized pool meta info when it will be available: MLTOOLS-2392.
                NCatboostOptions::TBinarizationOptions(
                    EBorderSelectionType::GreedyLogSum, // default value
                    SafeIntegerCast<ui32>(poolQuantizationSchema.Borders[0].size()),
                    ENanMode::Forbidden // default value
                ),
                TMap<ui32, NCatboostOptions::TBinarizationOptions>()
            );

            FillQuantizedFeaturesInfo(
                poolQuantizationSchema,
                Data.ObjectsData.Data.QuantizedFeaturesInfo.Get()
            );

            Data.ObjectsData.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
                *metaInfo.FeaturesLayout,
                TVector<TExclusiveFeaturesBundle>() // TODO(akhropov): bundle quantized data
            );

            Data.ObjectsData.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
                *metaInfo.FeaturesLayout,
                *Data.ObjectsData.Data.QuantizedFeaturesInfo,
                Data.ObjectsData.ExclusiveFeatureBundlesData,

                // packed binary features are present only in TQuantizedForCPUObjectsDataProvider
                /*dontPack*/ !Options.CpuCompatibleFormat || Options.GpuDistributedFormat
            );

            Data.ObjectsData.FeaturesGroupsData = TFeatureGroupsData(
                *metaInfo.FeaturesLayout,
                TVector<TFeaturesGroup>() // TODO(ilyzhin): group quantized data
            );

            Data.CommonObjectsData.ResourceHolders = std::move(resourceHolders);
            Data.CommonObjectsData.Order = objectsOrder;

            PrepareBinaryFeaturesStorage();

            if (DatasetSubset.HasFeatures) {
                FloatFeaturesStorage.PrepareForInitialization(
                    *metaInfo.FeaturesLayout,
                    objectCount,
                    Data.ObjectsData.Data.QuantizedFeaturesInfo,
                    BinaryFeaturesStorage,
                    Data.ObjectsData.PackedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex
                );
                CategoricalFeaturesStorage.PrepareForInitialization(
                    *metaInfo.FeaturesLayout,
                    objectCount,
                    Data.ObjectsData.Data.QuantizedFeaturesInfo,
                    BinaryFeaturesStorage,
                    Data.ObjectsData.PackedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex
                );
            }

            if (Data.MetaInfo.TargetType == ERawTargetType::String) {
                PrepareForInitialization(
                    Data.MetaInfo.TargetCount,
                    ObjectCount,
                    /*prevTailSize*/ 0,
                    &StringTarget
                );
            } else {
                PrepareForInitialization(
                    Data.MetaInfo.TargetCount,
                    ObjectCount,
                    /*prevTailSize*/ 0,
                    &FloatTarget
                );
            }

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
            ui8 bitsPerDocumentFeature,
            TMaybeOwningConstArrayHolder<ui8> featuresPart // per-feature data size depends on BitsPerKey
        ) override {
            FloatFeaturesStorage.Set(
                GetInternalFeatureIdx<EFeatureType::Float>(flatFeatureIdx),
                objectOffset,
                bitsPerDocumentFeature,
                *featuresPart,
                LocalExecutor
            );
        }

        void AddCatFeaturePart(
            ui32 flatFeatureIdx,
            ui32 objectOffset,
            ui8 bitsPerDocumentFeature,
            TMaybeOwningConstArrayHolder<ui8> featuresPart // per-feature data size depends on BitsPerKey
        ) override {
            CategoricalFeaturesStorage.Set(
                GetInternalFeatureIdx<EFeatureType::Categorical>(flatFeatureIdx),
                objectOffset,
                bitsPerDocumentFeature,
                *featuresPart,
                LocalExecutor
            );
        }

        // TRawTargetData

        void AddTargetPart(ui32 objectOffset, TUnalignedArrayBuf<float> targetPart) override {
            AddTargetPart(0, objectOffset, targetPart);
        }

        void AddTargetPart(ui32 objectOffset, TMaybeOwningConstArrayHolder<TString> targetPart) override {
            AddTargetPart(0, objectOffset, targetPart);
        }

        void AddTargetPart(ui32 flatTargetIdx, ui32 objectOffset, TMaybeOwningConstArrayHolder<TString> targetPart) override {
            Y_ASSERT(Data.MetaInfo.TargetType == ERawTargetType::String);
            Copy(
                (*targetPart).begin(),
                (*targetPart).end(),
                std::next(StringTarget[flatTargetIdx].begin(), objectOffset)
            );
        }

        void AddTargetPart(ui32 flatTargetIdx, ui32 objectOffset, TUnalignedArrayBuf<float> targetPart) override {
            if (!StringClassLabels.empty()) {
                Y_ASSERT(Data.MetaInfo.TargetType == ERawTargetType::String);
                AddTargetPartWithClassLabels<TString>(
                    objectOffset,
                    targetPart,
                    StringClassLabels,
                    StringTarget[flatTargetIdx]
                );
            } else {
                Y_ASSERT(
                    (Data.MetaInfo.TargetType == ERawTargetType::Integer) ||
                    (Data.MetaInfo.TargetType == ERawTargetType::Float)
                );
                if (!FloatClassLabels.empty()) {
                    AddTargetPartWithClassLabels<float>(
                       objectOffset,
                       targetPart,
                       FloatClassLabels,
                       FloatTarget[flatTargetIdx]
                   );
                } else {
                    TArrayRef<float> dstPart(
                        FloatTarget[flatTargetIdx].data() + objectOffset,
                        targetPart.GetSize()
                    );
                    targetPart.WriteTo(&dstPart);
                }
            }
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

        // separate method because they can be loaded from a separate data source
        void SetBaseline(TVector<TVector<float>>&& baseline) override {
            Data.TargetData.Baseline = std::move(baseline);
        }

        void SetPairs(TVector<TPair>&& pairs) override {
            Data.TargetData.Pairs = std::move(pairs);
        }

        void SetTimestamps(TVector<ui64>&& timestamps) override {
            CheckDataSize(timestamps.size(), (size_t)ObjectCount, "timestamps");
            Data.CommonObjectsData.Timestamp = std::move(timestamps);
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
            GetTargetAndBinaryFeaturesData();

            if (DatasetSubset.HasFeatures) {
                FloatFeaturesStorage.GetResult(
                    ObjectCount,
                    *Data.MetaInfo.FeaturesLayout,
                    Data.CommonObjectsData.SubsetIndexing.Get(),
                    Data.ObjectsData.PackedBinaryFeaturesData.SrcData,
                    &Data.ObjectsData.Data.FloatFeatures
                );

                CategoricalFeaturesStorage.GetResult(
                    ObjectCount,
                    *Data.MetaInfo.FeaturesLayout,
                    Data.CommonObjectsData.SubsetIndexing.Get(),
                    Data.ObjectsData.PackedBinaryFeaturesData.SrcData,
                    &Data.ObjectsData.Data.CatFeatures
                );
            }

            SetResultsTaken();

            if (Options.CpuCompatibleFormat && !Options.GpuDistributedFormat) {
                return MakeDataProvider<TQuantizedForCPUObjectsDataProvider>(
                    /*objectsGrouping*/ Nothing(), // will init from data
                    std::move(Data),
                    Options.SkipCheck || !DatasetSubset.HasFeatures,
                    LocalExecutor
                )->CastMoveTo<TObjectsDataProvider>();
            } else {
                return MakeDataProvider<TQuantizedObjectsDataProvider>(
                    /*objectsGrouping*/ Nothing(), // will init from data
                    CastToBase(std::move(Data)),
                    Options.SkipCheck,
                    LocalExecutor
                )->CastMoveTo<TObjectsDataProvider>();
            }
        }

        void GetTargetAndBinaryFeaturesData() {
            CB_ENSURE_INTERNAL(!InProcess, "Attempt to GetResult before finishing processing");
            CB_ENSURE_INTERNAL(!ResultTaken, "Attempt to GetResult several times");

            if (Data.MetaInfo.TargetType == ERawTargetType::String) {
                for (auto targetIdx : xrange(Data.MetaInfo.TargetCount)) {
                    Data.TargetData.Target[targetIdx] = std::move(StringTarget[targetIdx]);
                }
            } else {
                for (auto targetIdx : xrange(Data.MetaInfo.TargetCount)) {
                    Data.TargetData.Target[targetIdx] =
                        (ITypedSequencePtr<float>)MakeIntrusive<TTypeCastArrayHolder<float, float>>(
                            std::move(FloatTarget[targetIdx])
                        );
                }
            }
            if (Data.MetaInfo.HasWeights) {
                Data.TargetData.Weights = TWeights<float>(std::move(WeightsBuffer));
            }
            if (Data.MetaInfo.HasGroupWeight) {
                Data.TargetData.GroupWeights = TWeights<float>(std::move(GroupWeightsBuffer));
            }

            Data.CommonObjectsData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(ObjectCount)
            );

            GetBinaryFeaturesDataResult();
        }

        TQuantizedForCPUBuilderData& GetDataRef() {
            return Data;
        }

        ui32 GetObjectCount() const {
            return ObjectCount;
        }

        void SetResultsTaken() {
            ResultTaken = true;
        }

        template <EFeatureType FeatureType>
        static TVector<bool> MakeIsAvailable(const TFeaturesLayout& featuresLayout) {
            const size_t perTypeFeatureCount = (size_t)featuresLayout.GetFeatureCount(FeatureType);
            TVector<bool> isAvailable(perTypeFeatureCount, false);
            const auto& metaInfos = featuresLayout.GetExternalFeaturesMetaInfo();
            for (auto perTypeFeatureIdx : xrange(perTypeFeatureCount)) {
                const auto flatFeatureIdx = featuresLayout.GetExternalFeatureIdx(perTypeFeatureIdx, FeatureType);
                if (!metaInfos[flatFeatureIdx].IsAvailable) {
                    continue;
                }
                isAvailable[perTypeFeatureIdx] = true;
            }
            return isAvailable;
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

                info->SetBorders(typedFeatureIdx, TVector<float>(schema.Borders[i]));
                info->SetNanMode(typedFeatureIdx, nanMode);
            }

            for (size_t i = 0, iEnd = schema.CatFeatureIndices.size(); i < iEnd; ++i) {
                const auto flatFeatureIdx = schema.CatFeatureIndices[i];
                const auto& metaInfo = metaInfos[flatFeatureIdx];
                CB_ENSURE(
                    metaInfo.Type == EFeatureType::Categorical,
                    "quantization schema's feature type for feature " LabeledOutput(flatFeatureIdx)
                    << " (categorical) is inconsistent with features layout");
                if (!metaInfo.IsAvailable) {
                    continue;
                }

                const auto typedFeatureIdx = featuresLayout.GetInternalFeatureIdx<EFeatureType::Categorical>(
                    flatFeatureIdx);
                TCatFeaturePerfectHash perfectHash{
                    Nothing(),
                    schema.FeaturesPerfectHash[i]
                };
                info->UpdateCategoricalFeaturesPerfectHash(typedFeatureIdx, std::move(perfectHash));
            }
        }

        void PrepareBinaryFeaturesStorage() {
            auto binaryFeaturesStorageSize = CeilDiv(
                Data.ObjectsData.PackedBinaryFeaturesData.PackedBinaryToSrcIndex.size(),
                sizeof(TBinaryFeaturesPack) * CHAR_BIT
            );
            BinaryFeaturesStorage.resize(binaryFeaturesStorageSize);
            if (!binaryFeaturesStorageSize) {
                return;
            }

            TIndexHelper<ui64> indexHelper(sizeof(TBinaryFeaturesPack) * CHAR_BIT);
            const ui64 storageVectorSize = indexHelper.CompressedSize(ObjectCount);

            LocalExecutor->ExecRangeWithThrow(
                [&] (int i) {
                    auto& binaryFeaturesStorageElement = BinaryFeaturesStorage[i];
                    if (!binaryFeaturesStorageElement || (binaryFeaturesStorageElement->RefCount() > 1)) {
                        /* storage is either uninited or shared with some other references
                         * so it has to be reset to be reused
                         */
                        binaryFeaturesStorageElement = MakeIntrusive<TVectorHolder<ui64>>();
                    }
                    auto& data = binaryFeaturesStorageElement->Data;
                    data.yresize(storageVectorSize);
                    Fill(data.begin(), data.end(), ui64(0));
                },
                0,
                SafeIntegerCast<int>(binaryFeaturesStorageSize),
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
        }

        void GetBinaryFeaturesDataResult() {
            auto& dst = Data.ObjectsData.PackedBinaryFeaturesData.SrcData;
            dst.clear();
            for (auto& binaryFeaturesStorageElement : BinaryFeaturesStorage) {
                dst.push_back(
                    MakeHolder<TBinaryPacksArrayHolder>(
                        0,
                        TCompressedArray(
                            ObjectCount,
                            sizeof(TBinaryFeaturesPack) * CHAR_BIT,
                            TMaybeOwningArrayHolder<ui64>::CreateOwning(
                                binaryFeaturesStorageElement->Data,
                                binaryFeaturesStorageElement
                            )
                        ),
                        Data.CommonObjectsData.SubsetIndexing.Get()
                    )
                );
            }
        }

        template <class T>
        static void AddTargetPartWithClassLabels(
            ui32 objectOffset,
            TUnalignedArrayBuf<float> targetPart,
            TConstArrayRef<T> classLabels,
            TArrayRef<T> target
        ) {
            for (auto it = targetPart.GetIterator(); !it.AtEnd(); it.Next(), ++objectOffset) {
                target[objectOffset] = classLabels[int(it.Cur())];
            }
        }

    private:
        // ui64 because passed to TCompressedArray later
        using TBinaryFeaturesStorage = TVector<TIntrusivePtr<TVectorHolder<ui64>>>;


        template <EFeatureType FeatureType>
        class TFeaturesStorage {
        private:
            static_assert(FeatureType == EFeatureType::Float || FeatureType == EFeatureType::Categorical,
                "Only float and categorical features are currently supported");

            /******************************************************************************************/
            // non-binary features

            /* shared between this builder and created data provider for efficiency
             * (can be reused for cache here if there's no more references to data
             *  (data provider was freed))
             */
            TVector<TIntrusivePtr<TVectorHolder<ui64>>> DenseDataStorage; // [perTypeFeatureIdx]

            // view into storage for faster access
            TVector<TArrayRef<ui64>> DenseDstView; // [perTypeFeatureIdx]

            TVector<TIndexHelper<ui64>> IndexHelpers; // [perTypeFeatureIdx]

            /******************************************************************************************/
            // binary features

            // view into BinaryStorage for faster access
            // BinaryStorage is not owned by TFeaturesStorage but common for all FeatureTypes
            TVector<TArrayRef<TBinaryFeaturesPack>> DstBinaryView; // [packIdx][objectIdx][bitIdx]

            TVector<TMaybe<TPackedBinaryIndex>> FeatureIdxToPackedBinaryIndex; // [perTypeFeatureIdx]


            /******************************************************************************************/

            // copy from Data.MetaInfo.FeaturesLayout for fast access
            TVector<bool> IsAvailable; // [perTypeFeatureIdx]

        public:
            void PrepareForInitialization(
                const TFeaturesLayout& featuresLayout,
                ui32 objectCount,
                const TQuantizedFeaturesInfoPtr& quantizedFeaturesInfoPtr,
                TBinaryFeaturesStorage& binaryStorage,
                TConstArrayRef<TMaybe<TPackedBinaryIndex>> flatFeatureIndexToPackedBinaryIndex
            ) {
                const size_t perTypeFeatureCount = (size_t)featuresLayout.GetFeatureCount(FeatureType);
                DenseDataStorage.resize(perTypeFeatureCount);
                DenseDstView.resize(perTypeFeatureCount);
                IndexHelpers.resize(perTypeFeatureCount, TIndexHelper<ui64>(8));
                FeatureIdxToPackedBinaryIndex.resize(perTypeFeatureCount);

                IsAvailable = MakeIsAvailable<FeatureType>(featuresLayout);

                const auto metaInfos = featuresLayout.GetExternalFeaturesMetaInfo();

                const bool isFloatType = (FeatureType == EFeatureType::Float);
                for (auto perTypeFeatureIdx : xrange(perTypeFeatureCount)) {
                    const auto flatFeatureIdx = featuresLayout.GetExternalFeatureIdx(perTypeFeatureIdx, FeatureType);
                    if (!metaInfos[flatFeatureIdx].IsAvailable) {
                        continue;
                    }

                    ui8 bitsPerFeature;
                    if (isFloatType) {
                        bitsPerFeature = CalcHistogramWidthForBorders(
                            quantizedFeaturesInfoPtr->GetBorders(TFloatFeatureIdx(perTypeFeatureIdx)).size());
                    } else {
                        const ui32 countUnique =
                            quantizedFeaturesInfoPtr->GetUniqueValuesCounts(TCatFeatureIdx(perTypeFeatureIdx)).OnAll;
                        if (countUnique <= 1ULL << 8) {
                            bitsPerFeature = 8;
                        } else if (countUnique <= 1ULL << 16) {
                            bitsPerFeature = 16;
                        } else { //TODO
                            bitsPerFeature = 32;
                        }
                    }

                    IndexHelpers[perTypeFeatureIdx] = TIndexHelper<ui64>(bitsPerFeature);
                    FeatureIdxToPackedBinaryIndex[perTypeFeatureIdx]
                        = flatFeatureIndexToPackedBinaryIndex[flatFeatureIdx];
                }

                for (auto perTypeFeatureIdx : xrange(perTypeFeatureCount)) {
                    if (featuresLayout.GetInternalFeatureMetaInfo(
                            perTypeFeatureIdx,
                            FeatureType
                        ).IsAvailable &&
                        !FeatureIdxToPackedBinaryIndex[perTypeFeatureIdx])
                    {
                        CB_ENSURE(
                            IsAvailable[perTypeFeatureIdx],
                            FeatureType << " feature #" << perTypeFeatureIdx
                            << " has no data in quantized pool"
                        );

                        auto& maybeSharedStoragePtr = DenseDataStorage[perTypeFeatureIdx];
                        if (!maybeSharedStoragePtr || (maybeSharedStoragePtr->RefCount() > 1)) {
                            /* storage is either uninited or shared with some other references
                             * so it has to be reset to be reused
                             */
                            DenseDataStorage[perTypeFeatureIdx] = MakeIntrusive<TVectorHolder<ui64>>();
                        }
                        maybeSharedStoragePtr->Data.yresize(
                            IndexHelpers[perTypeFeatureIdx].CompressedSize(objectCount)
                        );
                        DenseDstView[perTypeFeatureIdx] =  maybeSharedStoragePtr->Data;
                    } else {
                        DenseDataStorage[perTypeFeatureIdx] = nullptr;
                        DenseDstView[perTypeFeatureIdx] = TArrayRef<ui64>();
                    }
                }

                for (auto& binaryStorageElement : binaryStorage) {
                    DstBinaryView.push_back(
                        TArrayRef<ui8>((ui8*)binaryStorageElement->Data.data(), objectCount)
                    );
                }
            }

            void Set(
                TFeatureIdx<FeatureType> perTypeFeatureIdx,
                ui32 objectOffset,
                ui8 bitsPerDocumentFeature,
                TConstArrayRef<ui8> featuresPart,
                NPar::TLocalExecutor* localExecutor
            ) {
                if (!IsAvailable[*perTypeFeatureIdx]) {
                    return;
                }

                if (FeatureIdxToPackedBinaryIndex[*perTypeFeatureIdx]) {
                    auto packedBinaryIndex = *FeatureIdxToPackedBinaryIndex[*perTypeFeatureIdx];
                    auto dstSlice = DstBinaryView[packedBinaryIndex.PackIdx].Slice(
                        objectOffset,
                        featuresPart.size()
                    );
                    ParallelSetBinaryFeatureInPackArray(
                        featuresPart,
                        packedBinaryIndex.BitIdx,
                        /*needToClearDstBits*/ false,
                        localExecutor,
                        &dstSlice
                    );
                } else {
                    CB_ENSURE_INTERNAL(
                        bitsPerDocumentFeature == 8 || bitsPerDocumentFeature == 16 || bitsPerDocumentFeature == 32,
                        "Only 8, 16 or 32 bits per document supported, got: " << bitsPerDocumentFeature);
                    CB_ENSURE_INTERNAL(IndexHelpers[*perTypeFeatureIdx].GetBitsPerKey() == bitsPerDocumentFeature,
                        "BitsPerKey should be equal to bitsPerDocumentFeature");

                    const auto bytesPerDocument = bitsPerDocumentFeature / (sizeof(ui8) * CHAR_BIT);

                    const auto dstCapacityInBytes =
                        DenseDstView[*perTypeFeatureIdx].size() *
                        sizeof(decltype(*DenseDstView[*perTypeFeatureIdx].data()));
                    const auto objectOffsetInBytes = objectOffset * bytesPerDocument;

                    CB_ENSURE_INTERNAL(
                        objectOffsetInBytes < dstCapacityInBytes,
                        LabeledOutput(perTypeFeatureIdx, objectOffset, objectOffsetInBytes, featuresPart.size(), dstCapacityInBytes));
                    CB_ENSURE_INTERNAL(
                        objectOffsetInBytes + featuresPart.size() <= dstCapacityInBytes,
                        LabeledOutput(perTypeFeatureIdx, objectOffset, objectOffsetInBytes, featuresPart.size(), dstCapacityInBytes));


                    memcpy(
                        ((ui8*)DenseDstView[*perTypeFeatureIdx].data()) + objectOffset,
                        featuresPart.data(),
                        featuresPart.size());
                }
            }

            template <class T, EFeatureValuesType FeatureValuesType>
            void GetResult(
                ui32 objectCount,
                const TFeaturesLayout& featuresLayout,
                const TFeaturesArraySubsetIndexing* subsetIndexing,
                const TVector<THolder<TBinaryPacksHolder>>& binaryFeaturesData,
                TVector<THolder<TTypedFeatureValuesHolder<T, FeatureValuesType>>>* result
            ) {
                CB_ENSURE_INTERNAL(
                    DenseDataStorage.size() == DenseDstView.size(),
                    "DenseDataStorage is inconsistent with DenseDstView; "
                    LabeledOutput(DenseDataStorage.size(), DenseDstView.size()));

                const size_t featureCount = (size_t)featuresLayout.GetFeatureCount(FeatureType);

                CB_ENSURE_INTERNAL(
                    DenseDataStorage.size() == featureCount,
                    "DenseDataStorage is inconsistent with feature Layout; "
                    LabeledOutput(DenseDataStorage.size(), featureCount));

                result->clear();
                result->reserve(featureCount);
                for (auto perTypeFeatureIdx : xrange(featureCount)) {
                    if (IsAvailable[perTypeFeatureIdx]) {
                        auto featureId = featuresLayout.GetExternalFeatureIdx(perTypeFeatureIdx, FeatureType);

                        if (FeatureIdxToPackedBinaryIndex[perTypeFeatureIdx]) {
                            auto packedBinaryIndex = *FeatureIdxToPackedBinaryIndex[perTypeFeatureIdx];

                            result->push_back(
                                MakeHolder<TPackedBinaryValuesHolderImpl<T, FeatureValuesType>>(
                                    featureId,
                                    binaryFeaturesData[packedBinaryIndex.PackIdx].Get(),
                                    packedBinaryIndex.BitIdx
                                )
                            );
                        } else {
                            result->push_back(
                                MakeHolder<TCompressedValuesHolderImpl<T, FeatureValuesType>>(
                                    featureId,
                                    TCompressedArray(
                                        objectCount,
                                        IndexHelpers[perTypeFeatureIdx].GetBitsPerKey(),
                                        TMaybeOwningArrayHolder<ui64>::CreateOwning(
                                            DenseDstView[perTypeFeatureIdx],
                                            DenseDataStorage[perTypeFeatureIdx]
                                        )
                                    ),
                                    subsetIndexing
                                )
                            );
                        }
                    } else {
                        result->push_back(nullptr);
                    }
                }
            }
        };

    private:
        ui32 ObjectCount;

        /* ForCPU because TQuantizedForCPUObjectsData is more generic than TQuantizedObjectsData -
         * it contains it as a subset
         */
        TQuantizedForCPUBuilderData Data;

        TVector<TString> StringClassLabels; // if poolQuantizationSchema.ClassLabels has Strings
        TVector<float> FloatClassLabels;    // if poolQuantizationSchema.ClassLabels has Integer or Float

        // will be moved to Data.RawTargetData at GetResult
        TVector<TVector<TString>> StringTarget;
        TVector<TVector<float>> FloatTarget;

        // will be moved to Data.RawTargetData at GetResult
        TVector<float> WeightsBuffer;
        TVector<float> GroupWeightsBuffer;

        TFeaturesStorage<EFeatureType::Float> FloatFeaturesStorage;
        TFeaturesStorage<EFeatureType::Categorical> CategoricalFeaturesStorage;
        TBinaryFeaturesStorage BinaryFeaturesStorage;

        TDataProviderBuilderOptions Options;
        TDatasetSubset DatasetSubset;

        NPar::TLocalExecutor* LocalExecutor;

        bool InProcess;
        bool ResultTaken;
    };


    class TLazyQuantizedFeaturesDataProviderBuilder : public TQuantizedFeaturesDataProviderBuilder
    {
    public:
        TLazyQuantizedFeaturesDataProviderBuilder(
            const TDataProviderBuilderOptions& options,
            NPar::TLocalExecutor* localExecutor
        )
            : TQuantizedFeaturesDataProviderBuilder(options, TDatasetSubset::MakeColumns(/*hasFeatures*/false), localExecutor)
            , Options(options)
            , PoolLoader(GetProcessor<IQuantizedPoolLoader, const TPathWithScheme&>(options.PoolPath, options.PoolPath))
            , LocalExecutor(localExecutor)
        {
            CB_ENSURE(PoolLoader, "Cannot load dataset because scheme " << options.PoolPath.Scheme << " is unsupported");
            CB_ENSURE(Options.GpuDistributedFormat, "Lazy columns are supported only in distributed GPU mode");
        }

        TDataProviderPtr GetResult() override {
            GetTargetAndBinaryFeaturesData();

            auto& dataRef = GetDataRef();
            const auto& subsetIndexing = *dataRef.CommonObjectsData.SubsetIndexing;
            const auto& featuresLayout = *dataRef.MetaInfo.FeaturesLayout;

            CB_ENSURE(featuresLayout.GetFeatureCount(EFeatureType::Categorical) == 0, "Categorical Lazy columns are not supported");
            dataRef.ObjectsData.Data.CatFeatures.clear();

            const size_t featureCount = (size_t)featuresLayout.GetFeatureCount(EFeatureType::Float);

            const auto& flatFeatureIdxToPackedBinaryIdx =
                dataRef.ObjectsData.PackedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex;

            const auto& isAvailable = MakeIsAvailable<EFeatureType::Float>(featuresLayout);

            TVector<THolder<IQuantizedFloatValuesHolder>>& lazyQuantizedColumns =
                dataRef.ObjectsData.Data.FloatFeatures;
            lazyQuantizedColumns.clear();
            lazyQuantizedColumns.reserve(featureCount);
            for (auto perTypeFeatureIdx : xrange(featureCount)) {
                if (isAvailable[perTypeFeatureIdx]) {
                    auto flatFeatureIdx = featuresLayout.GetExternalFeatureIdx(perTypeFeatureIdx, EFeatureType::Float);
                    CB_ENSURE(!flatFeatureIdxToPackedBinaryIdx[flatFeatureIdx], "Packed lazy columns are not supported");
                    lazyQuantizedColumns.push_back(
                        MakeHolder<TLazyCompressedValuesHolderImpl<ui8, EFeatureValuesType::QuantizedFloat>>(
                            flatFeatureIdx,
                            &subsetIndexing,
                            PoolLoader)
                    );
                } else {
                    lazyQuantizedColumns.push_back(nullptr);
                }
            }

            SetResultsTaken();

            return MakeDataProvider<TQuantizedObjectsDataProvider>(
                /*objectsGrouping*/ Nothing(), // will init from data
                CastToBase(std::move(dataRef)),
                Options.SkipCheck,
                LocalExecutor
            )->CastMoveTo<TObjectsDataProvider>();
        }

    private:
        TDataProviderBuilderOptions Options;
        TAtomicSharedPtr<IQuantizedPoolLoader> PoolLoader;
        NPar::TLocalExecutor* const LocalExecutor;
    };


    THolder<IDataProviderBuilder> CreateDataProviderBuilder(
        EDatasetVisitorType visitorType,
        const TDataProviderBuilderOptions& options,
        TDatasetSubset loadSubset,
        NPar::TLocalExecutor* localExecutor
    ) {
        switch (visitorType) {
            case EDatasetVisitorType::RawObjectsOrder:
                return MakeHolder<TRawObjectsOrderDataProviderBuilder>(options, localExecutor);
            case EDatasetVisitorType::RawFeaturesOrder:
                return MakeHolder<TRawFeaturesOrderDataProviderBuilder>(options, localExecutor);
            case EDatasetVisitorType::QuantizedFeatures:
                if (options.GpuDistributedFormat) {
                    return MakeHolder<TLazyQuantizedFeaturesDataProviderBuilder>(options, localExecutor);
                } else {
                    return MakeHolder<TQuantizedFeaturesDataProviderBuilder>(options, loadSubset, localExecutor);
                }
            default:
                return nullptr;
        }
    }
}
