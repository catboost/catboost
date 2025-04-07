#include "data_provider_builders.h"

#include "cat_feature_perfect_hash.h"
#include "columns.h"
#include "data_provider.h"
#include "feature_index.h"
#include "graph.h"
#include "lazy_columns.h"
#include "sparse_columns.h"
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

#include <library/cpp/threading/local_executor/local_executor.h>

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

    class TRawObjectsOrderDataProviderBuilder final : public IDataProviderBuilder,
                                                      public IRawObjectsOrderDataVisitor
    {
    public:
        TRawObjectsOrderDataProviderBuilder(
            const TDataProviderBuilderOptions& options,
            NPar::ILocalExecutor* localExecutor
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
            Data.CommonObjectsData.SetBuildersArrayRef(
                metaInfo,
                &NumGroupIdsRef,
                &StringGroupIdsRef,
                &NumSubgroupIdsRef,
                &StringSubgroupIdsRef
            );

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
            prepareFeaturesStorage(EmbeddingFeaturesStorage);

            switch (Data.MetaInfo.TargetType) {
                case ERawTargetType::Boolean:
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
            Y_ASSERT(!Data.CommonObjectsData.StoreStringColumns);
            NumGroupIdsRef[Cursor + localObjectIdx] = value;
        }

        void AddGroupId(ui32 localObjectIdx, const TString& value) override {
            Y_ASSERT(Data.CommonObjectsData.StoreStringColumns);
            StringGroupIdsRef[Cursor + localObjectIdx] = value;
        }

        void AddSubgroupId(ui32 localObjectIdx, TSubgroupId value) override {
            Y_ASSERT(!Data.CommonObjectsData.StoreStringColumns);
            NumSubgroupIdsRef[Cursor + localObjectIdx] = value;
        }

        void AddSubgroupId(ui32 localObjectIdx, const TString& value) override {
            Y_ASSERT(Data.CommonObjectsData.StoreStringColumns);
            StringSubgroupIdsRef[Cursor + localObjectIdx] = value;
        }

        void AddSampleId(ui32 localObjectIdx, const TString& value) override {
            Y_ASSERT(Data.CommonObjectsData.StoreStringColumns);
            (*Data.CommonObjectsData.SampleId)[Cursor + localObjectIdx] = value;
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

        void SetGraph(TRawPairsData&& graph) override {
            const auto matrix = ConvertGraphToAdjMatrix(graph, ObjectCount);
            FloatFeaturesStorage.SetAggregatedFeatures(matrix);
            Data.TargetData.Graph = std::move(graph);
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
            ui32 sparseFeatureCount = 0;
            features.ForEachNonDefault(
                [&] (ui32 perTypeFeatureIdx, float /*value*/) {
                    sparseFeatureCount += FloatFeaturesStorage.IsSparse(TFloatFeatureIdx(perTypeFeatureIdx));
                }
            );
            auto objectIdx = Cursor + localObjectIdx;
            if (features.GetNonDefaultSize() == sparseFeatureCount) {
                features.ForBlockNonDefault(
                    [&] (auto indexBlock, auto valueBlock) {
                        FloatFeaturesStorage.SetSparseFeatureBlock(objectIdx, indexBlock, valueBlock, &FloatFeaturesStorage);
                    },
                    /*blockSize*/ 1024
                );
                return;
            }
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
            CheckThreadId(hashPartIdx);
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

        void AddEmbeddingFeature(
            ui32 localObjectIdx,
            ui32 flatFeatureIdx,
            TMaybeOwningConstArrayHolder<float> feature
        ) override {
            auto embeddingFeatureIdx = GetInternalFeatureIdx<EFeatureType::Embedding>(flatFeatureIdx);
            EmbeddingFeaturesStorage.Set(
                embeddingFeatureIdx,
                Cursor + localObjectIdx,
                std::move(feature)
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
                (Data.MetaInfo.TargetType == ERawTargetType::Boolean) ||
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

        void SetPairs(TRawPairsData&& pairs) override {
            Data.TargetData.Pairs = std::move(pairs);
        }

        void SetTimestamps(TVector<ui64>&& timestamps) override {
            CheckDataSize(timestamps.size(), (size_t)ObjectCount, "timestamps");
            Data.CommonObjectsData.Timestamp = std::move(timestamps);
        }

        // needed for checking groupWeights consistency while loading from separate file
        TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const override {
            return Data.CommonObjectsData.GroupIds.GetMaybeNumData();
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
                        TStringBuf("Weights"),
                        /*allWeightsCanBeZero*/ true
                    );
                }
                if (Data.MetaInfo.HasGroupWeight) {
                    Data.TargetData.GroupWeights = TWeights<float>(
                        TVector<float>(GroupWeightsBuffer),
                        TStringBuf("GroupWeights"),
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
                        TStringBuf("Weights"),
                        /*allWeightsCanBeZero*/ InBlock
                    );
                }
                if (Data.MetaInfo.HasGroupWeight) {
                    Data.TargetData.GroupWeights = TWeights<float>(
                        std::move(GroupWeightsBuffer),
                        TStringBuf("GroupWeights"),
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
            getFeaturesResult(EmbeddingFeaturesStorage, &Data.ObjectsData.EmbeddingFeatures);

            ResultTaken = true;

            if (InBlock && Data.MetaInfo.HasGroupId) {
                auto fullData = MakeDataProvider<TRawObjectsDataProvider>(
                    /*objectsGrouping*/ Nothing(), // will init from data
                    std::move(Data),
                    Options.SkipCheck,
                    Data.MetaInfo.ForceUnitAutoPairWeights,
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
                    Data.MetaInfo.ForceUnitAutoPairWeights,
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
                Data.MetaInfo.ForceUnitAutoPairWeights,
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
        template <class T>
        inline void RollbackNextCursorToLastGroupStartImpl(const TVector<T>& groupIds) {
            auto rit = groupIds.rbegin();
            const T& lastGroupId = *rit;
            for (++rit; rit != groupIds.rend(); ++rit) {
                if (*rit != lastGroupId) {
                    break;
                }
            }
            // always rollback to the second last group
            NextCursor = ObjectCount - (rit - groupIds.rbegin());
        }

        void RollbackNextCursorToLastGroupStart() {
            if (ObjectCount == 0) {
                return;
            }
            if (Data.CommonObjectsData.StoreStringColumns) {
                RollbackNextCursorToLastGroupStartImpl(*Data.CommonObjectsData.GroupIds.GetMaybeStringData());
            } else {
                RollbackNextCursorToLastGroupStartImpl(*Data.CommonObjectsData.GroupIds.GetMaybeNumData());
            }
        }

        template <EFeatureType FeatureType>
        TFeatureIdx<FeatureType> GetInternalFeatureIdx(ui32 flatFeatureIdx) const {
            return Data.MetaInfo.FeaturesLayout->GetExpandingInternalFeatureIdx<FeatureType>(flatFeatureIdx);
        }

        static void CheckThreadId(int threadId) {
            CB_ENSURE(
                threadId < CB_THREAD_LIMIT,
                "thread ID exceeds an internal thread limit (" << CB_THREAD_LIMIT << "),"
                "try decreasing the specified number of threads"
            );
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

            NPar::ILocalExecutor* LocalExecutor = nullptr;

            TVector<TPerFeatureData> PerFeatureData; // [perTypeFeatureIdx]

            TVector<TPerFeatureData> GraphAggregatedFeaturesData; // [perTypeFeatureIdx * kFloatAggregationFeaturesCount]

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
                storage->PerFeatureData[*perTypeFeatureIdx].DenseDstView[objectIdx] = std::move(value);
            }

            static void SetAggregatedFeature(
                TFeatureIdx<FeatureType> perTypeFeatureIdx,
                ui32 objectIdx,
                T value,
                TFeaturesStorage* storage
            ) {
                storage->GraphAggregatedFeaturesData[*perTypeFeatureIdx].DenseDstView[objectIdx] = std::move(value);
            }

            static void SetSparseFeature(
                TFeatureIdx<FeatureType> perTypeFeatureIdx,
                ui32 objectIdx,
                T value,
                TFeaturesStorage* storage
            ) {
                const auto featureIdx = *perTypeFeatureIdx;
                SetSparseFeatureBlock(objectIdx, MakeArrayRef(&featureIdx, 1), MakeArrayRef(&value, 1), storage);
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
                                    if (i != dstIdx) {
                                        sparseDataPart.Values[dstIdx] = std::move(sparseDataPart.Values[i]);
                                    }
                                    ++dstIdx;
                                }
                            }
                        }
                        sparseDataPart.Indices.resize(dstIdx);
                        sparseDataPart.Values.resize(dstIdx);
                    },
                    0,
                    SafeIntegerCast<int>(SparseDataParts.size()),
                    NPar::TLocalExecutor::WAIT_COMPLETE
                );
            }


            TVector<TMaybe<TConstPolymorphicValuesSparseArray<T, ui32>>> CreateSparseArrays(
                ui32 objectCount,
                ESparseArrayIndexingType sparseArrayIndexingType,
                NPar::ILocalExecutor* localExecutor
            ) {
                TVector<size_t> sizesForBuilders(PerFeatureData.size());
                auto builderCount = PerFeatureData.size();
                for (auto& sparseDataPart : SparseDataParts) {
                    for (auto index2d : sparseDataPart.Indices) {
                        if (index2d.PerTypeFeatureIdx >= builderCount) {
                            // add previously unknown features
                            builderCount = index2d.PerTypeFeatureIdx + 1;
                            sizesForBuilders.resize(builderCount);
                        }
                        sizesForBuilders[index2d.PerTypeFeatureIdx] += 1;
                    }
                }
                if (builderCount == 0) {
                    return TVector<TMaybe<TConstPolymorphicValuesSparseArray<T, ui32>>>{0};
                }
                TVector<TSparseDataForBuider> sparseDataForBuilders(builderCount); // [perTypeFeatureIdx]

                for (auto idx : xrange(sparseDataForBuilders.size())) {
                    sparseDataForBuilders[idx].ObjectIndices.yresize(sizesForBuilders[idx]);
                    sparseDataForBuilders[idx].Values.yresize(sizesForBuilders[idx]);
                }

                const auto valueCount = Accumulate(sizesForBuilders, ui64(0));
                const auto valueCountPerRange = CeilDiv<ui64>(valueCount, localExecutor->GetThreadCount() + 1);
                TVector<NCB::TIndexRange<ui32>> builderRanges;
                ui32 rangeSize = 0;
                ui32 rangeOffset = 0;
                for (ui32 featureIdx : xrange(sizesForBuilders.size())) {
                    if (rangeSize >= valueCountPerRange) {
                        builderRanges.push_back({rangeOffset, featureIdx});
                        rangeOffset = featureIdx;
                        rangeSize = 0;
                    }
                    rangeSize += sizesForBuilders[featureIdx];
                }
                if (rangeSize > 0) {
                    builderRanges.push_back({rangeOffset, ui32(sizesForBuilders.size())});
                }

                TVector<size_t> idxForBuilders(sparseDataForBuilders.size());
                const auto rangeCount = builderRanges.size();
                NPar::ParallelFor(
                    *localExecutor,
                    0,
                    rangeCount,
                    [&] (ui32 rangeIdx) {
                        for (auto& sparseDataPart : SparseDataParts) {
                            if (sparseDataPart.Indices.empty()) {
                                continue;
                            }
                            const auto indicesRef = MakeArrayRef(sparseDataPart.Indices);
                            const auto valuesRef = MakeArrayRef(sparseDataPart.Values);
                            const auto idxRef = MakeArrayRef(idxForBuilders);
                            const auto buildersRef = MakeArrayRef(sparseDataForBuilders);
                            const auto rangeBegin = builderRanges[rangeIdx].Begin;
                            const auto rangeEnd = builderRanges[rangeIdx].End;
                            for (auto i : xrange(indicesRef.size())) {
                                const auto index2d = indicesRef[i];
                                const auto featureIdx = index2d.PerTypeFeatureIdx;
                                if (featureIdx >= rangeBegin && featureIdx < rangeEnd) {
                                    auto& dataForBuilder = buildersRef[featureIdx];
                                    dataForBuilder.ObjectIndices[idxRef[featureIdx]] = index2d.ObjectIdx;
                                    dataForBuilder.Values[idxRef[featureIdx]] = valuesRef[i];
                                    ++idxRef[featureIdx];
                                }
                            }
                        }
                    }
                );
                if (!DataCanBeReusedForNextBlock) {
                    for (auto& sparseDataPart : SparseDataParts) {
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
                NPar::ILocalExecutor* localExecutor
            ) {
                ui32 prevObjectCount = ObjectCount;
                ObjectCount = objectCount;

                DataCanBeReusedForNextBlock = dataCanBeReusedForNextBlock;

                LocalExecutor = localExecutor;

                HasSparseData = haveUnknownNumberOfSparseFeatures;


                const size_t featureCount = (size_t) featuresLayout.GetFeatureCount(FeatureType);
                PerFeatureData.resize(featureCount);
                PerFeatureCallbacks.resize(featureCount + 1);

                const size_t aggregatedFeatureCount = (size_t) featuresLayout.GetAggregatedFeatureCount(FeatureType);
                GraphAggregatedFeaturesData.resize(aggregatedFeatureCount);

                for (auto perTypeFeatureIdx : xrange(aggregatedFeatureCount)) {
                    if (PerFeatureData[perTypeFeatureIdx / kFloatAggregationFeaturesCount].MetaInfo.IsAvailable) {
                        auto &maybeSharedStoragePtr = GraphAggregatedFeaturesData[perTypeFeatureIdx].DenseDataStorage;
                        CB_ENSURE_INTERNAL(!prevTailSize, "No dense data storage to store remainder of previous block");
                        maybeSharedStoragePtr = MakeIntrusive<TVectorHolder<T>>();
                        maybeSharedStoragePtr->Data.yresize(objectCount);

                        GraphAggregatedFeaturesData[perTypeFeatureIdx].DenseDstView = maybeSharedStoragePtr->Data;
                     }
                }

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
                                CB_ENSURE(!prevTailSize, "No dense data storage to store remainder of previous block");
                                maybeSharedStoragePtr = MakeIntrusive<TVectorHolder<T>>();
                                maybeSharedStoragePtr->Data.yresize(objectCount);
                            } else {
                                CB_ENSURE(
                                    prevTailSize <= maybeSharedStoragePtr->Data.size(),
                                    "Dense data storage is too small to to store remainder of previous block");
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

            inline void Set(TFeatureIdx<FeatureType> perTypeFeatureIdx, ui32 objectIdx, T value) {
                PerFeatureCallbacks[Min(size_t(*perTypeFeatureIdx), PerFeatureCallbacks.size() - 1)](
                    perTypeFeatureIdx,
                    objectIdx,
                    value,
                    this
                );
            }

            inline void AddAggFeaturesToLayout(TFeaturesLayout* featuresLayout) {
                if (!featuresLayout->HasGraphForAggregatedFeatures()) {
                    return;
                }
                const auto aggregationTypes = GetAggregationTypeNames(FeatureType);
                if (aggregationTypes.empty()) {
                    return;
                }
                for (auto perTypeFeatureIdx : xrange(PerFeatureData.size())) {
                    const auto& metaInfo = PerFeatureData[perTypeFeatureIdx].MetaInfo;
                    for (auto aggregationType : aggregationTypes) {
                        TStringStream name;
                        if (metaInfo.Name) {
                            name << "Aggregated for float " << metaInfo.Name;
                        } else {
                            name << "Aggregated for float " << perTypeFeatureIdx;
                        }
                        featuresLayout->AddFeature(
                            TFeatureMetaInfo(
                                FeatureType,
                                /*name*/ ToString(aggregationType) + name.Str(),
                                /*isSparse*/ false,
                                /*isIgnored*/ metaInfo.IsIgnored,
                                /*isAvailable*/ metaInfo.IsAvailable,
                                /*isAggregated*/ true));
                    }
                }
            }

            inline void SetFloatAggregatedFeatures(const TVector<TVector<ui32>>& matrix) {
                for (auto perTypeFeatureIdx : xrange(PerFeatureData.size())) {
                    if (!PerFeatureData[perTypeFeatureIdx].MetaInfo.IsAvailable) {
                        continue;
                    }
                    auto featureAccessor = [&](size_t index) {
                        return PerFeatureData[perTypeFeatureIdx].DenseDstView[index];
                    };

                    for (ui32 objectIdx = 0; objectIdx < ObjectCount; ++objectIdx) {
                        TFloatAggregation agg = CalcAggregationFeatures(
                            matrix,
                            featureAccessor,
                            objectIdx
                        );

                        auto idx = perTypeFeatureIdx * kFloatAggregationFeaturesCount;
                        SetAggregatedFeature(TFeatureIdx<FeatureType>(idx), objectIdx, agg.Mean, this);
                        SetAggregatedFeature(TFeatureIdx<FeatureType>(idx + 1), objectIdx, agg.Min, this);
                        SetAggregatedFeature(TFeatureIdx<FeatureType>(idx + 2), objectIdx, agg.Max, this);
                    }
                }
            }

            inline void SetAggregatedFeatures(const TVector<TVector<ui32>>& matrix) {
                CB_ENSURE(!HasSparseData, "Sparse data is not supported\n");
                switch (FeatureType) {
                    case EFeatureType::Float:
                        SetFloatAggregatedFeatures(matrix);
                    case EFeatureType::Categorical:
                        break;
                    case EFeatureType::Text:
                        break;
                    case EFeatureType::Embedding:
                        break;
                }
            }

            inline bool IsSparse(TFeatureIdx<FeatureType> perTypeFeatureIdx) const {
                const auto idx = *perTypeFeatureIdx;
                return idx + 1 < PerFeatureCallbacks.size() && PerFeatureCallbacks[idx] == SetSparseFeature;
            }

            Y_FORCE_INLINE static void SetSparseFeatureBlock(
                ui32 objectIdx,
                TConstArrayRef<ui32> indices,
                TConstArrayRef<T> values,
                TFeaturesStorage* storage
            ) {
                Y_POD_STATIC_THREAD(int) threadId(-1);
                if (Y_UNLIKELY(threadId == -1)) {
                    threadId = storage->LocalExecutor->GetWorkerThreadId();
                    TRawObjectsOrderDataProviderBuilder::CheckThreadId(threadId);
                }
                auto& sparseDataPart = storage->SparseDataParts[threadId];
                for (auto idx : indices) {
                    sparseDataPart.Indices.emplace_back(TSparseIndex2d{idx, objectIdx});
                }
                if (values.size() == 1) {
                    sparseDataPart.Values.push_back(values[0]);
                } else {
                    sparseDataPart.Values.insert(sparseDataPart.Values.end(), values.begin(), values.end());
                }
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

            template <class TColumn>
            void GetResult(
                const TFeaturesArraySubsetIndexing* subsetIndexing,
                ESparseArrayIndexingType sparseArrayIndexingType,
                TFeaturesLayout* featuresLayout,
                TVector<THolder<TColumn>>* result
            ) {
                size_t featureCount = (size_t)featuresLayout->GetFeatureCount(FeatureType);

                CB_ENSURE_INTERNAL(
                    PerFeatureData.size() >= featureCount,
                    "PerFeatureData is inconsistent with feature Layout"
                );

                AddAggFeaturesToLayout(featuresLayout);
                TVector<TMaybe<TConstPolymorphicValuesSparseArray<T, ui32>>> sparseData = CreateSparseArrays(
                    subsetIndexing->Size(),
                    sparseArrayIndexingType,
                    LocalExecutor
                );
                featureCount = (size_t)featuresLayout->GetFeatureCount(FeatureType);

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
                                MakeHolder<TSparsePolymorphicArrayValuesHolder<TColumn>>(
                                    /* featureId */ flatFeatureIdx,
                                    std::move(*(sparseData[perTypeFeatureIdx]))
                                )
                            );
                        } else if (metaInfo.IsAggregated) {
                            const auto& perFeatureData = GraphAggregatedFeaturesData[perTypeFeatureIdx - PerFeatureData.size()];
                            result->push_back(
                                MakeHolder<TPolymorphicArrayValuesHolder<TColumn>>(
                                    /* featureId */ flatFeatureIdx,
                                    TMaybeOwningConstArrayHolder<T>::CreateOwning(
                                        perFeatureData.DenseDstView,
                                        perFeatureData.DenseDataStorage
                                    ),
                                    subsetIndexing
                                )
                            );
                        } else {
                            result->push_back(
                                MakeHolder<TPolymorphicArrayValuesHolder<TColumn>>(
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
        TFeaturesStorage<EFeatureType::Embedding, TConstEmbedding> EmbeddingFeaturesStorage;

        TArrayRef<TGroupId> NumGroupIdsRef;
        TArrayRef<TString> StringGroupIdsRef;
        TArrayRef<TSubgroupId> NumSubgroupIdsRef;
        TArrayRef<TString> StringSubgroupIdsRef;

        std::array<THashPart, CB_THREAD_LIMIT> HashMapParts;


        static constexpr const ui32 NotSet = Max<ui32>();
        ui32 Cursor;
        ui32 NextCursor;

        TDataProviderBuilderOptions Options;

        NPar::ILocalExecutor* LocalExecutor;

        bool InProcess;
        bool ResultTaken;
    };


    class TRawFeaturesOrderDataProviderBuilder final : public IDataProviderBuilder,
                                                       public IRawFeaturesOrderDataVisitor
    {
    public:
        static constexpr int OBJECT_CALC_BLOCK_SIZE = 10000;

        TRawFeaturesOrderDataProviderBuilder(
            const TDataProviderBuilderOptions& options,
            NPar::ILocalExecutor* localExecutor
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
                new NPar::ILocalExecutor::TExecRangeParams(0, SafeIntegerCast<int>(ObjectCount))
            );
            ObjectCalcParams->SetBlockSize(OBJECT_CALC_BLOCK_SIZE);

            Data.MetaInfo = metaInfo;
            Data.TargetData.PrepareForInitialization(metaInfo, ObjectCount, 0);
            Data.CommonObjectsData.PrepareForInitialization(metaInfo, ObjectCount, 0);
            Data.ObjectsData.PrepareForInitialization(metaInfo);
            Data.CommonObjectsData.SetBuildersArrayRef(
                metaInfo,
                &NumGroupIdsRef,
                &StringGroupIdsRef,
                &NumSubgroupIdsRef,
                &StringSubgroupIdsRef
            );

            Data.CommonObjectsData.ResourceHolders = std::move(resourceHolders);
            Data.CommonObjectsData.Order = objectsOrder;

            Data.CommonObjectsData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(ObjectCount)
            );
        }

        // TCommonObjectsData
        void AddGroupId(ui32 objectIdx, TGroupId value) override {
            Y_ASSERT(!Data.CommonObjectsData.StoreStringColumns);
            NumGroupIdsRef[objectIdx] = value;
        }

        void AddSubgroupId(ui32 objectIdx, TSubgroupId value) override {
            Y_ASSERT(!Data.CommonObjectsData.StoreStringColumns);
            NumSubgroupIdsRef[objectIdx] = value;
        }

        void AddGroupId(ui32 objectIdx, const TString& value) override {
            Y_ASSERT(Data.CommonObjectsData.StoreStringColumns);
            StringGroupIdsRef[objectIdx] = value;
        }

        void AddSubgroupId(ui32 objectIdx, const TString& value) override {
            Y_ASSERT(Data.CommonObjectsData.StoreStringColumns);
            StringSubgroupIdsRef[objectIdx] = value;
        }

        void AddSampleId(ui32 objectIdx, const TString& value) override {
            (*Data.CommonObjectsData.SampleId)[objectIdx] = value;
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

        void SetGraph(TRawPairsData&& graph) override {
            Y_UNUSED(graph);
            CB_ENSURE_INTERNAL(false, "Unimplemented");
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

        void AddEmbeddingFeature(ui32 flatFeatureIdx, ITypedSequencePtr<TMaybeOwningConstArrayHolder<float>> features) override {
            auto embeddingFeatureIdx = GetInternalFeatureIdx<EFeatureType::Embedding>(flatFeatureIdx);
            Data.ObjectsData.EmbeddingFeatures[*embeddingFeatureIdx]
               = MakeHolder<TEmbeddingArrayValuesHolder>(
                    flatFeatureIdx,
                    features->GetSubset(Data.CommonObjectsData.SubsetIndexing.Get())
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

        void SetPairs(TRawPairsData&& pairs) override {
            Data.TargetData.Pairs = std::move(pairs);
        }

        void SetTimestamps(TVector<ui64>&& timestamps) override {
            CheckDataSize(timestamps.size(), (size_t)ObjectCount, "timestamps");
            Data.CommonObjectsData.Timestamp = timestamps;
        }

        // needed for checking groupWeights consistency while loading from separate file
        TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const override {
            return Data.CommonObjectsData.GroupIds.GetMaybeNumData();
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
                Data.MetaInfo.ForceUnitAutoPairWeights,
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

        TArrayRef<TGroupId> NumGroupIdsRef;
        TArrayRef<TString> StringGroupIdsRef;
        TArrayRef<TSubgroupId> NumSubgroupIdsRef;
        TArrayRef<TString> StringSubgroupIdsRef;

        TDataProviderBuilderOptions Options;

        NPar::ILocalExecutor* LocalExecutor;

        // have to make it THolder because NPar::ILocalExecutor::TExecRangeParams is unassignable/unmoveable
        THolder<NPar::ILocalExecutor::TExecRangeParams> ObjectCalcParams;

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
            NPar::ILocalExecutor* localExecutor
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

            const NCB::TPoolQuantizationSchema& poolQuantizationSchema,
            bool wholeColumns
        ) override {
            CB_ENSURE(!InProcess, "Attempt to start new processing without finishing the last");

            CB_ENSURE(poolQuantizationSchema.HasAvailableFeatures(), "No features in quantized pool!");

            TConstArrayRef<NJson::TJsonValue> schemaClassLabels = poolQuantizationSchema.ClassLabels;

            if (metaInfo.TargetType == ERawTargetType::String && !schemaClassLabels.empty()) {
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
                    case NJson::JSON_BOOLEAN:
                        CheckBooleanClassLabels(schemaClassLabels);
                        FloatClassLabels.push_back(0.0f);
                        FloatClassLabels.push_back(1.0f);
                        break;
                    case NJson::JSON_INTEGER:
                        for (const NJson::TJsonValue& classLabel : schemaClassLabels) {
                            FloatClassLabels.push_back(static_cast<float>(classLabel.GetInteger()));
                        }
                        break;
                    case NJson::JSON_DOUBLE:
                        for (const NJson::TJsonValue& classLabel : schemaClassLabels) {
                            FloatClassLabels.push_back(static_cast<float>(classLabel.GetDouble()));
                        }
                        break;
                    default:
                        CB_ENSURE_INTERNAL(
                            false,
                            "Unexpected JSON label type for non-string target type : "
                            << schemaClassLabels[0].GetType()
                        );
                }
                CB_ENSURE_INTERNAL(
                    metaInfo.TargetType == GetRawTargetType(schemaClassLabels[0].GetType()),
                    "metaInfo.TargetType (" << metaInfo.TargetType
                    << ") is inconsistent with schemaClassLabels type ("
                    << schemaClassLabels[0].GetType() << ')'
                );
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
            Data.ObjectsData.PrepareForInitialization(
                metaInfo,

                // TODO(akhropov): get from quantized pool meta info when it will be available: MLTOOLS-2392.
                NCatboostOptions::TBinarizationOptions(
                    EBorderSelectionType::GreedyLogSum, // default value
                    (poolQuantizationSchema.Borders.size() > 0) ?
                        SafeIntegerCast<ui32>(poolQuantizationSchema.Borders[0].size())
                        : 32,
                    ENanMode::Forbidden // default value
                ),
                TMap<ui32, NCatboostOptions::TBinarizationOptions>()
            );

            FillQuantizedFeaturesInfo(
                poolQuantizationSchema,
                Data.ObjectsData.QuantizedFeaturesInfo.Get()
            );

            Data.ObjectsData.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
                *metaInfo.FeaturesLayout,
                TVector<TExclusiveFeaturesBundle>() // TODO(akhropov): bundle quantized data
            );

            Data.ObjectsData.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
                *metaInfo.FeaturesLayout,
                *Data.ObjectsData.QuantizedFeaturesInfo,
                Data.ObjectsData.ExclusiveFeatureBundlesData,
                /*dontPack*/ Options.GpuDistributedFormat
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
                    Data.ObjectsData.QuantizedFeaturesInfo,
                    BinaryFeaturesStorage,
                    Data.ObjectsData.PackedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex,
                    wholeColumns
                );
                CategoricalFeaturesStorage.PrepareForInitialization(
                    *metaInfo.FeaturesLayout,
                    objectCount,
                    Data.ObjectsData.QuantizedFeaturesInfo,
                    BinaryFeaturesStorage,
                    Data.ObjectsData.PackedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex,
                    wholeColumns
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
                objectOffset <= dstData->size(),
                LabeledOutput(objectOffset, srcPart.GetSize(), dstData->size()));
            CB_ENSURE_INTERNAL(
                objectOffset + srcPart.GetSize() <= dstData->size(),
                LabeledOutput(objectOffset, srcPart.GetSize(), dstData->size()));

            TArrayRef<T> dstArrayRef(dstData->data() + objectOffset, srcPart.GetSize());
            srcPart.WriteTo(&dstArrayRef);
        }

        // TCommonObjectsData
        void AddGroupIdPart(ui32 objectOffset, TUnalignedArrayBuf<TGroupId> groupIdPart) override {
            CopyPart(objectOffset, groupIdPart, &(*Data.CommonObjectsData.GroupIds.GetMaybeNumData()));
        }

        void AddSubgroupIdPart(ui32 objectOffset, TUnalignedArrayBuf<TSubgroupId> subgroupIdPart) override {
            CopyPart(objectOffset, subgroupIdPart, &(*Data.CommonObjectsData.SubgroupIds.GetMaybeNumData()));
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
                std::move(featuresPart),
                LocalExecutor
            );
        }

        void SetGraph(TRawPairsData&& graph) override {
            Y_UNUSED(graph);
            CB_ENSURE_INTERNAL(false, "Unimplemented");

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
                std::move(featuresPart),
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
                    (Data.MetaInfo.TargetType == ERawTargetType::Boolean) ||
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

        void SetPairs(TRawPairsData&& pairs) override {
            Data.TargetData.Pairs = std::move(pairs);
        }

        void SetTimestamps(TVector<ui64>&& timestamps) override {
            CheckDataSize(timestamps.size(), (size_t)ObjectCount, "timestamps");
            Data.CommonObjectsData.Timestamp = std::move(timestamps);
        }

        // needed for checking groupWeights consistency while loading from separate file
        TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const override {
            return Data.CommonObjectsData.GroupIds.GetMaybeNumData();
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
                    &Data.ObjectsData.FloatFeatures
                );

                CategoricalFeaturesStorage.GetResult(
                    ObjectCount,
                    *Data.MetaInfo.FeaturesLayout,
                    Data.CommonObjectsData.SubsetIndexing.Get(),
                    Data.ObjectsData.PackedBinaryFeaturesData.SrcData,
                    &Data.ObjectsData.CatFeatures
                );
            }

            SetResultsTaken();

            return MakeDataProvider<TQuantizedObjectsDataProvider>(
                /*objectsGrouping*/ Nothing(), // will init from data
                std::move(Data),
                // without HasFeatures dataprovider self-test fails on distributed train
                // on quantized pool
                Options.SkipCheck || !DatasetSubset.HasFeatures,
                Data.MetaInfo.ForceUnitAutoPairWeights,
                LocalExecutor
            )->CastMoveTo<TObjectsDataProvider>();
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

        TQuantizedBuilderData& GetDataRef() {
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
            for (size_t i = 0, iEnd = schema.FloatFeatureIndices.size(); i < iEnd; ++i) {
                const auto flatFeatureIdx = schema.FloatFeatureIndices[i];
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

            bool WholeColumns = false;
            size_t ObjectCount = 0;

            /******************************************************************************************/
            // non-binary features

            /* shared between this builder and created data provider for efficiency
             * (can be reused for cache here if there's no more references to data
             *  (data provider was freed))
             *
             *   Used if WholeColumns == false
             */
            TVector<TIntrusivePtr<TVectorHolder<ui64>>> DenseDataStorage; // [perTypeFeatureIdx]

            // view into storage for faster access
            TVector<TArrayRef<ui64>> DenseDstView; // [perTypeFeatureIdx]

            // whole columns

            /*
             * Used if WholeColumns == true
             *  if src array data is properly aligned (as ui64, because it is used in TCompressedArray)
             *    just copy TMaybeOwningArrayHolder as, make a copy otherwise
             *  but const cast is used because TCompressedArray accepts only TMaybeOwningArrayHolder
             *  whereas in fact it is used only as read-only data later in data provider
             *  TODO(akhropov): Propagate const-correctness here
             */

            TVector<TMaybeOwningArrayHolder<ui64>> DenseWholeColumns; // [perTypeFeatureIdx]


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
                TConstArrayRef<TMaybe<TPackedBinaryIndex>> flatFeatureIndexToPackedBinaryIndex,
                bool wholeColumns
            ) {
                ObjectCount = objectCount;
                WholeColumns = wholeColumns;

                const size_t perTypeFeatureCount = (size_t)featuresLayout.GetFeatureCount(FeatureType);
                const size_t aggPerTypeFeatureCount = (size_t)featuresLayout.GetAggregatedFeatureCount(FeatureType);
                if (WholeColumns) {
                    DenseWholeColumns.resize(perTypeFeatureCount + aggPerTypeFeatureCount);
                    DenseDataStorage.clear();
                    DenseDstView.clear();
                } else {
                    DenseWholeColumns.clear();
                    DenseDataStorage.resize(perTypeFeatureCount + aggPerTypeFeatureCount);
                    DenseDstView.resize(perTypeFeatureCount + aggPerTypeFeatureCount);
                }
                IndexHelpers.resize(perTypeFeatureCount + aggPerTypeFeatureCount, TIndexHelper<ui64>(8));
                FeatureIdxToPackedBinaryIndex.resize(perTypeFeatureCount + aggPerTypeFeatureCount);

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
                        bitsPerFeature = CalcHistogramWidthForUniqueValuesCount(
                            quantizedFeaturesInfoPtr->GetUniqueValuesCounts(TCatFeatureIdx(perTypeFeatureIdx)).OnAll);
                    }

                    IndexHelpers[perTypeFeatureIdx] = TIndexHelper<ui64>(bitsPerFeature);
                    FeatureIdxToPackedBinaryIndex[perTypeFeatureIdx]
                        = flatFeatureIndexToPackedBinaryIndex[flatFeatureIdx];
                }

                if (!WholeColumns) {
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
                TMaybeOwningConstArrayHolder<ui8> featuresPart,
                NPar::ILocalExecutor* localExecutor
            ) {
                if (!IsAvailable[*perTypeFeatureIdx]) {
                    return;
                }

                if (FeatureIdxToPackedBinaryIndex[*perTypeFeatureIdx]) {
                    auto packedBinaryIndex = *FeatureIdxToPackedBinaryIndex[*perTypeFeatureIdx];
                    auto dstSlice = DstBinaryView[packedBinaryIndex.PackIdx].Slice(
                        objectOffset,
                        featuresPart.GetSize()
                    );
                    ParallelSetBinaryFeatureInPackArray(
                        *featuresPart,
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

                    if (WholeColumns) {
                        CB_ENSURE(objectOffset == 0, "objectOffset must be 0 for WholeColumns");
                        const ui8* dataPtr = featuresPart.data();

                        if (reinterpret_cast<ui64>(dataPtr) % alignof(ui64)) {
                            // fallback to data copying if it is unaligned
                            TVector<ui64> dataCopy;
                            dataCopy.yresize(IndexHelpers[*perTypeFeatureIdx].CompressedSize(ObjectCount));
                            Copy(featuresPart.begin(), featuresPart.end(), (ui8*)dataCopy.data());

                            DenseWholeColumns[*perTypeFeatureIdx] =
                                TMaybeOwningArrayHolder<ui64>::CreateOwning(std::move(dataCopy));
                        } else {
                            DenseWholeColumns[*perTypeFeatureIdx] =
                                TMaybeOwningArrayHolder<ui64>::CreateOwning(
                                    TArrayRef<ui64>(
                                        (ui64*)const_cast<ui8*>(dataPtr),
                                        CeilDiv(featuresPart.GetSize(), sizeof(ui64))
                                    ),
                                    featuresPart.GetResourceHolder()
                                );
                        }
                    } else {
                        const auto dstCapacityInBytes =
                            DenseDstView[*perTypeFeatureIdx].size() *
                            sizeof(decltype(*DenseDstView[*perTypeFeatureIdx].data()));
                        const auto objectOffsetInBytes = objectOffset * bytesPerDocument;

                        CB_ENSURE_INTERNAL(
                            objectOffsetInBytes < dstCapacityInBytes,
                            LabeledOutput(perTypeFeatureIdx, objectOffset, objectOffsetInBytes, featuresPart.GetSize(), dstCapacityInBytes));
                        CB_ENSURE_INTERNAL(
                            objectOffsetInBytes + featuresPart.GetSize() <= dstCapacityInBytes,
                            LabeledOutput(perTypeFeatureIdx, objectOffset, objectOffsetInBytes, featuresPart.GetSize(), dstCapacityInBytes));


                        memcpy(
                            ((ui8*)DenseDstView[*perTypeFeatureIdx].data()) + objectOffsetInBytes,
                            featuresPart.data(),
                            featuresPart.GetSize());
                    }
                }
            }

            template <class TColumn>
            void GetResult(
                ui32 objectCount,
                const TFeaturesLayout& featuresLayout,
                const TFeaturesArraySubsetIndexing* subsetIndexing,
                const TVector<THolder<IBinaryPacksArray>>& binaryFeaturesData,
                TVector<THolder<TColumn>>* result
            ) {


                const size_t featureCount = (size_t)featuresLayout.GetFeatureCount(FeatureType);

                if (WholeColumns) {
                    CB_ENSURE_INTERNAL(
                        DenseWholeColumns.size() == featureCount,
                        "DenseWholeColumns is inconsistent with feature Layout; "
                        LabeledOutput(DenseWholeColumns.size(), featureCount));
                } else {
                    CB_ENSURE_INTERNAL(
                        DenseDataStorage.size() == featureCount,
                        "DenseDataStorage is inconsistent with feature Layout; "
                        LabeledOutput(DenseDataStorage.size(), featureCount));
                    CB_ENSURE_INTERNAL(
                        DenseDataStorage.size() == DenseDstView.size(),
                        "DenseDataStorage is inconsistent with DenseDstView; "
                        LabeledOutput(DenseDataStorage.size(), DenseDstView.size()));
                }

                result->clear();
                result->reserve(featureCount);
                for (auto perTypeFeatureIdx : xrange(featureCount)) {
                    if (IsAvailable[perTypeFeatureIdx]) {
                        auto featureId = featuresLayout.GetExternalFeatureIdx(perTypeFeatureIdx, FeatureType);

                        if (FeatureIdxToPackedBinaryIndex[perTypeFeatureIdx]) {
                            auto packedBinaryIndex = *FeatureIdxToPackedBinaryIndex[perTypeFeatureIdx];

                            result->push_back(
                                MakeHolder<TPackedBinaryValuesHolderImpl<TColumn>>(
                                    featureId,
                                    binaryFeaturesData[packedBinaryIndex.PackIdx].Get(),
                                    packedBinaryIndex.BitIdx
                                )
                            );
                        } else {
                            result->push_back(
                                MakeHolder<TCompressedValuesHolderImpl<TColumn>>(
                                    featureId,
                                    TCompressedArray(
                                        objectCount,
                                        IndexHelpers[perTypeFeatureIdx].GetBitsPerKey(),
                                        WholeColumns ?
                                            std::move(DenseWholeColumns[perTypeFeatureIdx])
                                          : TMaybeOwningArrayHolder<ui64>::CreateOwning(
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

        TQuantizedBuilderData Data;

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

        NPar::ILocalExecutor* LocalExecutor;

        bool InProcess;
        bool ResultTaken;
    };


    class TLazyQuantizedFeaturesDataProviderBuilder final : public TQuantizedFeaturesDataProviderBuilder
    {
    public:
        TLazyQuantizedFeaturesDataProviderBuilder(
            const TDataProviderBuilderOptions& options,
            NPar::ILocalExecutor* localExecutor
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
            CB_ENSURE(subsetIndexing.IsFullSubset(), "Subset indexing is not supported for lazy columns");
            const auto& featuresLayout = *dataRef.MetaInfo.FeaturesLayout;

            CB_ENSURE(featuresLayout.GetFeatureCount(EFeatureType::Categorical) == 0, "Categorical lazy columns are not supported");
            dataRef.ObjectsData.CatFeatures.clear();

            const size_t featureCount = (size_t)featuresLayout.GetFeatureCount(EFeatureType::Float);

            const auto& flatFeatureIdxToPackedBinaryIdx =
                dataRef.ObjectsData.PackedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex;

            const auto& isAvailable = MakeIsAvailable<EFeatureType::Float>(featuresLayout);

            TVector<THolder<IQuantizedFloatValuesHolder>>& lazyQuantizedColumns =
                dataRef.ObjectsData.FloatFeatures;
            lazyQuantizedColumns.clear();
            lazyQuantizedColumns.reserve(featureCount);
            for (auto perTypeFeatureIdx : xrange(featureCount)) {
                if (isAvailable[perTypeFeatureIdx]) {
                    auto flatFeatureIdx = featuresLayout.GetExternalFeatureIdx(perTypeFeatureIdx, EFeatureType::Float);
                    CB_ENSURE(!flatFeatureIdxToPackedBinaryIdx[flatFeatureIdx], "Packed lazy columns are not supported");
                    lazyQuantizedColumns.push_back(
                        MakeHolder<TLazyCompressedValuesHolderImpl<IQuantizedFloatValuesHolder>>(
                            flatFeatureIdx,
                            PoolLoader->GetPoolPathWithScheme(),
                            subsetIndexing.Size())
                    );
                } else {
                    lazyQuantizedColumns.push_back(nullptr);
                }
            }

            SetResultsTaken();

            return MakeDataProvider<TQuantizedObjectsDataProvider>(
                /*objectsGrouping*/ Nothing(), // will init from data
                std::move(dataRef),
                Options.SkipCheck,
                dataRef.MetaInfo.ForceUnitAutoPairWeights,
                LocalExecutor
            )->CastMoveTo<TObjectsDataProvider>();
        }

    private:
        TDataProviderBuilderOptions Options;
        TAtomicSharedPtr<IQuantizedPoolLoader> PoolLoader;
        NPar::ILocalExecutor* const LocalExecutor;
    };


    THolder<IDataProviderBuilder> CreateDataProviderBuilder(
        EDatasetVisitorType visitorType,
        const TDataProviderBuilderOptions& options,
        TDatasetSubset loadSubset,
        NPar::ILocalExecutor* localExecutor
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
