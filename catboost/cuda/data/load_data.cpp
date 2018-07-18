#include "load_data.h"

#include <catboost/cuda/data/binarized_features_meta_info.h>
#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/libs/column_description/column.h>
#include <catboost/libs/data/doc_pool_data_provider.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/data_util/path_with_scheme.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/quantization_schema/detail.h>
#include <catboost/libs/quantization_schema/schema.h>
#include <catboost/libs/quantization_schema/serialization.h>
#include <catboost/libs/quantized_pool/pool.h>
#include <catboost/libs/quantized_pool/serialization.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/is_in.h>
#include <util/system/types.h>
#include <util/system/unaligned_mem.h>

#include <limits>

using NCB::NQuantizationSchemaDetail::NanModeFromProto;

namespace NCatboostCuda {

    static inline void ValidateWeights(const TVector<float>& weights) {
        bool hasNonZero = false;
        for (const auto& w : weights) {
            CB_ENSURE(w >= 0, "Weights can't be negative");
            hasNonZero |= Abs(w) != 0;
        }
        CB_ENSURE(hasNonZero, "Error: all weights are zero");
    }

    void TDataProviderBuilder::StartNextBlock(ui32 blockSize) {
        Cursor = DataProvider.Targets.size();
        const auto newDataSize = Cursor + blockSize;

        DataProvider.Targets.resize(newDataSize);
        DataProvider.Weights.resize(newDataSize, 1.0);
        DataProvider.QueryIds.resize(newDataSize);
        DataProvider.SubgroupIds.resize(newDataSize);
        DataProvider.Timestamp.resize(newDataSize);

        for (ui32 i = Cursor; i < DataProvider.QueryIds.size(); ++i) {
            DataProvider.QueryIds[i] = TGroupId(i);
            DataProvider.SubgroupIds[i] = i;
        }

        for (auto& baseline : DataProvider.Baseline) {
            baseline.resize(newDataSize);
        }

        for (ui32 featureId = 0; featureId < FeatureBlobs.size(); ++featureId) {
            if (IgnoreFeatures.count(featureId) == 0) {
                FeatureBlobs[featureId].resize(newDataSize * GetBytesPerFeature(featureId));
            }
        }

        DataProvider.DocIds.resize(newDataSize);
    }

    static inline bool HasQueryIds(const TVector<TGroupId>& qids) {
        for (ui32 i = 0; i < qids.size(); ++i) {
            if (qids[i] != TGroupId(i)) {
                return true;
            }
        }
        return false;
    }

    template <class T>
    static inline TVector<T> MakeOrderedLine(const TVector<ui8>& source,
                                             const TVector<ui64>& order) {
        CB_ENSURE(source.size() == sizeof(T) * order.size(), "Error: size should be consistent " << source.size() << "  " << order.size() << " " << sizeof(T));
        TVector<T> line(order.size());

        for (size_t i = 0; i < order.size(); ++i) {
            const T* rawSourcePtr = reinterpret_cast<const T*>(source.data());
            line[i] = rawSourcePtr[order[i]];
        }
        return line;
    }

    void TDataProviderBuilder::Finish() {
        CB_ENSURE(!IsDone, "Error: can't finish more than once");
        DataProvider.Features.reserve(FeatureBlobs.size());

        DataProvider.Order.resize(DataProvider.Targets.size());
        std::iota(DataProvider.Order.begin(),
                  DataProvider.Order.end(), 0);

        if (!AreEqualTo<ui64>(DataProvider.Timestamp, 0)) {
            ShuffleFlag = false;
            DataProvider.Order = CreateOrderByKey(DataProvider.Timestamp);
        }

        bool hasQueryIds = HasQueryIds(DataProvider.QueryIds);
        if (!hasQueryIds) {
            DataProvider.QueryIds.resize(0);
        }

        //TODO(noxoomo): it's not safe here, if we change order with shuffle everything'll go wrong
        if (Pairs.size()) {
            //they are local, so we don't need shuffle
            CB_ENSURE(hasQueryIds, "Error: for GPU pairwise learning you should provide query id column. Query ids will be used to split data between devices and for dynamic boosting learning scheme.");
            DataProvider.FillQueryPairs(Pairs);
        }

        if (ShuffleFlag) {
            if (hasQueryIds) {
                //should not change order inside query for pairs consistency
                QueryConsistentShuffle(Seed, 1, DataProvider.QueryIds, &DataProvider.Order);
            } else {
                Shuffle(Seed, 1, DataProvider.Targets.size(), &DataProvider.Order);
            }
            DataProvider.SetShuffleSeed(Seed);
        }

        if (ShuffleFlag || !DataProvider.Timestamp.empty()) {
            DataProvider.ApplyOrderToMetaColumns();
        }

        TVector<TString> featureNames;
        featureNames.resize(FeatureBlobs.size());

        TAdaptiveLock lock;

        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(BuildThreads - 1);

        TVector<TFeatureColumnPtr> featureColumns(FeatureBlobs.size());

        if (!IsTest) {
            RegisterFeaturesInFeatureManager(featureColumns);
        }

        NPar::ParallelFor(executor, 0, static_cast<ui32>(FeatureBlobs.size()), [&](ui32 featureId) {
            auto featureName = GetFeatureName(featureId);
            featureNames[featureId] = featureName;

            if (FeatureBlobs[featureId].size() == 0) {
                return;
            }

            EFeatureValuesType featureValuesType = FeatureTypes[featureId];

            if (featureValuesType == EFeatureValuesType::Categorical) {
                CB_ENSURE(featureValuesType == EFeatureValuesType::Categorical, "Wrong type " << featureValuesType);

                auto line = MakeOrderedLine<float>(FeatureBlobs[featureId],
                                                   DataProvider.Order);

                static_assert(sizeof(float) == sizeof(ui32), "Error: float size should be equal to ui32 size");
                const bool shouldSkip = IsTest && (CatFeaturesPerfectHashHelper.GetUniqueValues(featureId) == 0);
                if (!shouldSkip) {
                    auto data = CatFeaturesPerfectHashHelper.UpdatePerfectHashAndBinarize(featureId,
                                                                                          ~line,
                                                                                          line.size());

                    const ui32 uniqueValues = CatFeaturesPerfectHashHelper.GetUniqueValues(featureId);

                    if (uniqueValues > 1) {
                        auto compressedData = CompressVector<ui64>(~data, line.size(), IntLog2(uniqueValues));
                        featureColumns[featureId] = MakeHolder<TCatFeatureValuesHolder>(featureId,
                                                                                        line.size(),
                                                                                        std::move(compressedData),
                                                                                        uniqueValues,
                                                                                        featureName);
                    }
                }
            } else if (featureValuesType == EFeatureValuesType::BinarizedFloat) {
                const TVector<float>& borders = Borders.at(featureId);
                const ENanMode nanMode = NanModes.at(featureId);
                if (borders.ysize() == 0) {
                    MATRIXNET_DEBUG_LOG << "Float Feature #" << featureId << " is empty" << Endl;
                    return;
                }

                TVector<ui8> binarizedData = MakeOrderedLine<ui8>(FeatureBlobs[featureId],
                                                                  DataProvider.Order);

                const int binCount = static_cast<const int>(borders.size() + 1 + (ENanMode::Forbidden != nanMode));
                auto compressedLine = CompressVector<ui64>(binarizedData, IntLog2(binCount));

                featureColumns[featureId] = MakeHolder<TBinarizedFloatValuesHolder>(featureId,
                                                                                    DataProvider.Order.size(),
                                                                                    nanMode,
                                                                                    borders,
                                                                                    std::move(compressedLine),
                                                                                    featureName);
                with_lock (lock) {
                    FeaturesManager.SetOrCheckNanMode(*featureColumns[featureId],
                                                      nanMode);
                }
            } else {
                CB_ENSURE(featureValuesType == EFeatureValuesType::Float, "Wrong feature values type (" << featureValuesType << ") for feature #" << featureId);
                TVector<float> line(DataProvider.Order.size());
                for (ui32 i = 0; i < DataProvider.Order.size(); ++i) {
                    const float* floatFeatureSource = reinterpret_cast<float*>(FeatureBlobs[featureId].data());
                    line[i] = floatFeatureSource[DataProvider.Order[i]];
                }
                auto floatFeature = MakeHolder<TFloatValuesHolder>(featureId,
                                                                   std::move(line),
                                                                   featureName);

                TVector<float>& borders = Borders[featureId];

                auto& nanMode = NanModes[featureId];
                {
                    TGuard<TAdaptiveLock> guard(lock);
                    nanMode = FeaturesManager.GetOrComputeNanMode(*floatFeature);
                }

                if (FeaturesManager.HasFloatFeatureBorders(*floatFeature)) {
                    borders = FeaturesManager.GetFloatFeatureBorders(*floatFeature);
                }

                if (borders.empty() && !IsTest) {
                    const auto& floatValues = floatFeature->GetValues();
                    NCatboostOptions::TBinarizationOptions config = FeaturesManager.GetFloatFeatureBinarization();
                    config.NanMode = nanMode;
                    borders = BuildBorders(floatValues, floatFeature->GetId(), config);
                }
                if (borders.ysize() == 0) {
                    MATRIXNET_DEBUG_LOG << "Float Feature #" << featureId << " is empty" << Endl;
                    return;
                }

                auto binarizedData = BinarizeLine(floatFeature->GetValues().data(),
                                                  floatFeature->GetValues().size(),
                                                  nanMode,
                                                  borders);

                const int binCount = static_cast<const int>(borders.size() + 1 + (ENanMode::Forbidden != nanMode));
                auto compressedLine = CompressVector<ui64>(binarizedData, IntLog2(binCount));

                featureColumns[featureId] = MakeHolder<TBinarizedFloatValuesHolder>(featureId,
                                                                                    floatFeature->GetValues().size(),
                                                                                    nanMode,
                                                                                    borders,
                                                                                    std::move(compressedLine),
                                                                                    featureName);
            }

            //Free memory
            {
                auto emptyVec = TVector<ui8>();
                FeatureBlobs[featureId].swap(emptyVec);
            }
        });

        for (ui32 featureId = 0; featureId < featureColumns.size(); ++featureId) {
            if (FeatureTypes[featureId] == EFeatureValuesType::Categorical) {
                if (featureColumns[featureId] == nullptr && (!IsTest)) {
                    MATRIXNET_DEBUG_LOG << "Cat Feature #" << featureId << " is empty" << Endl;
                }
            } else if (featureColumns[featureId] != nullptr) {
                if (!FeaturesManager.HasFloatFeatureBordersForDataProviderFeature(featureId)) {
                    FeaturesManager.SetFloatFeatureBordersForDataProviderId(featureId,
                                                                            std::move(Borders[featureId]));
                }
            }
            if (featureColumns[featureId] != nullptr) {
                DataProvider.Features.push_back(std::move(featureColumns[featureId]));
            }
        }

        DataProvider.BuildIndicesRemap();

        if (!IsTest) {
            TOnCpuGridBuilderFactory gridBuilderFactory;
            FeaturesManager.SetTargetBorders(TBordersBuilder(gridBuilderFactory,
                                                             DataProvider.GetTargets())(FeaturesManager.GetTargetBinarizationDescription()));
        }

        DataProvider.FeatureNames = featureNames;

        if (ClassesWeights.size()) {
            Reweight(DataProvider.Targets, ClassesWeights, &DataProvider.Weights);
        }
        ValidateWeights(DataProvider.Weights);

        IsDone = true;
    }

    void TDataProviderBuilder::WriteBinarizedFeatureToBlobImpl(ui32 localIdx, ui32 featureId, ui8 feature) {
        Y_ASSERT(IgnoreFeatures.count(featureId) == 0);
        Y_ASSERT(FeatureTypes[featureId] == EFeatureValuesType::BinarizedFloat);
        ui8* featureColumn = FeatureBlobs[featureId].data();
        featureColumn[GetLineIdx(localIdx)] = feature;
    }

    void TDataProviderBuilder::WriteFloatOrCatFeatureToBlobImpl(ui32 localIdx, ui32 featureId, float feature) {
        Y_ASSERT(IgnoreFeatures.count(featureId) == 0);
        Y_ASSERT(FeatureTypes[featureId] == EFeatureValuesType::Float || FeatureTypes[featureId] == EFeatureValuesType::Categorical);

        auto* featureColumn = reinterpret_cast<float*>(FeatureBlobs[featureId].data());
        featureColumn[GetLineIdx(localIdx)] = feature;
    }

    void TDataProviderBuilder::Start(const TPoolMetaInfo& poolMetaInfo,
                                     int docCount,
                                     const TVector<int>& catFeatureIds) {
        DataProvider.Features.clear();

        DataProvider.Baseline.clear();
        DataProvider.Baseline.resize(poolMetaInfo.BaselineCount);

        Cursor = 0;
        IsDone = false;

        FeatureBlobs.clear();
        FeatureBlobs.resize(poolMetaInfo.FeatureCount);

        FeatureTypes.resize(poolMetaInfo.FeatureCount, EFeatureValuesType::Float);
        for (int catFeature : catFeatureIds) {
            FeatureTypes[catFeature] = EFeatureValuesType::Categorical;
        }
        Borders.resize(poolMetaInfo.FeatureCount);
        NanModes.resize(poolMetaInfo.FeatureCount);

        for (size_t i = 0; i < BinarizedFeaturesMetaInfo.BinarizedFeatureIds.size(); ++i) {
            const size_t binarizedFeatureId = static_cast<const size_t>(BinarizedFeaturesMetaInfo.BinarizedFeatureIds[i]);
            const TVector<float>& borders = BinarizedFeaturesMetaInfo.Borders.at(i);
            CB_ENSURE(binarizedFeatureId < poolMetaInfo.FeatureCount, "Error: binarized feature " << binarizedFeatureId << " is out of range");
            FeatureTypes[binarizedFeatureId] = EFeatureValuesType::BinarizedFloat;
            NanModes[binarizedFeatureId] = BinarizedFeaturesMetaInfo.NanModes.at(i);
            Borders[binarizedFeatureId] = borders;
        }

        for (ui32 i = 0; i < poolMetaInfo.FeatureCount; ++i) {
            if (!IgnoreFeatures.has(i)) {
                ui32 bytesPerFeature = GetBytesPerFeature(i);
                FeatureBlobs[i].reserve(docCount * bytesPerFeature);
            }
        }

        DataProvider.CatFeatureIds = TSet<int>(catFeatureIds.begin(), catFeatureIds.end());
    }
}

static THashMap<size_t, size_t> GetColumnIndexToFeatureIndexMap(const NCB::TQuantizedPool& pool) {
    THashMap<size_t, size_t> map;
    for (size_t i = 0; i < pool.ColumnTypes.size(); ++i) {
        const auto localIndex = pool.ColumnIndexToLocalIndex.at(i);
        const auto columnType = pool.ColumnTypes[localIndex];
        if (!IsFactorColumn(columnType)) {
            continue;
        }

        map.emplace(i, map.size());
    }
    return map;
}

static NCatboostCuda::TBinarizedFloatFeaturesMetaInfo GetQuantizedFeatureMetaInfo(
    const NCB::TQuantizedPool& pool) {

    NCatboostCuda::TBinarizedFloatFeaturesMetaInfo metainfo;

    size_t featureIndex = 0;
    for (size_t i = 0; i < pool.ColumnTypes.size(); ++i) {
        const auto localIndex = pool.ColumnIndexToLocalIndex.at(i);
        const auto columnType = pool.ColumnTypes[localIndex];
        if (!IsFactorColumn(columnType)) {
            continue;
        }

        const auto incFeatureIndex = Finally([&featureIndex]{ ++featureIndex; });
        if (columnType != EColumn::Num) {
            continue;
        }

        metainfo.BinarizedFeatureIds.push_back(featureIndex);
        metainfo.Borders.push_back({});
        metainfo.NanModes.push_back(ENanMode::Min);

        const auto it = pool.QuantizationSchema.GetColumnIndexToSchema().find(i);
        if (it != pool.QuantizationSchema.GetColumnIndexToSchema().end()) {
            metainfo.Borders.back().assign(
                it->second.GetBorders().begin(),
                it->second.GetBorders().end());
            metainfo.NanModes.back() = NanModeFromProto(it->second.GetNanMode());
        }
    }

    return metainfo;
}

static TPoolMetaInfo GetPoolMetaInfo(const NCB::TQuantizedPool& pool) {
    TPoolMetaInfo metaInfo;

    // TODO(yazevnul): these two should be initialized by default c-tor of `TPoolMetaInfo`
    metaInfo.FeatureCount = 0;
    metaInfo.BaselineCount = 0;

    for (size_t i = 0; i < pool.ColumnIndexToLocalIndex.size(); ++i) {
        const auto columnType = pool.ColumnTypes[i];
        metaInfo.FeatureCount += static_cast<ui32>(IsFactorColumn(columnType));
        metaInfo.BaselineCount += static_cast<ui32>(columnType == EColumn::Baseline);
        metaInfo.HasGroupId |= columnType == EColumn::GroupId;
        metaInfo.HasGroupWeight |= columnType == EColumn::GroupWeight;
        metaInfo.HasSubgroupIds |= columnType == EColumn::SubgroupId;
        metaInfo.HasDocIds |= columnType == EColumn::DocId;
        metaInfo.HasWeights |= columnType == EColumn::Weight;
        metaInfo.HasTimestamp |= columnType == EColumn::Timestamp;
    }

    return metaInfo;
}

static TVector<int> GetCategoricalFeatureIds(const NCB::TQuantizedPool& pool) {
    TVector<int> categoricalIds;
    size_t featureIndex = 0;
    for (size_t i = 0; i < pool.ColumnTypes.size(); ++i) {
        const auto localIndex = pool.ColumnIndexToLocalIndex.at(i);
        const auto columnType = pool.ColumnTypes[localIndex];
        if (!IsFactorColumn(columnType)) {
            continue;
        }

        const auto incFeatureIndex = Finally([&featureIndex]{ ++featureIndex; });
        if (columnType == EColumn::Categ) {
            categoricalIds.push_back(featureIndex);
        }
    }

    return categoricalIds;
}

static TVector<size_t> GetIgnoredFeatureIndices(const NCB::TQuantizedPool& pool) {
    TVector<size_t> indices;
    size_t featureIndex = 0;
    for (size_t i = 0; i < pool.ColumnTypes.size(); ++i) {
        const auto localIndex = pool.ColumnIndexToLocalIndex.at(i);
        const auto columnType = pool.ColumnTypes[localIndex];
        if (columnType != EColumn::Num && columnType != EColumn::Categ) {
            continue;
        }

        const auto incFeatureIndex = Finally([&featureIndex]{ ++featureIndex; });
        if (IsIn(pool.IgnoredColumnIndices, i)) {
            indices.push_back(featureIndex);
            continue;
        }

        const auto it = pool.QuantizationSchema.GetColumnIndexToSchema().find(i);
        if (it == pool.QuantizationSchema.GetColumnIndexToSchema().end()) {
            // categorical features are not quantized right now
            indices.push_back(featureIndex);
            continue;
        } else if (it->second.GetBorders().empty()) {
            indices.push_back(featureIndex);
            continue;
        }
    }
    return indices;
}

static void AddColumn(
    const size_t featureIndex,
    const size_t baselineIndex,
    const EColumn columnType,
    const TConstArrayRef<NCB::TQuantizedPool::TChunkDescription> chunks,
    NCatboostCuda::TDataProviderBuilder* const builder) {

    switch (columnType) {
        case EColumn::Num: {
            for (const auto& descriptor : chunks) {
                CB_ENSURE(static_cast<size_t>(descriptor.Chunk->BitsPerDocument()) == sizeof(ui8) * 8);
                TUnalignedMemoryIterator<ui8> it(
                    descriptor.Chunk->Quants()->data(),
                    descriptor.Chunk->Quants()->size());
                for (ui32 i = descriptor.DocumentOffset; !it.AtEnd(); it.Next(), (void)++i) {
                    builder->AddBinarizedFloatFeature(i, featureIndex, it.Cur());
                }
            }
            break;
        }
        case EColumn::Label: {
            for (const auto& descriptor : chunks) {
                CB_ENSURE(static_cast<size_t>(descriptor.Chunk->BitsPerDocument()) == sizeof(float) * 8);
                TUnalignedMemoryIterator<float> it(
                    descriptor.Chunk->Quants()->data(),
                    descriptor.Chunk->Quants()->size());
                for (ui32 i = descriptor.DocumentOffset; !it.AtEnd(); it.Next(), (void)++i) {
                    builder->AddTarget(i, it.Cur());
                }
            }
            break;
        }
        case EColumn::Baseline: {
            for (const auto& descriptor : chunks) {
                CB_ENSURE(static_cast<size_t>(descriptor.Chunk->BitsPerDocument()) == sizeof(double) * 8);
                TUnalignedMemoryIterator<double> it(
                    descriptor.Chunk->Quants()->data(),
                    descriptor.Chunk->Quants()->size());
                for (ui32 i = descriptor.DocumentOffset; !it.AtEnd(); it.Next(), (void)++i) {
                    builder->AddBaseline(i, baselineIndex, it.Cur());
                }
            }
            break;
        }
        case EColumn::Weight:
        case EColumn::GroupWeight: {
            for (const auto& descriptor : chunks) {
                CB_ENSURE(static_cast<size_t>(descriptor.Chunk->BitsPerDocument()) == sizeof(float) * 8);
                TUnalignedMemoryIterator<float> it(
                    descriptor.Chunk->Quants()->data(),
                    descriptor.Chunk->Quants()->size());
                for (ui32 i = descriptor.DocumentOffset; !it.AtEnd(); it.Next(), (void)++i) {
                    builder->AddWeight(i, it.Cur());
                }
            }
            break;
        }
        case EColumn::DocId: {
            const size_t bufSize = std::numeric_limits<ui64>::digits10 + 1;
            char buf[bufSize];
            for (const auto& descriptor : chunks) {
                CB_ENSURE(static_cast<size_t>(descriptor.Chunk->BitsPerDocument()) == sizeof(ui64) * 8);
                TUnalignedMemoryIterator<ui64> it(
                    descriptor.Chunk->Quants()->data(),
                    descriptor.Chunk->Quants()->size());
                for (ui32 i = descriptor.DocumentOffset; !it.AtEnd(); it.Next(), ++i) {
                    ToString(it.Cur(), buf, bufSize);
                    builder->AddDocId(i, buf);
                }
            }
            break;
        }
        case EColumn::GroupId: {
            for (const auto& descriptor : chunks) {
                CB_ENSURE(static_cast<size_t>(descriptor.Chunk->BitsPerDocument()) == sizeof(ui64) * 8);
                TUnalignedMemoryIterator<ui64> it(
                    descriptor.Chunk->Quants()->data(),
                    descriptor.Chunk->Quants()->size());
                for (ui32 i = descriptor.DocumentOffset; !it.AtEnd(); it.Next(), (void)++i) {
                    builder->AddQueryId(i, it.Cur());
                }
            }
            break;
        }
        case EColumn::SubgroupId: {
            for (const auto& descriptor : chunks) {
                CB_ENSURE(static_cast<size_t>(descriptor.Chunk->BitsPerDocument()) == sizeof(ui32) * 8);
                TUnalignedMemoryIterator<ui32> it(
                    descriptor.Chunk->Quants()->data(),
                    descriptor.Chunk->Quants()->size());
                for (ui32 i = descriptor.DocumentOffset; !it.AtEnd(); it.Next(), (void)++i) {
                    builder->AddSubgroupId(i, it.Cur());
                }
            }
            break;
        }
        case EColumn::Categ:
            // TODO(yazevnul): categorical feature quantization on YT is still in progress
        case EColumn::Auxiliary:
            // Should not be present in quantized pool
        case EColumn::Timestamp:
            // not supported by quantized pools right now
        case EColumn::Sparse:
            // not supperted by CatBoost at all
        case EColumn::Prediction: {
            // can't be present in quantized pool
            ythrow TCatboostException() << "Unexpected column type " << columnType;
        }
    }
}

void NCatboostCuda::ReadPool(
    const ::NCB::TPathWithScheme& poolPath,
    const ::NCB::TPathWithScheme& pairsFilePath, // can be uninited
    const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
    const TVector<int>& ignoredFeatures,
    const bool verbose,
    const TVector<TString>& classNames,
    NPar::TLocalExecutor* const localExecutor,
    TDataProviderBuilder* const poolBuilder) {

    if (poolPath.Scheme != "quantized" && poolPath.Scheme != "yt-quantized") {
        ::NCB::ReadPool(
            poolPath,
            pairsFilePath,
            dsvPoolFormatParams,
            ignoredFeatures,
            verbose,
            classNames,
            localExecutor,
            poolBuilder);
        return;
    }

    if (poolPath.Scheme == "yt-quantized") {
        ythrow TCatboostException() << "\"yt-quantized\" schema is not supported yet";
    }

    // TODO(yazevnul): load data in multiple threads. One thread reads from disk, other adds chunk
    // to the `poolBuilder`

    // TODO(yazevnul): load using `TFile::Pread` instead of mapping entire file; at least until we
    // keep this interface where we are not using chunks directly

    NCB::TLoadQuantizedPoolParameters loadParameters;
    loadParameters.LockMemory = false;
    loadParameters.Precharge = false;

    const auto pool = NCB::LoadQuantizedPool(poolPath.Path, loadParameters);

    const auto columnIndexToFeatureIndex = GetColumnIndexToFeatureIndexMap(pool);
    poolBuilder->SetBinarizedFeaturesMetaInfo(GetQuantizedFeatureMetaInfo(pool));
    poolBuilder->AddIgnoredFeatures(GetIgnoredFeatureIndices(pool));
    poolBuilder->Start(
        GetPoolMetaInfo(pool),
        pool.DocumentCount,
        GetCategoricalFeatureIds(pool));
    poolBuilder->StartNextBlock(pool.DocumentCount);

    size_t baselineIndex = 0;
    for (const auto& kv : pool.ColumnIndexToLocalIndex) {
        const auto columnIndex = kv.first;
        const auto localIndex = kv.second;
        const auto columnType = pool.ColumnTypes[localIndex];

        if (pool.Chunks[localIndex].empty()) {
            continue;
        }

        const auto featureIndex = columnIndexToFeatureIndex.Value(columnIndex, 0);
        ::AddColumn(featureIndex, baselineIndex, columnType, pool.Chunks[localIndex], poolBuilder);

        baselineIndex += static_cast<size_t>(columnType == EColumn::Baseline);
    }

    if (pairsFilePath.Inited()) {
        poolBuilder->SetPairs(NCB::ReadPairs(pairsFilePath, pool.DocumentCount));
    }

    poolBuilder->Finish();
}
