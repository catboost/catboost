
#include "quantized.h"

#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/libs/helpers/exception.h>

#include <util/system/unaligned_mem.h>

THashMap<size_t, size_t> GetColumnIndexToFeatureIndexMap(const NCB::TQuantizedPool& pool) {
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

TPoolMetaInfo GetPoolMetaInfo(const NCB::TQuantizedPool& pool) {
    TPoolMetaInfo metaInfo;

    // TODO(yazevnul): these two should be initialized by default c-tor of `TPoolMetaInfo`
    metaInfo.FeatureCount = 0;
    metaInfo.BaselineCount = 0;
    const size_t columnsCount = pool.ColumnIndexToLocalIndex.size();
    metaInfo.ColumnsInfo = TPoolColumnsMetaInfo();
    metaInfo.ColumnsInfo->Columns.resize(columnsCount);

    for (size_t i = 0; i < columnsCount; ++i) {
        const auto columnType = pool.ColumnTypes[i];
        metaInfo.FeatureCount += static_cast<ui32>(IsFactorColumn(columnType));
        metaInfo.BaselineCount += static_cast<ui32>(columnType == EColumn::Baseline);
        metaInfo.HasGroupId |= columnType == EColumn::GroupId;
        metaInfo.HasGroupWeight |= columnType == EColumn::GroupWeight;
        metaInfo.HasSubgroupIds |= columnType == EColumn::SubgroupId;
        metaInfo.HasDocIds |= columnType == EColumn::DocId;
        metaInfo.HasWeights |= columnType == EColumn::Weight;
        metaInfo.HasTimestamp |= columnType == EColumn::Timestamp;
        metaInfo.ColumnsInfo->Columns[i].Type = columnType;
    }

    return metaInfo;
}

TVector<int> GetCategoricalFeatureIndices(const NCB::TQuantizedPool& pool) {
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

TVector<int> GetIgnoredFeatureIndices(const NCB::TQuantizedPool& pool) {
    TVector<int> indices;
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

        const auto it = pool.QuantizationSchema.GetFeatureIndexToSchema().find(featureIndex);
        if (it == pool.QuantizationSchema.GetFeatureIndexToSchema().end()) {
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

void AddColumn(
    const size_t featureIndex,
    const size_t baselineIndex,
    const EColumn columnType,
    const TConstArrayRef<NCB::TQuantizedPool::TChunkDescription> chunks,
    NCB::IPoolBuilder* const builder) {

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
