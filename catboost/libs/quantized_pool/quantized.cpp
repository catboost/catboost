#include "quantized.h"

#include <catboost/libs/column_description/column.h>

#include <util/generic/algorithm.h>

THashMap<size_t, size_t> GetColumnIndexToFeatureIndexMap(const NCB::TQuantizedPool& pool) {
    TVector<size_t> columnIndices;
    columnIndices.reserve(pool.ColumnIndexToLocalIndex.size());
    for (const auto& kv : pool.ColumnIndexToLocalIndex) {
        const auto columnType = pool.ColumnTypes[kv.second];
        if (!IsFactorColumn(columnType)) {
            continue;
        }

        columnIndices.push_back(kv.first);
    }

    Sort(columnIndices);

    THashMap<size_t, size_t> map;
    map.reserve(columnIndices.size());
    for (size_t i = 0; i < columnIndices.size(); ++i) {
        map.emplace(columnIndices[i], map.size());
    }

    return map;
}

TPoolMetaInfo GetPoolMetaInfo(const NCB::TQuantizedPool& pool) {
    TPoolMetaInfo metaInfo;

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
    const auto columnIndexToFeatureIndex = GetColumnIndexToFeatureIndexMap(pool);

    TVector<int> categoricalIds;
    for (const auto& kv : pool.ColumnIndexToLocalIndex) {
        const auto columnType = pool.ColumnTypes[kv.second];
        if (columnType != EColumn::Categ) {
            continue;
        }

        const auto featureIndex = columnIndexToFeatureIndex.at(kv.first);
        categoricalIds.push_back(static_cast<int>(featureIndex));
    }

    Sort(categoricalIds);

    return categoricalIds;
}

TVector<int> GetIgnoredFeatureIndices(const NCB::TQuantizedPool& pool) {
    const auto columnIndexToFeatureIndex = GetColumnIndexToFeatureIndexMap(pool);
    TVector<int> indices;
    for (const auto& kv : pool.ColumnIndexToLocalIndex) {
        const auto columnType = pool.ColumnTypes[kv.second];
        if (columnType != EColumn::Num && columnType != EColumn::Categ) {
            continue;
        }

        const auto featureIndex = columnIndexToFeatureIndex.at(kv.first);
        if (IsIn(pool.IgnoredColumnIndices, kv.first)) {
            indices.push_back(static_cast<int>(featureIndex));
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

    Sort(indices);

    return indices;
}
