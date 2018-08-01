
#include "quantized.h"

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
