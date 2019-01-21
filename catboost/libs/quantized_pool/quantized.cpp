#include "quantized.h"

#include <catboost/libs/column_description/column.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>

THashMap<size_t, size_t> GetColumnIndexToFlatIndexMap(const NCB::TQuantizedPool& pool) {
    TVector<size_t> columnIndices;
    columnIndices.reserve(pool.ColumnIndexToLocalIndex.size());
    for (const auto& [columnIndex, localIndex] : pool.ColumnIndexToLocalIndex) {
        const auto columnType = pool.ColumnTypes[localIndex];
        if (!IsFactorColumn(columnType)) {
            continue;
        }

        columnIndices.push_back(columnIndex);
    }

    Sort(columnIndices);

    THashMap<size_t, size_t> map;
    map.reserve(columnIndices.size());
    for (size_t i = 0; i < columnIndices.size(); ++i) {
        map.emplace(columnIndices[i], map.size());
    }

    return map;
}

THashMap<size_t, size_t> GetColumnIndexToBaselineIndexMap(const NCB::TQuantizedPool& pool) {
    TVector<size_t> baselineIndices;
    for (const auto [columnIdx, localIdx] : pool.ColumnIndexToLocalIndex) {
        if (EColumn::Baseline == pool.ColumnTypes[localIdx]) {
            continue;
        }

        baselineIndices.push_back(columnIdx);
    }

    Sort(baselineIndices);

    THashMap<size_t, size_t> map;
    for (size_t i = 0, iEnd = baselineIndices.size(); i < iEnd; ++i) {
        map.emplace(baselineIndices[i], map.size());
    }
    return map;
}

TVector<TString> GetFlatFeatureNames(const NCB::TQuantizedPool& pool) {
    const auto columnIndexToFlatIndex = GetColumnIndexToFlatIndexMap(pool);
    TVector<TString> names(columnIndexToFlatIndex.size());
    for (const auto [columnIndex, flatIndex] : columnIndexToFlatIndex) {
        const auto localIndex = pool.ColumnIndexToLocalIndex.at(columnIndex);
        names[flatIndex] = pool.ColumnNames[localIndex];
    }
    return names;
}

THashMap<size_t, size_t> GetColumnIndexToNumericFeatureIndexMap(const NCB::TQuantizedPool& pool) {
    TVector<size_t> columnIndices;
    columnIndices.reserve(pool.ColumnIndexToLocalIndex.size());
    for (const auto [columnIndex, localIndex] : pool.ColumnIndexToLocalIndex) {
        const auto columnType = pool.ColumnTypes[localIndex];
        if (columnType != EColumn::Num) {
            continue;
        }

        columnIndices.push_back(columnIndex);
    }

    Sort(columnIndices);

    THashMap<size_t, size_t> map;
    map.reserve(columnIndices.size());
    for (size_t i = 0; i < columnIndices.size(); ++i) {
        map.emplace(columnIndices[i], map.size());
    }

    return map;
}

NCB::TDataMetaInfo GetDataMetaInfo(
    const NCB::TQuantizedPool& pool,
    bool hasAdditionalGroupWeight,
    bool hasPairs
) {
    const size_t columnsCount = pool.ColumnIndexToLocalIndex.size();
    NCB::TDataColumnsMetaInfo dataColumnsMetaInfo;
    dataColumnsMetaInfo.Columns.resize(columnsCount);

    for (const auto [columnIndex, localIndex] : pool.ColumnIndexToLocalIndex) {
        dataColumnsMetaInfo.Columns[columnIndex].Type = pool.ColumnTypes[localIndex];
        dataColumnsMetaInfo.Columns[columnIndex].Id = pool.ColumnNames[localIndex];
    }

    NCB::TDataMetaInfo metaInfo(std::move(dataColumnsMetaInfo), hasAdditionalGroupWeight, hasPairs);
    metaInfo.Validate();
    return metaInfo;
}

TVector<int> GetCategoricalFeatureIndices(const NCB::TQuantizedPool& pool) {
    const auto columnIndexToFeatureIndex = GetColumnIndexToFlatIndexMap(pool);

    TVector<int> categoricalIds;
    for (const auto [columnIndex, localIndex] : pool.ColumnIndexToLocalIndex) {
        const auto columnType = pool.ColumnTypes[localIndex];
        if (columnType != EColumn::Categ) {
            continue;
        }

        const auto featureIndex = columnIndexToFeatureIndex.at(columnIndex);
        categoricalIds.push_back(static_cast<int>(featureIndex));
    }

    Sort(categoricalIds);

    return categoricalIds;
}

TVector<ui32> GetIgnoredFlatIndices(const NCB::TQuantizedPool& pool) {
    const auto columnIndexToFeatureIndex = GetColumnIndexToFlatIndexMap(pool);
    TVector<ui32> indices;
    for (const auto [columnIndex, localIndex] : pool.ColumnIndexToLocalIndex) {
        const auto columnType = pool.ColumnTypes[localIndex];
        if (columnType != EColumn::Num && columnType != EColumn::Categ) {
            continue;
        }

        const auto featureIndex = columnIndexToFeatureIndex.at(columnIndex);
        if (IsIn(pool.IgnoredColumnIndices, columnIndex)) {
            indices.push_back(SafeIntegerCast<ui32>(featureIndex));
            continue;
        }

        const auto it = pool.QuantizationSchema.GetFeatureIndexToSchema().find(featureIndex);
        if (it == pool.QuantizationSchema.GetFeatureIndexToSchema().end()) {
            // categorical features are not quantized right now
            indices.push_back(SafeIntegerCast<ui32>(featureIndex));
            continue;
        } else if (it->second.GetBorders().empty()) {
            indices.push_back(SafeIntegerCast<ui32>(featureIndex));
            continue;
        }
    }

    Sort(indices);

    return indices;
}
