#include "quantized.h"

#include <catboost/libs/column_description/column.h>
#include <catboost/libs/data/loader.h>
#include <catboost/libs/helpers/exception.h>

#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/pool_metainfo_options.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>

THashMap<size_t, size_t> GetColumnIndexToTargetIndexMap(const NCB::TQuantizedPool& pool) {
    TVector<size_t> columnIndices;
    columnIndices.reserve(pool.ColumnIndexToLocalIndex.size());
    for (const auto& [columnIndex, localIndex] : pool.ColumnIndexToLocalIndex) {
        const auto columnType = pool.ColumnTypes[localIndex];
        if (columnType != EColumn::Label) {
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
        if (EColumn::Baseline != pool.ColumnTypes[localIdx]) {
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

NCB::TDataMetaInfo GetDataMetaInfo(
    const NCB::TQuantizedPool& pool,
    bool hasAdditionalGroupWeight,
    bool hasTimestamps,
    bool hasPairs,
    bool forceUnitAutoPairWeights,
    TMaybe<ui32> baselineCount,
    const NCB::TPathWithScheme& featureNamesPath,
    const NCB::TPathWithScheme& poolMetaInfoPath
) {
    const size_t columnsCount = pool.ColumnIndexToLocalIndex.size();
    NCB::TDataColumnsMetaInfo dataColumnsMetaInfo;
    dataColumnsMetaInfo.Columns.resize(columnsCount);

    bool hasTargets = false;

    for (const auto [columnIndex, localIndex] : pool.ColumnIndexToLocalIndex) {
        const auto columnType = pool.ColumnTypes[localIndex];
        if (columnType == EColumn::Label) {
            hasTargets = true;
        }
        dataColumnsMetaInfo.Columns[columnIndex].Type = pool.ColumnTypes[localIndex];
        dataColumnsMetaInfo.Columns[columnIndex].Id = pool.ColumnNames[localIndex];
    }

    NCB::ERawTargetType targetType;
    if (hasTargets) {
        if (pool.QuantizationSchema.IntegerClassLabelsSize()) {
            targetType = NCB::ERawTargetType::Integer;
        } else if (pool.QuantizationSchema.FloatClassLabelsSize()) {
            targetType = NCB::ERawTargetType::Float;
        } else if (pool.QuantizationSchema.ClassNamesSize()) {
            targetType = NCB::ERawTargetType::String;
        } else {
            targetType = NCB::ERawTargetType::Float;
        }
    } else {
        targetType = NCB::ERawTargetType::None;
    }

    const TVector<TString> featureNames = NCB::GetFeatureNames(
        dataColumnsMetaInfo,
        /*headerColumns*/ Nothing(),
        featureNamesPath
    );

    const auto poolMetaInfoOptions = NCatboostOptions::LoadPoolMetaInfoOptions(poolMetaInfoPath);

    NCB::TDataMetaInfo metaInfo(
        std::move(dataColumnsMetaInfo),
        targetType,
        hasAdditionalGroupWeight,
        hasTimestamps,
        hasPairs,
        /*hasGraph*/ false,
        /*loadSampleIds*/ false,
        forceUnitAutoPairWeights,
        baselineCount,
        &featureNames,
        &poolMetaInfoOptions.Tags.Get()
    );
    metaInfo.Validate();
    return metaInfo;
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

        if (columnType == EColumn::Num) {
            const auto it = pool.QuantizationSchema.GetFeatureIndexToSchema().find(featureIndex);

            if (it == pool.QuantizationSchema.GetFeatureIndexToSchema().end() ||
                it->second.GetBorders().empty())
            {
                indices.push_back(SafeIntegerCast<ui32>(featureIndex));
                continue;
            }
        } else {
            CB_ENSURE(columnType == EColumn::Categ);
            const auto it = pool.QuantizationSchema.GetCatFeatureIndexToSchema().find(featureIndex);

            if (it == pool.QuantizationSchema.GetCatFeatureIndexToSchema().end() ||
                (it->second.GetPerfectHashes().size() < 2))
            {
                indices.push_back(SafeIntegerCast<ui32>(featureIndex));
                continue;
            }
        }
    }

    Sort(indices);

    return indices;
}
