#include "meta_info.h"

#include <catboost/libs/column_description/feature_tag.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/json_helpers.h>
#include <catboost/libs/helpers/serialization.h>

#include <util/generic/algorithm.h>
#include <util/generic/ptr.h>
#include <util/generic/xrange.h>
#include <util/string/split.h>

#include <tuple>


using namespace NCB;


TDataColumnsMetaInfo::operator NJson::TJsonValue() const {
    return VectorToJson(Columns);
}

ui32 TDataColumnsMetaInfo::CountColumns(const EColumn columnType) const {
    return CountIf(
        Columns.begin(),
        Columns.end(),
        [&columnType](const auto x) -> bool {
            return x.Type == columnType;
        }
    );
}


void TDataColumnsMetaInfo::Validate() const {
    CB_ENSURE(CountColumns(EColumn::Weight) <= 1, "Too many Weight columns.");
    CB_ENSURE(CountColumns(EColumn::SampleId) <= 1, "Too many SampleId columns.");
    CB_ENSURE(
        CountColumns(EColumn::GroupId) <= 1,
        "Too many GroupId columns. Maybe you've specified QueryId and GroupId, QueryId is a synonym for GroupId."
    );
    CB_ENSURE(CountColumns(EColumn::GroupWeight) <= 1, "Too many GroupWeight columns.");
    CB_ENSURE(CountColumns(EColumn::SubgroupId) <= 1, "Too many SubgroupId columns.");
    CB_ENSURE(CountColumns(EColumn::Timestamp) <= 1, "Too many Timestamp columns.");
}


NCB::TTargetStats::operator NJson::TJsonValue() const {
    return NJson::TJsonValue(NJson::JSON_MAP)
        .InsertValue("MinValue"sv, MinValue)
        .InsertValue("MaxValue"sv, MaxValue);
}


TDataMetaInfo::TDataMetaInfo(
    TMaybe<TDataColumnsMetaInfo>&& columnsInfo,
    ERawTargetType targetType,
    bool hasAdditionalGroupWeight,
    bool hasTimestamp,
    bool hasPairs,
    bool hasGraph,
    bool loadSampleIds,
    bool forceUnitAutoPairWeights,
    TMaybe<ui32> additionalBaselineCount,
    TMaybe<const TVector<TString>*> featureNames,
    TMaybe<const THashMap<TString, TTagDescription>*> featureTags,
    const TVector<NJson::TJsonValue>& classLabels
)
    : TargetType(targetType)
    , ClassLabels(classLabels)
    , ColumnsInfo(std::move(columnsInfo))
{
    ColumnsInfo->Validate();

    FeaturesLayout = TFeaturesLayout::CreateFeaturesLayout(ColumnsInfo->Columns, featureNames, featureTags, hasGraph);

    TargetCount = ColumnsInfo->CountColumns(EColumn::Label);
    if (TargetCount) {
        CB_ENSURE(TargetType != ERawTargetType::None, "data has target columns, but target type specified as None");
    } else {
        CB_ENSURE(
            TargetType == ERawTargetType::None,
            "data has no target columns, but target type specified as not None"
        );
    }

    BaselineCount = additionalBaselineCount ? *additionalBaselineCount : ColumnsInfo->CountColumns(EColumn::Baseline);

    HasGroupId = ColumnsInfo->CountColumns(EColumn::GroupId) != 0;
    HasGroupWeight = ColumnsInfo->CountColumns(EColumn::GroupWeight) != 0 || hasAdditionalGroupWeight;
    HasSubgroupIds = ColumnsInfo->CountColumns(EColumn::SubgroupId) != 0;
    if (loadSampleIds) {
        HasSampleId = ColumnsInfo->CountColumns(EColumn::SampleId) != 0;
        if (HasSampleId) {
            StoreStringColumns = true;
        }
    }
    HasWeights = ColumnsInfo->CountColumns(EColumn::Weight) != 0;
    HasTimestamp = ColumnsInfo->CountColumns(EColumn::Timestamp) != 0 || hasTimestamp;
    HasPairs = hasPairs;
    HasGraph = hasGraph;
    ForceUnitAutoPairWeights = forceUnitAutoPairWeights;

    Validate();
}

bool TDataMetaInfo::EqualTo(const TDataMetaInfo& rhs, bool ignoreSparsity) const {
    if (FeaturesLayout) {
        if (rhs.FeaturesLayout) {
            if (!FeaturesLayout->EqualTo(*rhs.FeaturesLayout, ignoreSparsity)) {
                return false;
            }
        } else {
            return false;
        }
    } else if (rhs.FeaturesLayout) {
        return false;
    }

    return std::tie(
        TargetType,
        TargetCount,
        BaselineCount,
        HasGroupId,
        HasGroupWeight,
        HasSubgroupIds,
        HasSampleId,
        HasWeights,
        HasTimestamp,
        HasPairs,
        HasGraph,
        StoreStringColumns,
        ClassLabels,
        ColumnsInfo
    ) == std::tie(
        rhs.TargetType,
        rhs.TargetCount,
        rhs.BaselineCount,
        rhs.HasGroupId,
        rhs.HasGroupWeight,
        rhs.HasSubgroupIds,
        rhs.HasSampleId,
        rhs.HasWeights,
        rhs.HasTimestamp,
        rhs.HasPairs,
        rhs.HasGraph,
        rhs.StoreStringColumns,
        ClassLabels,
        rhs.ColumnsInfo
    );
}

void TDataMetaInfo::Validate() const {
    CB_ENSURE(GetFeatureCount() > 0, "Pool should have at least one factor");
    CB_ENSURE(!HasGroupWeight || (HasGroupWeight && HasGroupId),
        "You should provide GroupId when providing GroupWeight.");
    if ((BaselineCount != 0) && !ClassLabels.empty()) {
        if (BaselineCount == 1) {
            CB_ENSURE(
                ClassLabels.size() == 2,
                "Inconsistent columns specification: Baseline columns count " << BaselineCount
                << " and class labels count "  << ClassLabels.size() << ". Either wrong baseline count for "
                " multiclassification or wrong class count for binary classification"
            );
        } else {
            CB_ENSURE(
                BaselineCount == ClassLabels.size(),
                "Baseline columns count " << BaselineCount << " and class labels count "
                << ClassLabels.size() << " are not equal"
            );
        }
    }
}

TDataMetaInfo::operator NJson::TJsonValue() const {
    NJson::TJsonValue result(NJson::JSON_MAP);
    result.InsertValue("ObjectCount"sv, ObjectCount);
    result.InsertValue("FeaturesLayout"sv, *FeaturesLayout);
    result.InsertValue("MaxCatFeaturesUniqValuesOnLearn"sv, MaxCatFeaturesUniqValuesOnLearn);
    result.InsertValue("TargetType"sv, ToString(TargetType));
    result.InsertValue("TargetCount"sv, TargetCount);

    if (TargetStats) {
        result.InsertValue("TargetStats"sv, *TargetStats);
    }

    result.InsertValue("BaselineCount"sv, BaselineCount);
    result.InsertValue("HasGroupId"sv, HasGroupId);
    result.InsertValue("HasGroupWeight"sv, HasGroupWeight);
    result.InsertValue("HasSubgroupIds"sv, HasSubgroupIds);
    result.InsertValue("HasSampleId"sv, HasSampleId);
    result.InsertValue("HasWeights"sv, HasWeights);
    result.InsertValue("HasTimestamp"sv, HasTimestamp);
    result.InsertValue("HasPairs"sv, HasPairs);
    result.InsertValue("HasGraph"sv, HasGraph);
    result.InsertValue("StoreStringColumns"sv, StoreStringColumns);
    result.InsertValue("ForceUnitAutoPairWeights"sv, ForceUnitAutoPairWeights);

    if (!ClassLabels.empty()) {
        result.InsertValue("ClassLabels"sv, VectorToJson(ClassLabels));
    }

    if (ColumnsInfo) {
        result.InsertValue("ColumnsInfo"sv, *ColumnsInfo);
    }

    return result;
}


static bool AreAllColumnIdsEmpty(TConstArrayRef<TColumn> columns, const TMaybe<TVector<TString>>& header) {
    for (const auto& column : columns) {
        if (column.Type == EColumn::Features) {
            CB_ENSURE(!header.Defined(), "Header columns cannot be defined if Features meta column is present");
            for (const auto& subColumn : column.SubColumns) {
                CB_ENSURE(
                    IsFactorColumn(subColumn.Type),
                    "Non-features sub columns are not supported in Features meta column"
                );
                if (!subColumn.Id.empty()) {
                    return false;
                }
            }
        } else if (!column.Id.empty()) {
            return false;
        }
    }
    return true;
}

TVector<TString> TDataColumnsMetaInfo::GenerateFeatureIds(const TMaybe<TVector<TString>>& header) const {
    TVector<TString> featureIds;
    // TODO: this convoluted logic is for compatibility
    if (!AreAllColumnIdsEmpty(Columns, header)) {
        for (auto column : Columns) {
            if (IsFactorColumn(column.Type)) {
                featureIds.push_back(column.Id);
            } else if (column.Type == EColumn::Features) {
                for (const auto& subColumn : column.SubColumns) {
                    featureIds.push_back(subColumn.Id);
                }
            }
        }
    } else if (header.Defined()) {
        for (auto i : xrange(header->size())) {
            if (IsFactorColumn(Columns[i].Type)) {
                featureIds.push_back((*header)[i]);
            }
        }
    }
    return featureIds;
}

void NCB::AddWithShared(IBinSaver* binSaver, TDataMetaInfo* data) {
    AddWithShared(binSaver, &(data->FeaturesLayout));
    binSaver->AddMulti(
        data->TargetType,
        data->TargetCount,
        data->BaselineCount,
        data->HasGroupId,
        data->HasGroupWeight,
        data->HasSubgroupIds,
        data->HasSampleId,
        data->HasWeights,
        data->HasTimestamp,
        data->HasPairs,
        data->HasGraph,
        data->StoreStringColumns,
        data->ClassLabels,
        data->ColumnsInfo
    );
}
