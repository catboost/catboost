#include "meta_info.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/serialization.h>

#include <util/generic/algorithm.h>
#include <util/generic/ptr.h>
#include <util/generic/xrange.h>
#include <util/string/split.h>

#include <tuple>


using namespace NCB;


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
    CB_ENSURE(CountColumns(EColumn::GroupId) <= 1, "Too many GroupId columns. Maybe you've specified QueryId and GroupId, QueryId is synonym for GroupId.");
    CB_ENSURE(CountColumns(EColumn::GroupWeight) <= 1, "Too many GroupWeight columns.");
    CB_ENSURE(CountColumns(EColumn::SubgroupId) <= 1, "Too many SubgroupId columns.");
    CB_ENSURE(CountColumns(EColumn::Timestamp) <= 1, "Too many Timestamp columns.");
}

TDataMetaInfo::TDataMetaInfo(
    TMaybe<TDataColumnsMetaInfo>&& columnsInfo,
    ERawTargetType targetType,
    bool hasAdditionalGroupWeight,
    bool hasTimestamp,
    bool hasPairs,
    TMaybe<ui32> additionalBaselineCount,
    TMaybe<const TVector<TString>*> featureNames,
    const TVector<NJson::TJsonValue>& classLabels
)
    : TargetType(targetType)
    , ClassLabels(classLabels)
    , ColumnsInfo(std::move(columnsInfo))
{
    TargetCount = ColumnsInfo->CountColumns(EColumn::Label);
    if (TargetCount) {
        CB_ENSURE(TargetType != ERawTargetType::None, "data has target columns, but target type specified as None");
    } else {
        CB_ENSURE(TargetType == ERawTargetType::None, "data has no target columns, but target type specified as not None");
    }

    BaselineCount = additionalBaselineCount ? *additionalBaselineCount : ColumnsInfo->CountColumns(EColumn::Baseline);
    HasWeights = ColumnsInfo->CountColumns(EColumn::Weight) != 0;
    HasGroupId = ColumnsInfo->CountColumns(EColumn::GroupId) != 0;
    HasGroupWeight = ColumnsInfo->CountColumns(EColumn::GroupWeight) != 0 || hasAdditionalGroupWeight;
    HasSubgroupIds = ColumnsInfo->CountColumns(EColumn::SubgroupId) != 0;
    HasTimestamp = ColumnsInfo->CountColumns(EColumn::Timestamp) != 0 || hasTimestamp;
    HasPairs = hasPairs;

    // if featureNames is defined - take from it, otherwise take from Id in columns
    TVector<TString> finalFeatureNames;
    if (featureNames) {
        finalFeatureNames = **featureNames;
    }

    TVector<ui32> catFeatureIndices;
    TVector<ui32> textFeatureIndices;

    ui32 featureIdx = 0;
    for (const auto& column : ColumnsInfo->Columns) {
        if (IsFactorColumn(column.Type)) {
            if (!featureNames) {
                finalFeatureNames.push_back(column.Id);
            }
            if (column.Type == EColumn::Categ) {
                catFeatureIndices.push_back(featureIdx);
            } else if (column.Type == EColumn::Text) {
                textFeatureIndices.push_back(featureIdx);
            }
            ++featureIdx;
        }
    }

    FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
        featureIdx,
        std::move(catFeatureIndices),
        std::move(textFeatureIndices),
        finalFeatureNames);

    ColumnsInfo->Validate();
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
        HasWeights,
        HasTimestamp,
        HasPairs,
        ClassLabels,
        ColumnsInfo
    ) == std::tie(
        rhs.TargetType,
        rhs.TargetCount,
        rhs.BaselineCount,
        rhs.HasGroupId,
        rhs.HasGroupWeight,
        rhs.HasSubgroupIds,
        rhs.HasWeights,
        rhs.HasTimestamp,
        rhs.HasPairs,
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
                "Baseline columns count " << BaselineCount << " and class labels count "  << ClassLabels.size() << " are not equal"
            );
        }
    }
}

TVector<TString> TDataColumnsMetaInfo::GenerateFeatureIds(const TMaybe<TVector<TString>>& header) const {
    TVector<TString> featureIds;
    // TODO: this convoluted logic is for compatibility
    if (!AllOf(Columns.begin(), Columns.end(), [](const TColumn& column) { return column.Id.empty(); })) {
        for (auto column : Columns) {
            if (IsFactorColumn(column.Type)) {
                featureIds.push_back(column.Id);
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
        data->HasWeights,
        data->HasTimestamp,
        data->HasPairs,
        data->ClassLabels,
        data->ColumnsInfo
    );
}
