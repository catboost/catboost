#include "meta_info.h"

#include <catboost/libs/helpers/exception.h>

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
    CB_ENSURE(CountColumns(EColumn::Label) <= 1, "Too many Label columns.");
    CB_ENSURE(CountColumns(EColumn::DocId) <= 1, "Too many DocId columns.");
    CB_ENSURE(CountColumns(EColumn::GroupId) <= 1, "Too many GroupId columns. Maybe you've specified QueryId and GroupId, QueryId is synonym for GroupId.");
    CB_ENSURE(CountColumns(EColumn::GroupWeight) <= 1, "Too many GroupWeight columns.");
    CB_ENSURE(CountColumns(EColumn::SubgroupId) <= 1, "Too many SubgroupId columns.");
    CB_ENSURE(CountColumns(EColumn::Timestamp) <= 1, "Too many Timestamp columns.");
}

TDataMetaInfo::TDataMetaInfo(
    TMaybe<TDataColumnsMetaInfo>&& columnsInfo,
    bool hasAdditionalGroupWeight,
    bool hasPairs,
    TMaybe<const TVector<TString>*> featureNames
)
    : ColumnsInfo(std::move(columnsInfo))
{
    HasTarget = ColumnsInfo->CountColumns(EColumn::Label);
    BaselineCount = ColumnsInfo->CountColumns(EColumn::Baseline);
    HasWeights = ColumnsInfo->CountColumns(EColumn::Weight) != 0;
    HasGroupId = ColumnsInfo->CountColumns(EColumn::GroupId) != 0;
    HasGroupWeight = ColumnsInfo->CountColumns(EColumn::GroupWeight) != 0 || hasAdditionalGroupWeight;
    HasSubgroupIds = ColumnsInfo->CountColumns(EColumn::SubgroupId) != 0;
    HasTimestamp = ColumnsInfo->CountColumns(EColumn::Timestamp) != 0;
    HasPairs = hasPairs;

    // if featureNames is defined - take from it, otherwise take from Id in columns
    TVector<TString> finalFeatureNames;
    if (featureNames) {
        finalFeatureNames = **featureNames;
    }

    TVector<ui32> catFeatureIndices;

    ui32 featureIdx = 0;
    for (const auto& column : ColumnsInfo->Columns) {
        if (IsFactorColumn(column.Type)) {
            if (!featureNames) {
                finalFeatureNames.push_back(column.Id);
            }
            if (column.Type == EColumn::Categ) {
                catFeatureIndices.push_back(featureIdx);
            }
            ++featureIdx;
        }
    }

    FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
        featureIdx,
        std::move(catFeatureIndices),
        finalFeatureNames,
        nullptr);

    ColumnsInfo->Validate();
    Validate();
}

bool TDataMetaInfo::operator==(const TDataMetaInfo& rhs) const {
    if (FeaturesLayout) {
        if (rhs.FeaturesLayout) {
            if (!(*FeaturesLayout == *rhs.FeaturesLayout)) {
                return false;
            }
        } else {
            return false;
        }
    } else if (rhs.FeaturesLayout) {
        return false;
    }

    return std::tie(
        HasTarget,
        BaselineCount,
        HasGroupId,
        HasGroupWeight,
        HasSubgroupIds,
        HasWeights,
        HasTimestamp,
        HasPairs,
        ColumnsInfo
    ) == std::tie(
        rhs.HasTarget,
        rhs.BaselineCount,
        rhs.HasGroupId,
        rhs.HasGroupWeight,
        rhs.HasSubgroupIds,
        rhs.HasWeights,
        rhs.HasTimestamp,
        rhs.HasPairs,
        rhs.ColumnsInfo
    );
}

void TDataMetaInfo::Validate() const {
    CB_ENSURE(GetFeatureCount() > 0, "Pool should have at least one factor");
    CB_ENSURE(!HasGroupWeight || (HasGroupWeight && HasGroupId),
        "You should provide GroupId when providing GroupWeight.");
}

TVector<TString> TDataColumnsMetaInfo::GenerateFeatureIds(const TMaybe<TVector<TString>>& header) const {
    TVector<TString> featureIds;
    // TODO: this convoluted logic is for compatibility
    if (!AllOf(Columns.begin(), Columns.end(), [](const TColumn& column) { return column.Id.empty(); })) {
        for (auto column : Columns) {
            if (column.Type == EColumn::Categ || column.Type == EColumn::Num) {
                featureIds.push_back(column.Id);
            }
        }
    } else if (header.Defined()) {
        for (auto i : xrange(header->size())) {
            if (Columns[i].Type == EColumn::Categ || Columns[i].Type == EColumn::Num) {
                featureIds.push_back((*header)[i]);
            }
        }
    }
    return featureIds;
}

void NCB::AddWithShared(IBinSaver* binSaver, TDataMetaInfo* data) {
    AddWithShared(binSaver, &(data->FeaturesLayout));
    binSaver->AddMulti(
        data->HasTarget,
        data->BaselineCount,
        data->HasGroupId,
        data->HasGroupWeight,
        data->HasSubgroupIds,
        data->HasWeights,
        data->HasTimestamp,
        data->HasPairs,
        data->ColumnsInfo
    );
}
