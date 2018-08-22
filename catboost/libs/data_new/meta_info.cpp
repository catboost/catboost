#include "meta_info.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/xrange.h>
#include <util/string/split.h>


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
    TVector<TColumn>&& columns,
    bool hasAdditionalGroupWeight,
    bool hasPairs,
    const TMaybe<TVector<TString>>& header
)
    : ColumnsInfo(TDataColumnsMetaInfo{std::move(columns)})
{
    BaselineCount = ColumnsInfo->CountColumns(EColumn::Baseline);
    HasWeights = ColumnsInfo->CountColumns(EColumn::Weight) != 0;
    HasDocIds = ColumnsInfo->CountColumns(EColumn::DocId) != 0;
    HasGroupId = ColumnsInfo->CountColumns(EColumn::GroupId) != 0;
    HasGroupWeight = ColumnsInfo->CountColumns(EColumn::GroupWeight) != 0 || hasAdditionalGroupWeight;
    HasSubgroupIds = ColumnsInfo->CountColumns(EColumn::SubgroupId) != 0;
    HasTimestamp = ColumnsInfo->CountColumns(EColumn::Timestamp) != 0;
    HasPairs = hasPairs;

    TVector<int> catFeatureIndices;

    int featureIdx = 0;
    for (const auto& column : ColumnsInfo->Columns) {
        if (IsFactorColumn(column.Type)) {
            if (column.Type == EColumn::Categ) {
                catFeatureIndices.push_back(featureIdx);
            }
            ++featureIdx;
        }
    }

    FeaturesLayout = TFeaturesLayout(
        featureIdx,
        std::move(catFeatureIndices),
        ColumnsInfo->GenerateFeatureIds(header)
    );

    ColumnsInfo->Validate();
    Validate();
}

void TDataMetaInfo::Validate() const {
    CB_ENSURE(GetFeatureCount() > 0, "Pool should have at least one factor");
    CB_ENSURE(!(HasWeights && HasGroupWeight), "Pool must have either Weight column or GroupWeight column");
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
