#include "pool_builder.h"

#include <catboost/libs/helpers/exception.h>

#include <util/string/split.h>

ui32 TPoolColumnsMetaInfo::CountColumns(const EColumn columnType) const {
    return CountIf(
        Columns.begin(),
        Columns.end(),
        [&columnType](const auto x) -> bool {
            return x.Type == columnType;
        }
    );
}

TVector<int> TPoolColumnsMetaInfo::GetCategFeatures() const {
    Y_ASSERT(!Columns.empty());
    TVector<int> categFeatures;
    int featureId = 0;
    for (const TColumn& column : Columns) {
        switch (column.Type) {
            case EColumn::Categ:
                categFeatures.push_back(featureId);
                ++featureId;
                break;
            case EColumn::Num:
                ++featureId;
                break;
            case EColumn::Auxiliary:
            case EColumn::Label:
            case EColumn::Baseline:
            case EColumn::Weight:
            case EColumn::DocId:
            case EColumn::GroupId:
            case EColumn::GroupWeight:
            case EColumn::SubgroupId:
            case EColumn::Timestamp:
                break;
            default:
                CB_ENSURE(false, "this column type is not supported");
        }
    }
    return categFeatures;
}

void TPoolColumnsMetaInfo::Validate() const {
    CB_ENSURE(CountColumns(EColumn::Weight) <= 1, "Too many Weight columns.");
    CB_ENSURE(CountColumns(EColumn::Label) <= 1, "Too many Label columns.");
    CB_ENSURE(CountColumns(EColumn::DocId) <= 1, "Too many DocId columns.");
    CB_ENSURE(CountColumns(EColumn::GroupId) <= 1, "Too many GroupId columns. Maybe you've specified QueryId and GroupId, QueryId is synonym for GroupId.");
    CB_ENSURE(CountColumns(EColumn::GroupWeight) <= 1, "Too many GroupWeight columns.");
    CB_ENSURE(CountColumns(EColumn::SubgroupId) <= 1, "Too many SubgroupId columns.");
    CB_ENSURE(CountColumns(EColumn::Timestamp) <= 1, "Too many Timestamp columns.");
}

TPoolMetaInfo::TPoolMetaInfo(TVector<TColumn>&& columns, bool hasAdditionalGroupWeight)
    : ColumnsInfo(TPoolColumnsMetaInfo{std::move(columns)})
{
    BaselineCount = ColumnsInfo->CountColumns(EColumn::Baseline);
    HasWeights = ColumnsInfo->CountColumns(EColumn::Weight) != 0;
    HasDocIds = ColumnsInfo->CountColumns(EColumn::DocId) != 0;
    HasGroupId = ColumnsInfo->CountColumns(EColumn::GroupId) != 0;
    HasGroupWeight = ColumnsInfo->CountColumns(EColumn::GroupWeight) != 0 || hasAdditionalGroupWeight;
    HasSubgroupIds = ColumnsInfo->CountColumns(EColumn::SubgroupId) != 0;
    HasTimestamp = ColumnsInfo->CountColumns(EColumn::Timestamp) != 0;
    FeatureCount = (const ui32)CountIf(
        ColumnsInfo->Columns.begin(),
        ColumnsInfo->Columns.end(),
        [](const auto x) -> bool {
            return IsFactorColumn(x.Type);
        }
    );
    ColumnsInfo->Validate();
    Validate();
}

void TPoolMetaInfo::Validate() const {
    CB_ENSURE(FeatureCount > 0, "Pool should have at least one factor");
    CB_ENSURE(!(HasWeights && HasGroupWeight), "Pool must have either Weight column or GroupWeight column");
    CB_ENSURE(!HasGroupWeight || (HasGroupWeight && HasGroupId),
        "You should provide GroupId when providing GroupWeight.");
}

TVector<TString> TPoolColumnsMetaInfo::GenerateFeatureIds(const TMaybe<TString>& header, char fieldDelimiter) const {
    TVector<TString> featureIds;
    // TODO: this convoluted logic is for compatibility
    if (!AllOf(Columns.begin(), Columns.end(), [](const TColumn& column) { return column.Id.empty(); })) {
        for (auto column : Columns) {
            if (column.Type == EColumn::Categ || column.Type == EColumn::Num) {
                featureIds.push_back(column.Id);
            }
        }
    } else if (header.Defined()) {
        TVector<TStringBuf> words;
        SplitRangeTo<const char, TVector<TStringBuf>>(~(*header), ~(*header) + header->size(), fieldDelimiter, &words);
        for (int i = 0; i < words.ysize(); ++i) {
            if (Columns[i].Type == EColumn::Categ || Columns[i].Type == EColumn::Num) {
                featureIds.push_back(ToString(words[i]));
            }
        }
    }
    return featureIds;
}
