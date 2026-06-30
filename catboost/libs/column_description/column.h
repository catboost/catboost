#pragma once

#include <library/cpp/binsaver/bin_saver.h>
#include <library/cpp/json/json_value.h>

#include <util/ysaveload.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

#include <tuple>


enum class EColumn {
    Num,
    Categ,
    HashedCateg,
    Label,
    Auxiliary,
    Baseline,
    Weight,
    SampleId,
    GroupId,
    GroupWeight,
    SubgroupId,
    Timestamp,
    Sparse,
    Prediction,
    Text,
    NumVector,
    Features
};

inline bool IsFactorColumn(EColumn column) {
    switch (column) {
        case EColumn::Num:
        case EColumn::Categ:
        case EColumn::HashedCateg:
        case EColumn::Sparse:
        case EColumn::Text:
        case EColumn::NumVector:
            return true;
        default:
            return false;
    }
}

inline bool CanBeOutputByColumnType(EColumn column) {
    return column == EColumn::Label ||
           column == EColumn::Baseline ||
           column == EColumn::Weight ||
           column == EColumn::SampleId ||
           column == EColumn::GroupId ||
           column == EColumn::GroupWeight ||
           column == EColumn::SubgroupId ||
           column == EColumn::Timestamp ||
           column == EColumn::Prediction;
}

TStringBuf ToCanonicalColumnName(TStringBuf columnName);

void ParseOutputColumnByIndex(const TString& outputColumn, ui32* columnNumber, TString* name);

struct TColumn {
    EColumn Type = EColumn::Num;
    TString Id = TString();
    TVector<TColumn> SubColumns; // only used for 'Features' column type now

public:
    TColumn(EColumn columnType = EColumn::Num, const TString& id = TString())
        : Type(columnType)
        , Id(id)
    {}

    bool operator==(const TColumn& rhs) const {
        return (Type == rhs.Type) && (Id == rhs.Id) && (SubColumns == rhs.SubColumns);
    }

    bool operator<(const TColumn& rhs) const {
        return std::tie(Type, Id, SubColumns) < std::tie(rhs.Type, rhs.Id, SubColumns);
    }

    Y_SAVELOAD_DEFINE(Type, Id, SubColumns);
    SAVELOAD(Type, Id, SubColumns);

    operator NJson::TJsonValue() const;
};
