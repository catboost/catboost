#pragma once

#include <library/binsaver/bin_saver.h>

#include <util/ysaveload.h>
#include <util/generic/string.h>

enum class EColumn {
    Num,
    Categ,
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
    Prediction
};

inline bool IsFactorColumn(EColumn column) {
    return column == EColumn::Num || column == EColumn::Categ || column == EColumn::Sparse;
}

TStringBuf ToCanonicalColumnName(TStringBuf columnName);


struct TColumn {
    EColumn Type;
    TString Id;

public:
    bool operator==(const TColumn& rhs) const {
        return (Type == rhs.Type) && (Id == rhs.Id);
    }

    Y_SAVELOAD_DEFINE(Type, Id);
    SAVELOAD(Type, Id)
};
