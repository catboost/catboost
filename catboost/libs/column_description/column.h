#pragma once


#include <util/generic/string.h>

enum class EColumn {
    Num,
    Categ,
    Label,
    Auxiliary,
    Baseline,
    Weight,
    DocId,
    GroupId,
    SubgroupId,
    Timestamp,
    Sparse,
    Prediction
};

inline bool IsFactorColumn(EColumn column) {
    return column == EColumn::Num || column == EColumn::Categ || column == EColumn::Sparse;
}

struct TColumn {
    EColumn Type;
    TString Id;
};
