#pragma once

#include <util/ysaveload.h>
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
    GroupWeight,
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

    Y_SAVELOAD_DEFINE(Type, Id);
};
