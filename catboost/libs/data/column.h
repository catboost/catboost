#pragma once


#include <util/generic/string.h>

enum class EColumn {
    Num,
    Categ,
    Target,
    Auxiliary,
    Baseline,
    Weight,
    DocId,
    QueryId
};

inline bool IsFactorColumn(EColumn column) {
    return column == EColumn::Num || column == EColumn::Categ;
}

struct TColumn {
    EColumn Type;
    TString Id;
};
