#pragma once

#include "column.h"

#include <util/stream/fwd.h>

struct TCdParserDefaults {
    bool UseDefaultType = false;
    EColumn DefaultColumnType;
    int ColumnCount;

    TCdParserDefaults() = default;
    TCdParserDefaults(EColumn defaultColumnType, int columnCount)
        : UseDefaultType(true)
        , DefaultColumnType(defaultColumnType)
        , ColumnCount(columnCount) {}
};

// Returns vector of columnsCount columns, where i-th element describes
// i-th column.
TVector<TColumn> ReadCD(IInputStream* in, const TCdParserDefaults& defaults = {});
TVector<TColumn> ReadCD(const TString& fileName, const TCdParserDefaults& defaults = {});
