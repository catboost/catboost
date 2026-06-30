#pragma once

#include "column.h"

#include <catboost/private/libs/data_util/path_with_scheme.h>

#include <util/stream/fwd.h>
#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>

struct TCdParserDefaults {
    bool UseDefaultColumnCount = false;
    TMaybe<EColumn> DefaultColumnType = Nothing();
    int ColumnCount;

public:
    TCdParserDefaults() = default;
    TCdParserDefaults(TMaybe<EColumn> defaultColumnType)
        : UseDefaultColumnCount(false)
        , DefaultColumnType(defaultColumnType)
    {}
    TCdParserDefaults(TMaybe<EColumn> defaultColumnType, int columnCount)
        : UseDefaultColumnCount(true)
        , DefaultColumnType(defaultColumnType)
        , ColumnCount(columnCount)
    {}
};

class ICdProvider {
public:
    virtual ~ICdProvider() = 0;

    // if columnsCount is undefined - return all known columns
    virtual TVector<TColumn> GetColumnsDescription(TMaybe<ui32> columnsCount) const = 0;
    virtual bool Inited() const = 0;
};

THolder<ICdProvider> MakeCdProviderFromArray(const TVector<TColumn>& columnsDescription);

THolder<ICdProvider> MakeCdProviderFromFile(const NCB::TPathWithScheme& path);

// Returns vector of columnsCount columns, where i-th element describes
// i-th column.
TVector<TColumn> ReadCD(const NCB::TPathWithScheme& path, const TCdParserDefaults& defaults = {});
TVector<TColumn> ReadCD(IInputStream* in, const TCdParserDefaults& defaults = {});
TVector<TColumn> MakeDefaultColumnsDescription(size_t columnsCount);
TVector<EColumn> GetColumnTypes(TConstArrayRef<TColumn> descriptions);
