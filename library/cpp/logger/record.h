#pragma once

#include "priority.h"

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/system/defaults.h>

#include <utility>

struct TLogRecord {
    using TMetaFlags = TVector<std::pair<TString, TString>>;

    const char* Data;
    size_t Len;
    ELogPriority Priority;
    TMetaFlags MetaFlags;

    inline TLogRecord(ELogPriority priority, const char* data, size_t len, TMetaFlags metaFlags = {}) noexcept
        : Data(data)
        , Len(len)
        , Priority(priority)
        , MetaFlags(std::move(metaFlags))
    {
    }
};
