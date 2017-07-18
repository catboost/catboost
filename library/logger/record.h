#pragma once

#include "priority.h"

#include <util/system/defaults.h>

struct TLogRecord {
    const char* Data;
    size_t Len;
    TLogPriority Priority;

    inline TLogRecord(TLogPriority priority, const char* data, size_t len) noexcept
        : Data(data)
        , Len(len)
        , Priority(priority)
    {
    }
};
