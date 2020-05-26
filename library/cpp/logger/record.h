#pragma once

#include "priority.h"

#include <util/system/defaults.h>

struct TLogRecord {
    const char* Data;
    size_t Len;
    ELogPriority Priority;

    inline TLogRecord(ELogPriority priority, const char* data, size_t len) noexcept
        : Data(data)
        , Len(len)
        , Priority(priority)
    {
    }
};
