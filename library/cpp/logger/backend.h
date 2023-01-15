#pragma once

#include "priority.h"

#include <util/generic/noncopyable.h>

#include <cstddef>

struct TLogRecord;

// NOTE: be aware that all `TLogBackend`s are registred in singleton.
class TLogBackend: public TNonCopyable {
public:
    TLogBackend() noexcept;
    virtual ~TLogBackend();

    virtual void WriteData(const TLogRecord& rec) = 0;
    virtual void ReopenLog() = 0;

    // Does not guarantee consistency with previous WriteData() calls:
    // log entries could be written to the new (reopened) log file due to
    // buffering effects.
    virtual void ReopenLogNoFlush();

    virtual ELogPriority FiltrationLevel() const;

    static void ReopenAllBackends(bool flush = true);

    virtual size_t QueueSize() const;
};
