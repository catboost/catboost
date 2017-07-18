#pragma once

#include <util/generic/noncopyable.h>
#include "priority.h"

struct TLogRecord;

class TLogBackend: public TNonCopyable {
public:
    TLogBackend() noexcept;
    virtual ~TLogBackend();

    virtual void WriteData(const TLogRecord& rec) = 0;
    virtual void ReopenLog() = 0;
    virtual TLogPriority FiltrationLevel() const;

    static void ReopenAllBackends();
};
