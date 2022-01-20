#pragma once

#include "priority.h"
#include "record.h"
#include "backend.h"
#include <util/generic/ptr.h>

class TFilteredLogBackend: public TLogBackend {
    THolder<TLogBackend> Backend;
    ELogPriority Level;

public:
    TFilteredLogBackend(THolder<TLogBackend>&& t, ELogPriority level = LOG_MAX_PRIORITY) noexcept
        : Backend(std::move(t))
        , Level(level)
    {
    }

    ELogPriority FiltrationLevel() const override {
        return Level;
    }

    void ReopenLog() override {
        Backend->ReopenLog();
    }

    void WriteData(const TLogRecord& rec) override {
        if (rec.Priority <= (ELogPriority)Level) {
            Backend->WriteData(rec);
        }
    }
};
