#pragma once

#include "priority.h"
#include "record.h"
#include "backend.h"
#include <util/generic/ptr.h>

template <class TBaseBackend>
class TFilteredLogBackend: public TLogBackend {
    THolder<TBaseBackend> Backend;
    TLogPriority Level;

public:
    TFilteredLogBackend(TBaseBackend* t, TLogPriority level = LOG_MAX_PRIORITY) noexcept
        : Backend(t)
        , Level(level)
    {
    }

    ~TFilteredLogBackend() override {
    }

    TLogPriority FiltrationLevel() const override {
        return Level;
    }

    void ReopenLog() override {
        Backend->ReopenLog();
    }

    void WriteData(const TLogRecord& rec) override {
        if (rec.Priority <= (TLogPriority)Level) {
            Backend->WriteData(rec);
        }
    }
};
