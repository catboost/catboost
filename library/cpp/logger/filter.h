#pragma once

#include "priority.h"
#include "record.h"
#include "backend.h"
#include <util/generic/ptr.h>

template <class TBaseBackend>
class TFilteredLogBackend: public TLogBackend {
    THolder<TBaseBackend> Backend;
    ELogPriority Level;

public:
    TFilteredLogBackend(TBaseBackend* t, ELogPriority level = LOG_MAX_PRIORITY) noexcept
        : Backend(t)
        , Level(level)
    {
    }

    ~TFilteredLogBackend() override {
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
