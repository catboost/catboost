#pragma once

#include "backend.h"

#include <util/generic/ptr.h>

class TThreadedLogBackend: public TLogBackend {
public:
    TThreadedLogBackend(TLogBackend* slave);
    TThreadedLogBackend(TLogBackend* slave, size_t queuelen);
    ~TThreadedLogBackend() override;

    void WriteData(const TLogRecord& rec) override;
    void ReopenLog() override;

    // Write an emergency message when the memory allocator is corrupted.
    // The TThreadedLogBackend object can't be used after this method is called.
    void WriteEmergencyData(const TLogRecord& rec);

private:
    class TImpl;
    THolder<TImpl> Impl_;
};
