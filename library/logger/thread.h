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
    void ReopenLogNoFlush() override;
    size_t QueueSize() const;

    // Write an emergency message when the memory allocator is corrupted.
    // The TThreadedLogBackend object can't be used after this method is called.
    void WriteEmergencyData(const TLogRecord& rec);

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

class TOwningThreadedLogBackend: private THolder<TLogBackend>, public TThreadedLogBackend {
public:
    TOwningThreadedLogBackend(TLogBackend* slave);
    TOwningThreadedLogBackend(TLogBackend* slave, size_t queuelen);
    ~TOwningThreadedLogBackend() override;
};
