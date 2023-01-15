#pragma once

#include "backend.h"

class IOutputStream;

class TStreamLogBackend: public TLogBackend {
public:
    TStreamLogBackend(IOutputStream* slave);
    ~TStreamLogBackend() override;

    void WriteData(const TLogRecord& rec) override;
    void ReopenLog() override;

private:
    IOutputStream* Slave_;
};
