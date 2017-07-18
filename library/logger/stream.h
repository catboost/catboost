#pragma once

#include "backend.h"

class TOutputStream;

class TStreamLogBackend: public TLogBackend {
public:
    TStreamLogBackend(TOutputStream* slave);
    ~TStreamLogBackend() override;

    void WriteData(const TLogRecord& rec) override;
    void ReopenLog() override;

private:
    TOutputStream* Slave_;
};
