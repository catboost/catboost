#pragma once

#include "backend.h"

class TNullLogBackend: public TLogBackend {
public:
    TNullLogBackend();
    ~TNullLogBackend() override;

    void WriteData(const TLogRecord& rec) override;
    void ReopenLog() override;
};
