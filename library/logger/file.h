#pragma once

#include "backend.h"

#include <util/generic/ptr.h>

class TString;

class TFileLogBackend: public TLogBackend {
public:
    TFileLogBackend(const TString& path);
    ~TFileLogBackend() override;

    void WriteData(const TLogRecord& rec) override;
    void ReopenLog() override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};
