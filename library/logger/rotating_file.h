#pragma once

#include "backend.h"

#include <util/generic/fwd.h>
#include <util/generic/ptr.h>

class TRotatingFileLogBackend: public TLogBackend {
public:
    TRotatingFileLogBackend(const TString& preRotatePath, const TString& postRotatePath, const ui64 maxSizeBytes);
    ~TRotatingFileLogBackend() override;

    void WriteData(const TLogRecord& rec) override;
    void ReopenLog() override;

private:
    class TImpl;
    TAtomicSharedPtr<TImpl> Impl_;
};
