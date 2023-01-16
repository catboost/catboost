#pragma once

#include "backend.h"
#include <util/generic/vector.h>

class TCompositeLogBackend: public TLogBackend {
public:
    virtual void WriteData(const TLogRecord& rec) override;
    virtual void ReopenLog() override;
    virtual void AddLogBackend(THolder<TLogBackend>&& backend);

private:
    TVector<THolder<TLogBackend>> Slaves;
};
