#include "stream.h"
#include "record.h"

#include <util/stream/output.h>

TStreamLogBackend::TStreamLogBackend(IOutputStream* slave)
    : Slave_(slave)
{
}

TStreamLogBackend::~TStreamLogBackend() {
}

void TStreamLogBackend::WriteData(const TLogRecord& rec) {
    Slave_->Write(rec.Data, rec.Len);
}

void TStreamLogBackend::ReopenLog() {
}
