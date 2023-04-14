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

TStreamWithContextLogBackend::TStreamWithContextLogBackend(IOutputStream* slave)
    : Slave_(slave)
{
}

TStreamWithContextLogBackend::~TStreamWithContextLogBackend() {
}

void TStreamWithContextLogBackend::WriteData(const TLogRecord& rec) {
    Slave_->Write(rec.Data, rec.Len);
    Slave_->Write(DELIMITER);
    for (const auto& [key, value] : rec.MetaFlags) {
        Slave_->Write(TString::Join(key, "=", value));
        Slave_->Write(DELIMITER);
    }
}

void TStreamWithContextLogBackend::ReopenLog() {
}
