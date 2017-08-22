#include "tee.h"

TTeeOutput::TTeeOutput(IOutputStream* l, IOutputStream* r) noexcept
    : L_(l)
    , R_(r)
{
}

TTeeOutput::~TTeeOutput() = default;

void TTeeOutput::DoWrite(const void* buf, size_t len) {
    L_->Write(buf, len);
    R_->Write(buf, len);
}

void TTeeOutput::DoFlush() {
    L_->Flush();
    R_->Flush();
}

void TTeeOutput::DoFinish() {
    L_->Finish();
    R_->Finish();
}
