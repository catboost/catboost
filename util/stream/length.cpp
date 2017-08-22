#include "length.h"

size_t TLengthLimitedInput::DoRead(void* buf, size_t len) {
    const size_t toRead = (size_t)Min<ui64>(Length_, len);
    const size_t ret = Slave_->Read(buf, toRead);

    Length_ -= ret;

    return ret;
}

size_t TLengthLimitedInput::DoSkip(size_t len) {
    len = (size_t)Min<ui64>((ui64)len, Length_);
    len = Slave_->Skip(len);
    Length_ -= len;
    return len;
}

size_t TCountingInput::DoRead(void* buf, size_t len) {
    const size_t ret = Slave_->Read(buf, len);
    Count_ += ret;
    return ret;
}

size_t TCountingInput::DoSkip(size_t len) {
    const size_t ret = Slave_->Skip(len);
    Count_ += ret;
    return ret;
}

size_t TCountingInput::DoReadTo(TString& st, char ch) {
    const size_t ret = Slave_->ReadTo(st, ch);
    Count_ += ret;
    return ret;
}

ui64 TCountingInput::DoReadAll(IOutputStream& out) {
    const ui64 ret = Slave_->ReadAll(out);
    Count_ += ret;
    return ret;
}

void TCountingOutput::DoWrite(const void* buf, size_t len) {
    Slave_->Write(buf, len);

    Count_ += len;
}
