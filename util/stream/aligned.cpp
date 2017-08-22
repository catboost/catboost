#include "aligned.h"

size_t TAlignedInput::DoRead(void* ptr, size_t len) {
    size_t ret = Stream_->Read(ptr, len);
    Position_ += ret;
    return ret;
}

size_t TAlignedInput::DoSkip(size_t len) {
    size_t ret = Stream_->Skip(len);
    Position_ += ret;
    return ret;
}

size_t TAlignedInput::DoReadTo(TString& st, char ch) {
    size_t ret = Stream_->ReadTo(st, ch);
    Position_ += ret;
    return ret;
}

ui64 TAlignedInput::DoReadAll(IOutputStream& out) {
    ui64 ret = Stream_->ReadAll(out);
    Position_ += ret;
    return ret;
}

void TAlignedOutput::DoWrite(const void* ptr, size_t len) {
    Stream_->Write(ptr, len);
    Position_ += len;
}
