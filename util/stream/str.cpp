#include "str.h"

TStringInput::~TStringInput() = default;

size_t TStringInput::DoNext(const void** ptr, size_t len) {
    len = Min(len, S_.size() - Pos_);
    *ptr = S_.data() + Pos_;
    Pos_ += len;
    return len;
}

void TStringInput::DoUndo(size_t len) {
    Y_VERIFY(len <= Pos_);
    Pos_ -= len;
}

TStringOutput::~TStringOutput() = default;

void TStringOutput::DoWrite(const void* buf, size_t len) {
    S_.append((const char*)buf, len);
}

TStringStream::~TStringStream() = default;
