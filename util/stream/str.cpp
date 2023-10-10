#include "str.h"

static constexpr size_t MIN_BUFFER_GROW_SIZE = 16;

TStringInput::~TStringInput() = default;

size_t TStringInput::DoNext(const void** ptr, size_t len) {
    len = Min(len, S_->size() - Pos_);
    *ptr = S_->data() + Pos_;
    Pos_ += len;
    return len;
}

void TStringInput::DoUndo(size_t len) {
    Y_ABORT_UNLESS(len <= Pos_);
    Pos_ -= len;
}

TStringOutput::~TStringOutput() = default;

size_t TStringOutput::DoNext(void** ptr) {
    if (S_->size() == S_->capacity()) {
        S_->reserve(FastClp2(S_->capacity() + MIN_BUFFER_GROW_SIZE));
    }
    size_t previousSize = S_->size();
    ResizeUninitialized(*S_, S_->capacity());
    *ptr = S_->begin() + previousSize;
    return S_->size() - previousSize;
}

void TStringOutput::DoUndo(size_t len) {
    Y_ABORT_UNLESS(len <= S_->size(), "trying to undo more bytes than actually written");
    S_->resize(S_->size() - len);
}

void TStringOutput::DoWrite(const void* buf, size_t len) {
    S_->append((const char*)buf, len);
}

void TStringOutput::DoWriteC(char c) {
    S_->push_back(c);
}

TStringStream::~TStringStream() = default;
