#include "mem.h"

#include <util/generic/yexception.h>

TMemoryInput::TMemoryInput() noexcept
    : Buf_(nullptr)
    , Len_(0)
{
}

TMemoryInput::TMemoryInput(const void* buf, size_t len) noexcept
    : Buf_((const char*)buf)
    , Len_(len)
{
}

TMemoryInput::TMemoryInput(const TStringBuf buf) noexcept
    : Buf_(buf.data())
    , Len_(buf.size())
{
}

TMemoryInput::~TMemoryInput() = default;

size_t TMemoryInput::DoNext(const void** ptr, size_t len) {
    len = Min(Len_, len);

    *ptr = Buf_;
    Len_ -= len;
    Buf_ += len;
    return len;
}

void TMemoryInput::DoUndo(size_t len) {
    Len_ += len;
    Buf_ -= len;
}

TMemoryOutput::~TMemoryOutput() = default;

size_t TMemoryOutput::DoNext(void** ptr) {
    Y_ENSURE(Buf_ < End_, TStringBuf("memory output stream exhausted"));
    *ptr = Buf_;
    size_t bufferSize = End_ - Buf_;
    Buf_ = End_;

    return bufferSize;
}

void TMemoryOutput::DoUndo(size_t len) {
    Buf_ -= len;
}

void TMemoryOutput::DoWrite(const void* buf, size_t len) {
    char* end = Buf_ + len;
    Y_ENSURE(end <= End_, TStringBuf("memory output stream exhausted"));

    memcpy(Buf_, buf, len);
    Buf_ = end;
}

void TMemoryOutput::DoWriteC(char c) {
    Y_ENSURE(Buf_ < End_, TStringBuf("memory output stream exhausted"));
    *Buf_++ = c;
}
