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
    : Buf_(~buf)
    , Len_(+buf)
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

void TMemoryOutput::DoWrite(const void* buf, size_t len) {
    char* end = Buf_ + len;
    Y_ENSURE(end <= End_, STRINGBUF("memory output stream exhausted"));

    memcpy(Buf_, buf, len);
    Buf_ = end;
}
