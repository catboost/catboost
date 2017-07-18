#include "walk.h"

#include <util/generic/string.h>

void TWalkInput::DoUndo(size_t len) {
    Len_ += len;
    Buf_ = static_cast<const char*>(Buf_) - len;
}

size_t TWalkInput::DoNext(const void** ptr, size_t len) {
    if (!Len_) {
        Len_ = DoUnboundedNext(&Buf_);
    }

    len = Min(Len_, len);
    *ptr = Buf_;

    Buf_ = static_cast<const char*>(Buf_) + len;
    Len_ -= len;

    return len;
}
