#include "buffer.h"
#include "mem_copy.h"
#include "string.h"
#include "ymath.h"

#include <util/system/sys_alloc.h>
#include <util/system/sanitizers.h>

TBuffer::TBuffer(size_t len)
    : Data_(nullptr)
    , Len_(0)
    , Pos_(0)
{
    Reserve(len);
}

TBuffer::TBuffer(TBuffer&& b) noexcept
    : Data_(nullptr)
    , Len_(0)
    , Pos_(0)
{
    Swap(b);
}

TBuffer::TBuffer(const char* buf, size_t len)
    : Data_(nullptr)
    , Len_(0)
    , Pos_(0)
{
    Append(buf, len);
}

TBuffer& TBuffer::operator=(TBuffer&& b) noexcept {
    y_deallocate(Data_);

    Data_ = b.Data_;
    Len_ = b.Len_;
    Pos_ = b.Pos_;

    b.Data_ = nullptr;
    b.Len_ = 0;
    b.Pos_ = 0;

    return *this;
}

void TBuffer::Append(const char* buf, size_t len) {
    if (len > Avail()) {
        Reserve(Pos_ + len);
    }

    Y_ASSERT(len <= Avail());

    MemCopy(Data() + Pos_, buf, len);
    NSan::Unpoison(Data() + Pos_, len);
    Pos_ += len;

    Y_ASSERT(Pos_ <= Len_);
}

void TBuffer::Fill(char ch, size_t len) {
    if (len > Avail()) {
        Reserve(Pos_ + len);
    }

    Y_ASSERT(len <= Avail());

    memset(Data() + Pos_, ch, len);
    NSan::Unpoison(Data() + Pos_, len);
    Pos_ += len;

    Y_ASSERT(Pos_ <= Len_);
}

void TBuffer::DoReserve(size_t realLen) {
    // FastClp2<T>(x) returns 0 on x from [Max<T>/2 + 2, Max<T>]
    const size_t len = Max<size_t>(FastClp2(realLen), realLen);

    Y_ASSERT(realLen > Len_);
    Y_ASSERT(len >= realLen);

    Realloc(len);
}

void TBuffer::Realloc(size_t len) {
    Y_ASSERT(Pos_ <= len);

    Data_ = (char*)y_reallocate(Data_, len);
    Len_ = len;
}

TBuffer::~TBuffer() {
    y_deallocate(Data_);
}

void TBuffer::AsString(TString& s) {
    s.assign(Data(), Size());
    Clear();
}
