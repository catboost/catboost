#include "buffer.h"
#include <util/generic/buffer.h>

class TBufferOutput::TImpl {
public:
    inline TImpl(TBuffer& buf)
        : Data_(buf)
    {
    }

    virtual ~TImpl() = default;

    inline void DoWrite(const void* buf, size_t len) {
        Data_.Append((const char*)buf, len);
    }

    inline TBuffer& Buffer() const noexcept {
        return Data_;
    }

private:
    TBuffer& Data_;
};

namespace {
    using TImpl = TBufferOutput::TImpl;

    class TOwnedImpl: private TBuffer, public TImpl {
    public:
        inline TOwnedImpl(size_t buflen)
            : TBuffer(buflen)
            , TImpl(static_cast<TBuffer&>(*this))
        {
        }
    };
}

TBufferOutput::TBufferOutput(size_t buflen)
    : Impl_(new TOwnedImpl(buflen))
{
}

TBufferOutput::TBufferOutput(TBuffer& buffer)
    : Impl_(new TImpl(buffer))
{
}

TBufferOutput::TBufferOutput(TBufferOutput&&) noexcept = default;
TBufferOutput& TBufferOutput::operator=(TBufferOutput&&) noexcept = default;

TBufferOutput::~TBufferOutput() = default;

TBuffer& TBufferOutput::Buffer() const noexcept {
    return Impl_->Buffer();
}

void TBufferOutput::DoWrite(const void* buf, size_t len) {
    Impl_->DoWrite(buf, len);
}

TBufferInput::TBufferInput(const TBuffer& buffer)
    : Buf_(buffer)
    , Readed_(0)
{
}

TBufferInput::~TBufferInput() = default;

const TBuffer& TBufferInput::Buffer() const noexcept {
    return Buf_;
}

void TBufferInput::Rewind() noexcept {
    Readed_ = 0;
}

size_t TBufferInput::DoNext(const void** ptr, size_t len) {
    len = Min(Buf_.Size() - Readed_, len);
    *ptr = ~Buf_ + Readed_;
    Readed_ += len;
    return len;
}

void TBufferInput::DoUndo(size_t len) {
    Y_VERIFY(len <= Readed_);
    Readed_ -= len;
}
