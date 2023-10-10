#include "buffer.h"
#include <util/generic/buffer.h>

class TBufferOutput::TImpl {
public:
    inline TImpl(TBuffer& buf)
        : Data_(buf)
    {
    }

    virtual ~TImpl() = default;

    inline size_t DoNext(void** ptr) {
        if (Data_.Avail() == 0) {
            Data_.Reserve(FastClp2(Data_.Capacity() + MinBufferGrowSize));
        }
        size_t previousSize = Data_.size();
        Data_.Resize(Data_.Capacity());
        *ptr = Data_.Begin() + previousSize;
        return Data_.Size() - previousSize;
    }

    inline void DoUndo(size_t len) {
        Y_ABORT_UNLESS(len <= Data_.Size(), "trying to undo more bytes than actually written");
        Data_.Resize(Data_.size() - len);
    }

    inline void DoWrite(const void* buf, size_t len) {
        Data_.Append((const char*)buf, len);
    }

    inline void DoWriteC(char c) {
        Data_.Append(c);
    }

    inline TBuffer& Buffer() const noexcept {
        return Data_;
    }

private:
    TBuffer& Data_;
    static constexpr size_t MinBufferGrowSize = 16;
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

size_t TBufferOutput::DoNext(void** ptr) {
    return Impl_->DoNext(ptr);
}

void TBufferOutput::DoUndo(size_t len) {
    Impl_->DoUndo(len);
}

void TBufferOutput::DoWrite(const void* buf, size_t len) {
    Impl_->DoWrite(buf, len);
}

void TBufferOutput::DoWriteC(char c) {
    Impl_->DoWriteC(c);
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
    *ptr = Buf_.data() + Readed_;
    Readed_ += len;
    return len;
}

void TBufferInput::DoUndo(size_t len) {
    Y_ABORT_UNLESS(len <= Readed_);
    Readed_ -= len;
}
