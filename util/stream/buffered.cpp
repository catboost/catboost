#include "mem.h"
#include "buffered.h"

#include <util/memory/addstorage.h>
#include <util/generic/yexception.h>
#include <util/generic/buffer.h>

class TBufferedInput::TImpl: public TAdditionalStorage<TImpl> {
public:
    inline TImpl(IInputStream* slave)
        : Slave_(slave)
        , MemInput_(nullptr, 0)
    {
    }

    inline ~TImpl() = default;

    inline size_t Next(const void** ptr, size_t len) {
        if (MemInput_.Exhausted()) {
            MemInput_.Reset(Buf(), Slave_->Read(Buf(), BufLen()));
        }

        return MemInput_.Next(ptr, len);
    }

    inline size_t Read(void* buf, size_t len) {
        if (MemInput_.Exhausted()) {
            if (len > BufLen() / 2) {
                return Slave_->Read(buf, len);
            }

            MemInput_.Reset(Buf(), Slave_->Read(Buf(), BufLen()));
        }

        return MemInput_.Read(buf, len);
    }

    inline size_t Skip(size_t len) {
        size_t totalSkipped = 0;
        while (len) {
            const size_t skipped = DoSkip(len);
            if (skipped == 0) {
                break;
            }

            totalSkipped += skipped;
            len -= skipped;
        }

        return totalSkipped;
    }

    inline size_t DoSkip(size_t len) {
        if (MemInput_.Exhausted()) {
            if (len > BufLen() / 2) {
                return Slave_->Skip(len);
            }

            MemInput_.Reset(Buf(), Slave_->Read(Buf(), BufLen()));
        }

        return MemInput_.Skip(len);
    }

    inline size_t ReadTo(TString& st, char to) {
        st.clear();

        TString s_tmp;

        size_t ret = 0;

        while (true) {
            if (MemInput_.Exhausted()) {
                const size_t bytesRead = Slave_->Read(Buf(), BufLen());

                if (!bytesRead) {
                    break;
                }

                MemInput_.Reset(Buf(), bytesRead);
            }

            const size_t a_len(MemInput_.Avail());
            size_t s_len = 0;
            if (st.empty()) {
                ret += MemInput_.ReadTo(st, to);
                s_len = st.length();
            } else {
                ret += MemInput_.ReadTo(s_tmp, to);
                s_len = s_tmp.length();
                st.append(s_tmp);
            }

            if (s_len != a_len) {
                break;
            }
        }

        return ret;
    }

    inline void Reset(IInputStream* slave) {
        Slave_ = slave;
    }

private:
    inline size_t BufLen() const noexcept {
        return AdditionalDataLength();
    }

    inline void* Buf() const noexcept {
        return AdditionalData();
    }

private:
    IInputStream* Slave_;
    TMemoryInput MemInput_;
};

TBufferedInput::TBufferedInput(IInputStream* slave, size_t buflen)
    : Impl_(new (buflen) TImpl(slave))
{
}

TBufferedInput::TBufferedInput(TBufferedInput&&) noexcept = default;
TBufferedInput& TBufferedInput::operator=(TBufferedInput&&) noexcept = default;

TBufferedInput::~TBufferedInput() = default;

size_t TBufferedInput::DoRead(void* buf, size_t len) {
    return Impl_->Read(buf, len);
}

size_t TBufferedInput::DoSkip(size_t len) {
    return Impl_->Skip(len);
}

size_t TBufferedInput::DoNext(const void** ptr, size_t len) {
    return Impl_->Next(ptr, len);
}

size_t TBufferedInput::DoReadTo(TString& st, char ch) {
    return Impl_->ReadTo(st, ch);
}

void TBufferedInput::Reset(IInputStream* slave) {
    Impl_->Reset(slave);
}

class TBufferedOutputBase::TImpl {
public:
    inline TImpl(IOutputStream* slave)
        : Slave_(slave)
        , MemOut_(nullptr, 0)
        , PropagateFlush_(false)
        , PropagateFinish_(false)
    {
    }

    virtual ~TImpl() = default;

    inline void Reset() {
        MemOut_.Reset(Buf(), Len());
    }

    inline size_t Next(void** ptr) {
        if (MemOut_.Avail() == 0) {
            Slave_->Write(Buf(), Stored());
            OnBufferExhausted();
            Reset();
        }

        return MemOut_.Next(ptr);
    }

    inline void Undo(size_t len) {
        Y_ABORT_UNLESS(len <= Stored(), "trying to undo more bytes than actually written");
        MemOut_.Undo(len);
    }

    inline void Write(const void* buf, size_t len) {
        if (len <= MemOut_.Avail()) {
            /*
             * fast path
             */

            MemOut_.Write(buf, len);
        } else {
            const size_t stored = Stored();
            const size_t full_len = stored + len;
            const size_t good_len = DownToBufferGranularity(full_len);
            const size_t write_from_buf = good_len - stored;

            using TPart = IOutputStream::TPart;

            alignas(TPart) char data[2 * sizeof(TPart)];
            TPart* parts = reinterpret_cast<TPart*>(data);
            TPart* end = parts;

            if (stored) {
                new (end++) TPart(Buf(), stored);
            }

            if (write_from_buf) {
                new (end++) TPart(buf, write_from_buf);
            }

            Slave_->Write(parts, end - parts);

            // grow buffer only on full flushes
            OnBufferExhausted();
            Reset();

            if (write_from_buf < len) {
                MemOut_.Write((const char*)buf + write_from_buf, len - write_from_buf);
            }
        }
    }

    inline void Write(char c) {
        if (Y_UNLIKELY(MemOut_.Avail() == 0)) {
            Slave_->Write(Buf(), Stored());
            OnBufferExhausted();
            Reset();
        }

        MemOut_.Write(c);
    }

    inline void SetFlushPropagateMode(bool mode) noexcept {
        PropagateFlush_ = mode;
    }

    inline void SetFinishPropagateMode(bool mode) noexcept {
        PropagateFinish_ = mode;
    }

    inline void Flush() {
        {
            Slave_->Write(Buf(), Stored());
            Reset();
        }

        if (PropagateFlush_) {
            Slave_->Flush();
        }
    }

    inline void Finish() {
        try {
            Flush();
        } catch (...) {
            try {
                DoFinish();
            } catch (...) {
                // ¯\_(ツ)_/¯
            }

            throw;
        }

        DoFinish();
    }

private:
    inline void DoFinish() {
        if (PropagateFinish_) {
            Slave_->Finish();
        }
    }

    inline size_t Stored() const noexcept {
        return Len() - MemOut_.Avail();
    }

    inline size_t DownToBufferGranularity(size_t l) const noexcept {
        return l - (l % Len());
    }

    virtual void OnBufferExhausted() = 0;
    virtual void* Buf() const noexcept = 0;
    virtual size_t Len() const noexcept = 0;

private:
    IOutputStream* Slave_;
    TMemoryOutput MemOut_;
    bool PropagateFlush_;
    bool PropagateFinish_;
};

namespace {
    struct TSimpleImpl: public TBufferedOutputBase::TImpl, public TAdditionalStorage<TSimpleImpl> {
        inline TSimpleImpl(IOutputStream* slave)
            : TBufferedOutputBase::TImpl(slave)
        {
            Reset();
        }

        ~TSimpleImpl() override = default;

        void OnBufferExhausted() final {
        }

        void* Buf() const noexcept override {
            return AdditionalData();
        }

        size_t Len() const noexcept override {
            return AdditionalDataLength();
        }
    };

    struct TAdaptiveImpl: public TBufferedOutputBase::TImpl {
        enum {
            Step = 4096
        };

        inline TAdaptiveImpl(IOutputStream* slave)
            : TBufferedOutputBase::TImpl(slave)
            , N_(0)
        {
            B_.Reserve(Step);
            Reset();
        }

        ~TAdaptiveImpl() override = default;

        void OnBufferExhausted() final {
            const size_t c = ((size_t)Step) << Min<size_t>(++N_ / 32, 10);

            if (c > B_.Capacity()) {
                TBuffer(c).Swap(B_);
            }
        }

        void* Buf() const noexcept override {
            return (void*)B_.Data();
        }

        size_t Len() const noexcept override {
            return B_.Capacity();
        }

        TBuffer B_;
        ui64 N_;
    };
} // namespace

TBufferedOutputBase::TBufferedOutputBase(IOutputStream* slave)
    : Impl_(new TAdaptiveImpl(slave))
{
}

TBufferedOutputBase::TBufferedOutputBase(IOutputStream* slave, size_t buflen)
    : Impl_(new (buflen) TSimpleImpl(slave))
{
}

TBufferedOutputBase::TBufferedOutputBase(TBufferedOutputBase&&) noexcept = default;
TBufferedOutputBase& TBufferedOutputBase::operator=(TBufferedOutputBase&&) noexcept = default;

TBufferedOutputBase::~TBufferedOutputBase() {
    try {
        Finish();
    } catch (...) {
        // ¯\_(ツ)_/¯
    }
}

size_t TBufferedOutputBase::DoNext(void** ptr) {
    Y_ENSURE(Impl_.Get(), "cannot call next in finished stream");
    return Impl_->Next(ptr);
}

void TBufferedOutputBase::DoUndo(size_t len) {
    Y_ENSURE(Impl_.Get(), "cannot call undo in finished stream");
    Impl_->Undo(len);
}

void TBufferedOutputBase::DoWrite(const void* data, size_t len) {
    Y_ENSURE(Impl_.Get(), "cannot write to finished stream");
    Impl_->Write(data, len);
}

void TBufferedOutputBase::DoWriteC(char c) {
    Y_ENSURE(Impl_.Get(), "cannot write to finished stream");
    Impl_->Write(c);
}

void TBufferedOutputBase::DoFlush() {
    if (Impl_.Get()) {
        Impl_->Flush();
    }
}

void TBufferedOutputBase::DoFinish() {
    THolder<TImpl> impl(Impl_.Release());

    if (impl) {
        impl->Finish();
    }
}

void TBufferedOutputBase::SetFlushPropagateMode(bool propagate) noexcept {
    if (Impl_.Get()) {
        Impl_->SetFlushPropagateMode(propagate);
    }
}

void TBufferedOutputBase::SetFinishPropagateMode(bool propagate) noexcept {
    if (Impl_.Get()) {
        Impl_->SetFinishPropagateMode(propagate);
    }
}

TBufferedOutput::TBufferedOutput(IOutputStream* slave, size_t buflen)
    : TBufferedOutputBase(slave, buflen)
{
}

TBufferedOutput::~TBufferedOutput() = default;

TAdaptiveBufferedOutput::TAdaptiveBufferedOutput(IOutputStream* slave)
    : TBufferedOutputBase(slave)
{
}

TAdaptiveBufferedOutput::~TAdaptiveBufferedOutput() = default;
