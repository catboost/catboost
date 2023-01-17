#include "pipe.h"

#include <util/generic/yexception.h>

ssize_t TPipeHandle::Read(void* buffer, size_t byteCount) const noexcept {
#ifdef _win_
    return recv(Fd_, (char*)buffer, byteCount, 0);
#else
    return read(Fd_, buffer, byteCount);
#endif
}

ssize_t TPipeHandle::Write(const void* buffer, size_t byteCount) const noexcept {
#ifdef _win_
    return send(Fd_, (const char*)buffer, byteCount, 0);
#else
    return write(Fd_, buffer, byteCount);
#endif
}

bool TPipeHandle::Close() noexcept {
    bool ok = true;
    if (Fd_ != INVALID_PIPEHANDLE) {
#ifdef _win_
        ok = closesocket(Fd_) == 0;
#else
        ok = close(Fd_) == 0;
#endif
    }
    Fd_ = INVALID_PIPEHANDLE;
    return ok;
}

void TPipeHandle::Pipe(TPipeHandle& reader, TPipeHandle& writer, EOpenMode mode) {
    PIPEHANDLE fds[2];
#ifdef _win_
    int r = SocketPair(fds, false /* non-overlapped */, mode & CloseOnExec /* cloexec */);
#elif defined(_linux_)
    int r = pipe2(fds, mode & CloseOnExec ? O_CLOEXEC : 0);
#else
    int r = pipe(fds);
#endif
    if (r < 0) {
        ythrow TFileError() << "failed to create a pipe";
    }

#if !defined(_win_) && !defined(_linux_)
    // Non-atomic wrt exec
    if (mode & CloseOnExec) {
        for (int i = 0; i < 2; ++i) {
            int flags = fcntl(fds[i], F_GETFD, 0);
            if (flags < 0) {
                ythrow TFileError() << "failed to get flags";
            }
            int r = fcntl(fds[i], F_SETFD, flags | FD_CLOEXEC);
            if (r < 0) {
                ythrow TFileError() << "failed to set flags";
            }
        }
    }
#endif

    TPipeHandle(fds[0]).Swap(reader);
    TPipeHandle(fds[1]).Swap(writer);
}

class TPipe::TImpl: public TAtomicRefCount<TImpl> {
public:
    TImpl()
        : Handle_(INVALID_PIPEHANDLE)
    {
    }

    TImpl(PIPEHANDLE fd)
        : Handle_(fd)
    {
    }

    inline ~TImpl() {
        Close();
    }

    bool IsOpen() {
        return Handle_.IsOpen();
    }

    inline void Close() {
        if (!Handle_.IsOpen()) {
            return;
        }
        if (!Handle_.Close()) {
            ythrow TFileError() << "failed to close pipe";
        }
    }

    TPipeHandle& GetHandle() noexcept {
        return Handle_;
    }

    size_t Read(void* buffer, size_t count) const {
        ssize_t r = Handle_.Read(buffer, count);
        if (r < 0) {
            ythrow TFileError() << "failed to read from pipe";
        }
        return r;
    }

    size_t Write(const void* buffer, size_t count) const {
        ssize_t r = Handle_.Write(buffer, count);
        if (r < 0) {
            ythrow TFileError() << "failed to write to pipe";
        }
        return r;
    }

private:
    TPipeHandle Handle_;
};

TPipe::TPipe()
    : Impl_(new TImpl)
{
}

TPipe::TPipe(PIPEHANDLE fd)
    : Impl_(new TImpl(fd))
{
}

TPipe::~TPipe() = default;

void TPipe::Close() {
    Impl_->Close();
}

PIPEHANDLE TPipe::GetHandle() const noexcept {
    return Impl_->GetHandle();
}

bool TPipe::IsOpen() const noexcept {
    return Impl_->IsOpen();
}

size_t TPipe::Read(void* buf, size_t len) const {
    return Impl_->Read(buf, len);
}

size_t TPipe::Write(const void* buf, size_t len) const {
    return Impl_->Write(buf, len);
}

void TPipe::Pipe(TPipe& reader, TPipe& writer, EOpenMode mode) {
    TImplRef r(new TImpl());
    TImplRef w(new TImpl());

    TPipeHandle::Pipe(r->GetHandle(), w->GetHandle(), mode);

    r.Swap(reader.Impl_);
    w.Swap(writer.Impl_);
}
