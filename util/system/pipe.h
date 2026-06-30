#pragma once

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Warray-bounds" // need because of bug in gcc4.9.2
#endif

#include "defaults.h"
#include "file.h"
#include <util/generic/ptr.h>
#include <util/network/pair.h>
#include <util/generic/noncopyable.h>

using PIPEHANDLE = SOCKET;
#define INVALID_PIPEHANDLE INVALID_SOCKET

/// Pipe-like object: pipe on POSIX and socket on windows
class TPipeHandle: public TNonCopyable {
public:
    inline TPipeHandle() noexcept
        : Fd_(INVALID_PIPEHANDLE)
    {
    }

    inline TPipeHandle(PIPEHANDLE fd) noexcept
        : Fd_(fd)
    {
    }

    inline ~TPipeHandle() {
        Close();
    }

    bool Close() noexcept;

    inline PIPEHANDLE Release() noexcept {
        PIPEHANDLE ret = Fd_;
        Fd_ = INVALID_PIPEHANDLE;
        return ret;
    }

    inline void Swap(TPipeHandle& r) noexcept {
        DoSwap(Fd_, r.Fd_);
    }

    inline operator PIPEHANDLE() const noexcept {
        return Fd_;
    }

    inline bool IsOpen() const noexcept {
        return Fd_ != INVALID_PIPEHANDLE;
    }

    ssize_t Read(void* buffer, size_t byteCount) const noexcept;
    ssize_t Write(const void* buffer, size_t byteCount) const noexcept;

    // Only CloseOnExec is supported
    static void Pipe(TPipeHandle& reader, TPipeHandle& writer, EOpenMode mode = 0);

private:
    PIPEHANDLE Fd_;
};

class TPipe {
public:
    TPipe();
    /// Takes ownership of handle, so closes it when the last holder of descriptor dies.
    explicit TPipe(PIPEHANDLE fd);
    ~TPipe();

    void Close();

    bool IsOpen() const noexcept;
    PIPEHANDLE GetHandle() const noexcept;

    size_t Read(void* buf, size_t len) const;
    size_t Write(const void* buf, size_t len) const;

    // Only CloseOnExec is supported
    static void Pipe(TPipe& reader, TPipe& writer, EOpenMode mode = 0);

private:
    class TImpl;
    using TImplRef = TSimpleIntrusivePtr<TImpl>;
    TImplRef Impl_;
};

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif
