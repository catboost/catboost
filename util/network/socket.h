#pragma once

#include "init.h"

#include <util/system/yassert.h>
#include <util/system/defaults.h>
#include <util/system/error.h>
#include <util/stream/output.h>
#include <util/stream/input.h>
#include <util/generic/ptr.h>
#include <util/generic/yexception.h>
#include <util/generic/noncopyable.h>
#include <util/datetime/base.h>

#include <cerrno>

#ifndef INET_ADDRSTRLEN
    #define INET_ADDRSTRLEN 16
#endif

#if defined(_unix_)
    #define get_host_error() h_errno
#elif defined(_win_)
    #pragma comment(lib, "Ws2_32.lib")

    #if _WIN32_WINNT < 0x0600
struct pollfd {
    SOCKET fd;
    short events;
    short revents;
};

        #define POLLIN (1 << 0)
        #define POLLRDNORM (1 << 1)
        #define POLLRDBAND (1 << 2)
        #define POLLPRI (1 << 3)
        #define POLLOUT (1 << 4)
        #define POLLWRNORM (1 << 5)
        #define POLLWRBAND (1 << 6)
        #define POLLERR (1 << 7)
        #define POLLHUP (1 << 8)
        #define POLLNVAL (1 << 9)

const char* inet_ntop(int af, const void* src, char* dst, socklen_t size);
int poll(struct pollfd fds[], nfds_t nfds, int timeout) noexcept;
    #else
        #define poll(fds, nfds, timeout) WSAPoll(fds, nfds, timeout)
    #endif

int inet_aton(const char* cp, struct in_addr* inp);

    #define get_host_error() WSAGetLastError()

    #define SHUT_RD SD_RECEIVE
    #define SHUT_WR SD_SEND
    #define SHUT_RDWR SD_BOTH

    #define INFTIM (-1)
#endif

template <class T>
static inline int SetSockOpt(SOCKET s, int level, int optname, T opt) noexcept {
    return setsockopt(s, level, optname, (const char*)&opt, sizeof(opt));
}

template <class T>
static inline int GetSockOpt(SOCKET s, int level, int optname, T& opt) noexcept {
    socklen_t len = sizeof(opt);

    return getsockopt(s, level, optname, (char*)&opt, &len);
}

template <class T>
static inline void CheckedSetSockOpt(SOCKET s, int level, int optname, T opt, const char* err) {
    if (SetSockOpt<T>(s, level, optname, opt)) {
        ythrow TSystemError() << "setsockopt() failed for " << err;
    }
}

template <class T>
static inline void CheckedGetSockOpt(SOCKET s, int level, int optname, T& opt, const char* err) {
    if (GetSockOpt<T>(s, level, optname, opt)) {
        ythrow TSystemError() << "getsockopt() failed for " << err;
    }
}

static inline void FixIPv6ListenSocket(SOCKET s) {
#if defined(IPV6_V6ONLY)
    SetSockOpt(s, IPPROTO_IPV6, IPV6_V6ONLY, 1);
#else
    (void)s;
#endif
}

namespace NAddr {
    class IRemoteAddr;
}

void SetSocketTimeout(SOCKET s, long timeout);
void SetSocketTimeout(SOCKET s, long sec, long msec);
void SetNoDelay(SOCKET s, bool value);
void SetKeepAlive(SOCKET s);
void SetLinger(SOCKET s, bool on, unsigned len);
void SetZeroLinger(SOCKET s);
void SetKeepAlive(SOCKET s, bool value);
void SetCloseOnExec(SOCKET s, bool value);
void SetOutputBuffer(SOCKET s, unsigned value);
void SetInputBuffer(SOCKET s, unsigned value);
void SetReusePort(SOCKET s, bool value);
void ShutDown(SOCKET s, int mode);
bool GetRemoteAddr(SOCKET s, char* str, socklen_t size);
size_t GetMaximumSegmentSize(SOCKET s);
size_t GetMaximumTransferUnit(SOCKET s);
void SetDeferAccept(SOCKET s);
void SetSocketToS(SOCKET s, int tos);
void SetSocketToS(SOCKET s, const NAddr::IRemoteAddr* addr, int tos);
int GetSocketToS(SOCKET s);
int GetSocketToS(SOCKET s, const NAddr::IRemoteAddr* addr);
void SetSocketPriority(SOCKET s, int priority);
void SetTcpFastOpen(SOCKET s, int qlen);
/**
 * Deprecated, consider using HasSocketDataToRead instead.
 **/
bool IsNotSocketClosedByOtherSide(SOCKET s);
enum class ESocketReadStatus {
    HasData,
    NoData,
    SocketClosed
};
/**
 * Useful for keep-alive connections.
 **/
ESocketReadStatus HasSocketDataToRead(SOCKET s);
/**
 * Determines whether connection on socket is local (same machine) or not.
 **/
bool HasLocalAddress(SOCKET socket);

bool IsNonBlock(SOCKET fd);
void SetNonBlock(SOCKET fd, bool nonBlock = true);

struct addrinfo;

class TNetworkResolutionError: public yexception {
public:
    // @param error error code (EAI_XXX) returned by getaddrinfo or getnameinfo (not errno)
    TNetworkResolutionError(int error);
};

struct TUnixSocketPath {
    TString Path;

    // Constructor for create unix domain socket path from string with path in filesystem
    // TUnixSocketPath("/tmp/unixsocket") -> "/tmp/unixsocket"
    explicit TUnixSocketPath(const TString& path)
        : Path(path)
    {
    }
};

class TNetworkAddress {
    friend class TSocket;

public:
    class TIterator {
    public:
        inline TIterator(struct addrinfo* begin)
            : C_(begin)
        {
        }

        inline void Next() noexcept {
            C_ = C_->ai_next;
        }

        inline TIterator operator++(int) noexcept {
            TIterator old(*this);

            Next();

            return old;
        }

        inline TIterator& operator++() noexcept {
            Next();

            return *this;
        }

        friend inline bool operator==(const TIterator& l, const TIterator& r) noexcept {
            return l.C_ == r.C_;
        }

        friend inline bool operator!=(const TIterator& l, const TIterator& r) noexcept {
            return !(l == r);
        }

        inline struct addrinfo& operator*() const noexcept {
            return *C_;
        }

        inline struct addrinfo* operator->() const noexcept {
            return C_;
        }

    private:
        struct addrinfo* C_;
    };

    TNetworkAddress(ui16 port);
    TNetworkAddress(const TString& host, ui16 port);
    TNetworkAddress(const TString& host, ui16 port, int flags);
    TNetworkAddress(const TUnixSocketPath& unixSocketPath, int flags = 0);
    ~TNetworkAddress();

    inline TIterator Begin() const noexcept {
        return TIterator(Info());
    }

    inline TIterator End() const noexcept {
        return TIterator(nullptr);
    }

private:
    struct addrinfo* Info() const noexcept;

private:
    class TImpl;
    TSimpleIntrusivePtr<TImpl> Impl_;
};

class TSocket;

class TSocketHolder: public TMoveOnly {
public:
    inline TSocketHolder()
        : Fd_(INVALID_SOCKET)
    {
    }

    inline TSocketHolder(SOCKET fd)
        : Fd_(fd)
    {
    }

    inline TSocketHolder(TSocketHolder&& other) noexcept {
        Fd_ = other.Fd_;
        other.Fd_ = INVALID_SOCKET;
    }

    inline TSocketHolder& operator=(TSocketHolder&& other) noexcept {
        Close();
        Swap(other);

        return *this;
    }

    inline ~TSocketHolder() {
        Close();
    }

    inline SOCKET Release() noexcept {
        SOCKET ret = Fd_;
        Fd_ = INVALID_SOCKET;
        return ret;
    }

    void Close() noexcept;

    inline void ShutDown(int mode) const {
        ::ShutDown(Fd_, mode);
    }

    inline void Swap(TSocketHolder& r) noexcept {
        DoSwap(Fd_, r.Fd_);
    }

    inline bool Closed() const noexcept {
        return Fd_ == INVALID_SOCKET;
    }

    inline operator SOCKET() const noexcept {
        return Fd_;
    }

private:
    SOCKET Fd_;

    // do not allow construction of TSocketHolder from TSocket
    TSocketHolder(const TSocket& fd);
};

class TSocket {
public:
    using TPart = IOutputStream::TPart;

    class TOps {
    public:
        inline TOps() noexcept = default;
        virtual ~TOps() = default;

        virtual ssize_t Send(SOCKET fd, const void* data, size_t len) = 0;
        virtual ssize_t Recv(SOCKET fd, void* buf, size_t len) = 0;
        virtual ssize_t SendV(SOCKET fd, const TPart* parts, size_t count) = 0;
    };

    TSocket();
    TSocket(SOCKET fd);
    TSocket(SOCKET fd, TOps* ops);
    TSocket(const TNetworkAddress& addr);
    TSocket(const TNetworkAddress& addr, const TDuration& timeOut);
    TSocket(const TNetworkAddress& addr, const TInstant& deadLine);

    ~TSocket();

    template <class T>
    inline void SetSockOpt(int level, int optname, T opt) {
        CheckedSetSockOpt(Fd(), level, optname, opt, "TSocket");
    }

    inline void SetSocketTimeout(long timeout) {
        ::SetSocketTimeout(Fd(), timeout);
    }

    inline void SetSocketTimeout(long sec, long msec) {
        ::SetSocketTimeout(Fd(), sec, msec);
    }

    inline void SetNoDelay(bool value) {
        ::SetNoDelay(Fd(), value);
    }

    inline void SetLinger(bool on, unsigned len) {
        ::SetLinger(Fd(), on, len);
    }

    inline void SetZeroLinger() {
        ::SetZeroLinger(Fd());
    }

    inline void SetKeepAlive(bool value) {
        ::SetKeepAlive(Fd(), value);
    }

    inline void SetOutputBuffer(unsigned value) {
        ::SetOutputBuffer(Fd(), value);
    }

    inline void SetInputBuffer(unsigned value) {
        ::SetInputBuffer(Fd(), value);
    }

    inline size_t MaximumSegmentSize() const {
        return GetMaximumSegmentSize(Fd());
    }

    inline size_t MaximumTransferUnit() const {
        return GetMaximumTransferUnit(Fd());
    }

    inline void ShutDown(int mode) const {
        ::ShutDown(Fd(), mode);
    }

    void Close();

    ssize_t Send(const void* data, size_t len);
    ssize_t Recv(void* buf, size_t len);

    /*
     * scatter/gather io
     */
    ssize_t SendV(const TPart* parts, size_t count);

    inline operator SOCKET() const noexcept {
        return Fd();
    }

private:
    SOCKET Fd() const noexcept;

private:
    class TImpl;
    TSimpleIntrusivePtr<TImpl> Impl_;
};

class TSocketInput: public IInputStream {
public:
    TSocketInput(const TSocket& s) noexcept;
    ~TSocketInput() override;

    TSocketInput(TSocketInput&&) noexcept = default;
    TSocketInput& operator=(TSocketInput&&) noexcept = default;

    const TSocket& GetSocket() const noexcept {
        return S_;
    }

private:
    size_t DoRead(void* buf, size_t len) override;

private:
    TSocket S_;
};

class TSocketOutput: public IOutputStream {
public:
    TSocketOutput(const TSocket& s) noexcept;
    ~TSocketOutput() override;

    TSocketOutput(TSocketOutput&&) noexcept = default;
    TSocketOutput& operator=(TSocketOutput&&) noexcept = default;

    const TSocket& GetSocket() const noexcept {
        return S_;
    }

private:
    void DoWrite(const void* buf, size_t len) override;
    void DoWriteV(const TPart* parts, size_t count) override;

private:
    TSocket S_;
};

//return -(error code) if error occured, or number of ready fds
ssize_t PollD(struct pollfd fds[], nfds_t nfds, const TInstant& deadLine) noexcept;
