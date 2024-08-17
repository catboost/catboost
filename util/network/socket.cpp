#include "ip.h"
#include "socket.h"
#include "address.h"
#include "pollerimpl.h"
#include "iovec.h"

#include <util/system/defaults.h>
#include <util/system/byteorder.h>

#if defined(_unix_)
    #include <netdb.h>
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <sys/un.h>
    #include <sys/ioctl.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
#endif

#if defined(_freebsd_)
    #include <sys/module.h>
    #define ACCEPT_FILTER_MOD
    #include <sys/socketvar.h>
#endif

#if defined(_win_)
    #include <cerrno>
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <wspiapi.h>

    #include <util/system/compat.h>
#endif

#include <util/generic/ylimits.h>

#include <util/string/cast.h>
#include <util/stream/mem.h>
#include <util/system/datetime.h>
#include <util/system/error.h>
#include <util/memory/tempbuf.h>
#include <util/generic/singleton.h>
#include <util/generic/hash_set.h>

#include <stddef.h>
#include <sys/uio.h>

using namespace NAddr;

#if defined(_win_)

int inet_aton(const char* cp, struct in_addr* inp) {
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    int psz = sizeof(addr);
    if (0 == WSAStringToAddress((char*)cp, AF_INET, nullptr, (LPSOCKADDR)&addr, &psz)) {
        memcpy(inp, &addr.sin_addr, sizeof(in_addr));
        return 1;
    }
    return 0;
}

    #if (_WIN32_WINNT < 0x0600)
const char* inet_ntop(int af, const void* src, char* dst, socklen_t size) {
    if (af != AF_INET) {
        errno = EINVAL;
        return 0;
    }
    const ui8* ia = (ui8*)src;
    if (snprintf(dst, size, "%u.%u.%u.%u", ia[0], ia[1], ia[2], ia[3]) >= (int)size) {
        errno = ENOSPC;
        return 0;
    }
    return dst;
}

struct evpair {
    int event;
    int winevent;
};

static const evpair evpairs_to_win[] = {
    {POLLIN, FD_READ | FD_CLOSE | FD_ACCEPT},
    {POLLRDNORM, FD_READ | FD_CLOSE | FD_ACCEPT},
    {POLLRDBAND, -1},
    {POLLPRI, -1},
    {POLLOUT, FD_WRITE | FD_CLOSE},
    {POLLWRNORM, FD_WRITE | FD_CLOSE},
    {POLLWRBAND, -1},
    {POLLERR, 0},
    {POLLHUP, 0},
    {POLLNVAL, 0}};

static const size_t nevpairs_to_win = sizeof(evpairs_to_win) / sizeof(evpairs_to_win[0]);

static const evpair evpairs_to_unix[] = {
    {FD_ACCEPT, POLLIN | POLLRDNORM},
    {FD_READ, POLLIN | POLLRDNORM},
    {FD_WRITE, POLLOUT | POLLWRNORM},
    {FD_CLOSE, POLLHUP},
};

static const size_t nevpairs_to_unix = sizeof(evpairs_to_unix) / sizeof(evpairs_to_unix[0]);

static int convert_events(int events, const evpair* evpairs, size_t nevpairs, bool ignoreUnknown) noexcept {
    int result = 0;
    for (size_t i = 0; i < nevpairs; ++i) {
        int event = evpairs[i].event;
        if (events & event) {
            events ^= event;
            long winEvent = evpairs[i].winevent;
            if (winEvent == -1)
                return -1;
            if (winEvent == 0)
                continue;
            result |= winEvent;
        }
    }
    if (events != 0 && !ignoreUnknown)
        return -1;
    return result;
}

class TWSAEventHolder {
private:
    HANDLE Event;

public:
    inline TWSAEventHolder(HANDLE event) noexcept
        : Event(event)
    {
    }

    inline ~TWSAEventHolder() {
        WSACloseEvent(Event);
    }

    inline HANDLE Get() noexcept {
        return Event;
    }
};

int poll(struct pollfd fds[], nfds_t nfds, int timeout) noexcept {
    HANDLE rawEvent = WSACreateEvent();
    if (rawEvent == WSA_INVALID_EVENT) {
        errno = EIO;
        return -1;
    }

    TWSAEventHolder event(rawEvent);

    int checked_sockets = 0;

    for (pollfd* fd = fds; fd < fds + nfds; ++fd) {
        int win_events = convert_events(fd->events, evpairs_to_win, nevpairs_to_win, false);
        if (win_events == -1) {
            errno = EINVAL;
            return -1;
        }
        fd->revents = 0;
        if (WSAEventSelect(fd->fd, event.Get(), win_events)) {
            int error = WSAGetLastError();
            if (error == WSAEINVAL || error == WSAENOTSOCK) {
                fd->revents = POLLNVAL;
                ++checked_sockets;
            } else {
                errno = EIO;
                return -1;
            }
        }
        fd_set readfds;
        fd_set writefds;
        struct timeval timeout = {0, 0};
        FD_ZERO(&readfds);
        FD_ZERO(&writefds);
        if (fd->events & POLLIN) {
            FD_SET(fd->fd, &readfds);
        }
        if (fd->events & POLLOUT) {
            FD_SET(fd->fd, &writefds);
        }
        int error = select(0, &readfds, &writefds, nullptr, &timeout);
        if (error > 0) {
            if (FD_ISSET(fd->fd, &readfds)) {
                fd->revents |= POLLIN;
            }
            if (FD_ISSET(fd->fd, &writefds)) {
                fd->revents |= POLLOUT;
            }
            ++checked_sockets;
        }
    }

    if (checked_sockets > 0) {
        // returns without wait since we already have sockets in desired conditions
        return checked_sockets;
    }

    HANDLE events[] = {event.Get()};
    DWORD wait_result = WSAWaitForMultipleEvents(1, events, TRUE, timeout, FALSE);
    if (wait_result == WSA_WAIT_TIMEOUT)
        return 0;
    else if (wait_result == WSA_WAIT_EVENT_0) {
        for (pollfd* fd = fds; fd < fds + nfds; ++fd) {
            if (fd->revents == POLLNVAL)
                continue;
            WSANETWORKEVENTS network_events;
            if (WSAEnumNetworkEvents(fd->fd, event.Get(), &network_events)) {
                errno = EIO;
                return -1;
            }
            fd->revents = 0;
            for (int i = 0; i < FD_MAX_EVENTS; ++i) {
                if ((network_events.lNetworkEvents & (1 << i)) != 0 && network_events.iErrorCode[i]) {
                    fd->revents = POLLERR;
                    break;
                }
            }
            if (fd->revents == POLLERR)
                continue;
            if (network_events.lNetworkEvents) {
                fd->revents = static_cast<short>(convert_events(network_events.lNetworkEvents, evpairs_to_unix, nevpairs_to_unix, true));
                if (fd->revents & POLLHUP) {
                    fd->revents &= POLLHUP | POLLIN | POLLRDNORM;
                }
            }
        }
        int chanded_sockets = 0;
        for (pollfd* fd = fds; fd < fds + nfds; ++fd)
            if (fd->revents != 0)
                ++chanded_sockets;
        return chanded_sockets;
    } else {
        errno = EIO;
        return -1;
    }
}
    #endif

#endif

bool GetRemoteAddr(SOCKET Socket, char* str, socklen_t size) {
    if (!size) {
        return false;
    }

    TOpaqueAddr addr;

    if (getpeername(Socket, addr.MutableAddr(), addr.LenPtr()) != 0) {
        return false;
    }

    try {
        TMemoryOutput out(str, size - 1);

        PrintHost(out, addr);
        *out.Buf() = 0;

        return true;
    } catch (...) {
        // ¯\_(ツ)_/¯
    }

    return false;
}

void SetSocketTimeout(SOCKET s, long timeout) {
    SetSocketTimeout(s, timeout, 0);
}

void SetSocketTimeout(SOCKET s, long sec, long msec) {
#ifdef SO_SNDTIMEO
    #ifdef _darwin_
    const timeval timeout = {sec, (__darwin_suseconds_t)msec * 1000};
    #elif defined(_emscripten_)
    const timeval timeout = {sec, static_cast<suseconds_t>(msec * 1000)};
    #elif defined(_unix_)
    const timeval timeout = {sec, msec * 1000};
    #else
    const int timeout = sec * 1000 + msec;
    #endif
    CheckedSetSockOpt(s, SOL_SOCKET, SO_RCVTIMEO, timeout, "recv timeout");
    CheckedSetSockOpt(s, SOL_SOCKET, SO_SNDTIMEO, timeout, "send timeout");
#endif
}

void SetLinger(SOCKET s, bool on, unsigned len) {
#ifdef SO_LINGER
    struct linger l = {on, (u_short)len};

    CheckedSetSockOpt(s, SOL_SOCKET, SO_LINGER, l, "linger");
#endif
}

void SetZeroLinger(SOCKET s) {
    SetLinger(s, 1, 0);
}

void SetKeepAlive(SOCKET s, bool value) {
    CheckedSetSockOpt(s, SOL_SOCKET, SO_KEEPALIVE, (int)value, "keepalive");
}

void SetOutputBuffer(SOCKET s, unsigned value) {
    CheckedSetSockOpt(s, SOL_SOCKET, SO_SNDBUF, value, "output buffer");
}

void SetInputBuffer(SOCKET s, unsigned value) {
    CheckedSetSockOpt(s, SOL_SOCKET, SO_RCVBUF, value, "input buffer");
}

void SetReusePort(SOCKET s, bool value) {
#if defined(_unix_)
    CheckedSetSockOpt(s, SOL_SOCKET, SO_REUSEPORT, (int)value, "reuse port");
#else
    Y_UNUSED(s);
    Y_UNUSED(value);
    ythrow TSystemError(ENOSYS) << "SO_REUSEPORT is not available on Windows";
#endif
}

void SetNoDelay(SOCKET s, bool value) {
    CheckedSetSockOpt(s, IPPROTO_TCP, TCP_NODELAY, (int)value, "tcp no delay");
}

void SetCloseOnExec(SOCKET s, bool value) {
#if defined(_unix_)
    int flags = fcntl(s, F_GETFD);
    if (flags == -1) {
        ythrow TSystemError() << "fcntl() failed";
    }
    if (value) {
        flags |= FD_CLOEXEC;
    } else {
        flags &= ~FD_CLOEXEC;
    }
    if (fcntl(s, F_SETFD, flags) == -1) {
        ythrow TSystemError() << "fcntl() failed";
    }
#else
    Y_UNUSED(s);
    Y_UNUSED(value);
#endif
}

size_t GetMaximumSegmentSize(SOCKET s) {
#if defined(TCP_MAXSEG)
    int val;

    if (GetSockOpt(s, IPPROTO_TCP, TCP_MAXSEG, val) == 0) {
        return (size_t)val;
    }
#endif

    /*
     * probably a good guess...
     */
    return 8192;
}

size_t GetMaximumTransferUnit(SOCKET /*s*/) {
    // for someone who'll dare to write it
    // Linux: there rummored to be IP_MTU getsockopt() request
    // FreeBSD: request to a socket of type PF_ROUTE
    //          with peer address as a destination argument
    return 8192;
}

int GetSocketToS(SOCKET s) {
    TOpaqueAddr addr;

    if (getsockname(s, addr.MutableAddr(), addr.LenPtr()) < 0) {
        ythrow TSystemError() << "getsockname() failed";
    }

    return GetSocketToS(s, &addr);
}

int GetSocketToS(SOCKET s, const IRemoteAddr* addr) {
    int result = 0;

    switch (addr->Addr()->sa_family) {
        case AF_INET:
            CheckedGetSockOpt(s, IPPROTO_IP, IP_TOS, result, "tos");
            break;

        case AF_INET6:
#ifdef IPV6_TCLASS
            CheckedGetSockOpt(s, IPPROTO_IPV6, IPV6_TCLASS, result, "tos");
#endif
            break;
    }

    return result;
}

void SetSocketToS(SOCKET s, const NAddr::IRemoteAddr* addr, int tos) {
    switch (addr->Addr()->sa_family) {
        case AF_INET:
            CheckedSetSockOpt(s, IPPROTO_IP, IP_TOS, tos, "tos");
            return;

        case AF_INET6:
#ifdef IPV6_TCLASS
            CheckedSetSockOpt(s, IPPROTO_IPV6, IPV6_TCLASS, tos, "tos");
            return;
#endif
            break;
    }

    ythrow yexception() << "SetSocketToS unsupported for family " << addr->Addr()->sa_family;
}

void SetSocketToS(SOCKET s, int tos) {
    TOpaqueAddr addr;

    if (getsockname(s, addr.MutableAddr(), addr.LenPtr()) < 0) {
        ythrow TSystemError() << "getsockname() failed";
    }

    SetSocketToS(s, &addr, tos);
}

void SetSocketPriority(SOCKET s, int priority) {
#if defined(SO_PRIORITY)
    CheckedSetSockOpt(s, SOL_SOCKET, SO_PRIORITY, priority, "priority");
#else
    Y_UNUSED(s);
    Y_UNUSED(priority);
#endif
}

bool HasLocalAddress(SOCKET socket) {
    TOpaqueAddr localAddr;
    if (getsockname(socket, localAddr.MutableAddr(), localAddr.LenPtr()) != 0) {
        ythrow TSystemError() << "HasLocalAddress: getsockname() failed. ";
    }
    if (IsLoopback(localAddr)) {
        return true;
    }

    TOpaqueAddr remoteAddr;
    if (getpeername(socket, remoteAddr.MutableAddr(), remoteAddr.LenPtr()) != 0) {
        ythrow TSystemError() << "HasLocalAddress: getpeername() failed. ";
    }
    return IsSame(localAddr, remoteAddr);
}

namespace {
#if defined(_linux_)
    #if !defined(TCP_FASTOPEN)
        #define TCP_FASTOPEN 23
    #endif
#endif

#if defined(TCP_FASTOPEN)
    struct TTcpFastOpenFeature {
        inline TTcpFastOpenFeature()
            : HasFastOpen_(false)
        {
            TSocketHolder tmp(socket(AF_INET, SOCK_STREAM, 0));
            int val = 1;
            int ret = SetSockOpt(tmp, IPPROTO_TCP, TCP_FASTOPEN, val);
            HasFastOpen_ = (ret == 0);
        }

        inline void SetFastOpen(SOCKET s, int qlen) const {
            if (HasFastOpen_) {
                CheckedSetSockOpt(s, IPPROTO_TCP, TCP_FASTOPEN, qlen, "setting TCP_FASTOPEN");
            }
        }

        static inline const TTcpFastOpenFeature* Instance() noexcept {
            return Singleton<TTcpFastOpenFeature>();
        }

        bool HasFastOpen_;
    };
#endif
}

void SetTcpFastOpen(SOCKET s, int qlen) {
#if defined(TCP_FASTOPEN)
    TTcpFastOpenFeature::Instance()->SetFastOpen(s, qlen);
#else
    Y_UNUSED(s);
    Y_UNUSED(qlen);
#endif
}

static bool IsBlocked(int lasterr) noexcept {
    return lasterr == EAGAIN || lasterr == EWOULDBLOCK;
}

struct TUnblockingGuard {
    SOCKET S_;

    TUnblockingGuard(SOCKET s)
        : S_(s)
    {
        SetNonBlock(S_, true);
    }

    ~TUnblockingGuard() {
        SetNonBlock(S_, false);
    }
};

static int MsgPeek(SOCKET s) {
    int flags = MSG_PEEK;

#if defined(_win_)
    TUnblockingGuard unblocker(s);
    Y_UNUSED(unblocker);
#else
    flags |= MSG_DONTWAIT;
#endif

    char c;
    return recv(s, &c, 1, flags);
}

bool IsNotSocketClosedByOtherSide(SOCKET s) {
    return HasSocketDataToRead(s) != ESocketReadStatus::SocketClosed;
}

ESocketReadStatus HasSocketDataToRead(SOCKET s) {
    const int r = MsgPeek(s);
    if (r == -1 && IsBlocked(LastSystemError())) {
        return ESocketReadStatus::NoData;
    }
    if (r > 0) {
        return ESocketReadStatus::HasData;
    }
    return ESocketReadStatus::SocketClosed;
}

#if defined(_win_)
static ssize_t DoSendMsg(SOCKET sock, const struct iovec* iov, int iovcnt) {
    return writev(sock, iov, iovcnt);
}
#else
static ssize_t DoSendMsg(SOCKET sock, const struct iovec* iov, int iovcnt) {
    struct msghdr message;

    Zero(message);
    message.msg_iov = const_cast<struct iovec*>(iov);
    message.msg_iovlen = iovcnt;

    return sendmsg(sock, &message, MSG_NOSIGNAL);
}
#endif

void TSocketHolder::Close() noexcept {
    if (Fd_ != INVALID_SOCKET) {
        bool ok = (closesocket(Fd_) == 0);
        if (!ok) {
// Do not quietly close bad descriptor,
// because often it means double close
// that is disasterous
#ifdef _win_
            Y_ABORT_UNLESS(WSAGetLastError() != WSAENOTSOCK, "must not quietly close bad socket descriptor");
#elif defined(_unix_)
            Y_ABORT_UNLESS(errno != EBADF, "must not quietly close bad descriptor: fd=%d", int(Fd_));
#else
    #error unsupported platform
#endif
        }

        Fd_ = INVALID_SOCKET;
    }
}

class TSocket::TImpl: public TAtomicRefCount<TImpl> {
    using TOps = TSocket::TOps;

public:
    inline TImpl(SOCKET fd, TOps* ops)
        : Fd_(fd)
        , Ops_(ops)
    {
    }

    inline ~TImpl() = default;

    inline SOCKET Fd() const noexcept {
        return Fd_;
    }

    inline ssize_t Send(const void* data, size_t len) {
        return Ops_->Send(Fd_, data, len);
    }

    inline ssize_t Recv(void* buf, size_t len) {
        return Ops_->Recv(Fd_, buf, len);
    }

    inline ssize_t SendV(const TPart* parts, size_t count) {
        return Ops_->SendV(Fd_, parts, count);
    }

    inline void Close() {
        Fd_.Close();
    }

private:
    TSocketHolder Fd_;
    TOps* Ops_;
};

template <>
void Out<const struct addrinfo*>(IOutputStream& os, const struct addrinfo* ai) {
    if (ai->ai_flags & AI_CANONNAME) {
        os << "`" << ai->ai_canonname << "' ";
    }

    os << '[';
    for (int i = 0; ai; ++i, ai = ai->ai_next) {
        if (i > 0) {
            os << ", ";
        }

        os << (const IRemoteAddr&)TAddrInfo(ai);
    }
    os << ']';
}

template <>
void Out<struct addrinfo*>(IOutputStream& os, struct addrinfo* ai) {
    Out<const struct addrinfo*>(os, static_cast<const struct addrinfo*>(ai));
}

template <>
void Out<TNetworkAddress>(IOutputStream& os, const TNetworkAddress& addr) {
    os << &*addr.Begin();
}

static inline const struct addrinfo* Iterate(const struct addrinfo* addr, const struct addrinfo* addr0, const int sockerr) {
    if (addr->ai_next) {
        return addr->ai_next;
    }

    ythrow TSystemError(sockerr) << "can not connect to " << addr0;
}

static inline SOCKET DoConnectImpl(const struct addrinfo* res, const TInstant& deadLine) {
    const struct addrinfo* addr0 = res;

    while (res) {
        TSocketHolder s(socket(res->ai_family, res->ai_socktype, res->ai_protocol));

        if (s.Closed()) {
            res = Iterate(res, addr0, LastSystemError());

            continue;
        }

        SetNonBlock(s, true);

        if (connect(s, res->ai_addr, (int)res->ai_addrlen)) {
            int err = LastSystemError();

            if (err == EINPROGRESS || err == EAGAIN || err == EWOULDBLOCK) {
                /*
                 * must wait
                 */
                struct pollfd p = {
                    (SOCKET)s,
                    POLLOUT,
                    0};

                const ssize_t n = PollD(&p, 1, deadLine);

                /*
                 * timeout occured
                 */
                if (n < 0) {
                    ythrow TSystemError(-(int)n) << "can not connect";
                }

                CheckedGetSockOpt(s, SOL_SOCKET, SO_ERROR, err, "socket error");

                if (!err) {
                    return s.Release();
                }
            }

            res = Iterate(res, addr0, err);

            continue;
        }

        return s.Release();
    }

    ythrow yexception() << "something went wrong: nullptr at addrinfo";
}

static inline SOCKET DoConnect(const struct addrinfo* res, const TInstant& deadLine) {
    TSocketHolder ret(DoConnectImpl(res, deadLine));

    SetNonBlock(ret, false);

    return ret.Release();
}

static inline ssize_t DoSendV(SOCKET fd, const struct iovec* iov, size_t count) {
    ssize_t ret = -1;
    do {
        ret = DoSendMsg(fd, iov, (int)count);
    } while (ret == -1 && errno == EINTR);

    if (ret < 0) {
        return -LastSystemError();
    }

    return ret;
}

template <bool isCompat>
struct TSender {
    using TPart = TSocket::TPart;

    static inline ssize_t SendV(SOCKET fd, const TPart* parts, size_t count) {
        return DoSendV(fd, (const iovec*)parts, count);
    }
};

template <>
struct TSender<false> {
    using TPart = TSocket::TPart;

    static inline ssize_t SendV(SOCKET fd, const TPart* parts, size_t count) {
        TTempBuf tempbuf(sizeof(struct iovec) * count);
        struct iovec* iov = (struct iovec*)tempbuf.Data();

        for (size_t i = 0; i < count; ++i) {
            struct iovec& io = iov[i];
            const TPart& part = parts[i];

            io.iov_base = (char*)part.buf;
            io.iov_len = part.len;
        }

        return DoSendV(fd, iov, count);
    }
};

class TCommonSockOps: public TSocket::TOps {
    using TPart = TSocket::TPart;

public:
    inline TCommonSockOps() noexcept {
    }

    ~TCommonSockOps() override = default;

    ssize_t Send(SOCKET fd, const void* data, size_t len) override {
        ssize_t ret = -1;
        do {
            ret = send(fd, (const char*)data, (int)len, MSG_NOSIGNAL);
        } while (ret == -1 && errno == EINTR);

        if (ret < 0) {
            return -LastSystemError();
        }

        return ret;
    }

    ssize_t Recv(SOCKET fd, void* buf, size_t len) override {
        ssize_t ret = -1;
        do {
            ret = recv(fd, (char*)buf, (int)len, 0);
        } while (ret == -1 && errno == EINTR);

        if (ret < 0) {
            return -LastSystemError();
        }

        return ret;
    }

    ssize_t SendV(SOCKET fd, const TPart* parts, size_t count) override {
        ssize_t ret = SendVImpl(fd, parts, count);

        if (ret < 0) {
            return ret;
        }

        size_t len = TContIOVector::Bytes(parts, count);

        if ((size_t)ret == len) {
            return ret;
        }

        return SendVPartial(fd, parts, count, ret);
    }

    inline ssize_t SendVImpl(SOCKET fd, const TPart* parts, size_t count) {
        return TSender < (sizeof(iovec) == sizeof(TPart)) && (offsetof(iovec, iov_base) == offsetof(TPart, buf)) && (offsetof(iovec, iov_len) == offsetof(TPart, len)) > ::SendV(fd, parts, count);
    }

    ssize_t SendVPartial(SOCKET fd, const TPart* constParts, size_t count, size_t written);
};

ssize_t TCommonSockOps::SendVPartial(SOCKET fd, const TPart* constParts, size_t count, size_t written) {
    TTempBuf tempbuf(sizeof(TPart) * count);
    TPart* parts = (TPart*)tempbuf.Data();

    for (size_t i = 0; i < count; ++i) {
        parts[i] = constParts[i];
    }

    TContIOVector vec(parts, count);
    vec.Proceed(written);

    while (!vec.Complete()) {
        ssize_t ret = SendVImpl(fd, vec.Parts(), vec.Count());

        if (ret < 0) {
            return ret;
        }

        written += ret;

        vec.Proceed((size_t)ret);
    }

    return written;
}

static inline TSocket::TOps* GetCommonSockOps() noexcept {
    return Singleton<TCommonSockOps>();
}

TSocket::TSocket()
    : Impl_(new TImpl(INVALID_SOCKET, GetCommonSockOps()))
{
}

TSocket::TSocket(SOCKET fd)
    : Impl_(new TImpl(fd, GetCommonSockOps()))
{
}

TSocket::TSocket(SOCKET fd, TOps* ops)
    : Impl_(new TImpl(fd, ops))
{
}

TSocket::TSocket(const TNetworkAddress& addr)
    : Impl_(new TImpl(DoConnect(addr.Info(), TInstant::Max()), GetCommonSockOps()))
{
}

TSocket::TSocket(const TNetworkAddress& addr, const TDuration& timeOut)
    : Impl_(new TImpl(DoConnect(addr.Info(), timeOut.ToDeadLine()), GetCommonSockOps()))
{
}

TSocket::TSocket(const TNetworkAddress& addr, const TInstant& deadLine)
    : Impl_(new TImpl(DoConnect(addr.Info(), deadLine), GetCommonSockOps()))
{
}

TSocket::~TSocket() = default;

SOCKET TSocket::Fd() const noexcept {
    return Impl_->Fd();
}

ssize_t TSocket::Send(const void* data, size_t len) {
    return Impl_->Send(data, len);
}

ssize_t TSocket::Recv(void* buf, size_t len) {
    return Impl_->Recv(buf, len);
}

ssize_t TSocket::SendV(const TPart* parts, size_t count) {
    return Impl_->SendV(parts, count);
}

void TSocket::Close() {
    Impl_->Close();
}

TSocketInput::TSocketInput(const TSocket& s) noexcept
    : S_(s)
{
}

TSocketInput::~TSocketInput() = default;

size_t TSocketInput::DoRead(void* buf, size_t len) {
    const ssize_t ret = S_.Recv(buf, len);

    if (ret >= 0) {
        return (size_t)ret;
    }

    ythrow TSystemError(-(int)ret) << "can not read from socket input stream";
}

TSocketOutput::TSocketOutput(const TSocket& s) noexcept
    : S_(s)
{
}

TSocketOutput::~TSocketOutput() {
    try {
        Finish();
    } catch (...) {
        // ¯\_(ツ)_/¯
    }
}

void TSocketOutput::DoWrite(const void* buf, size_t len) {
    size_t send = 0;
    while (len) {
        const ssize_t ret = S_.Send(buf, len);

        if (ret < 0) {
            ythrow TSystemError(-(int)ret) << "can not write to socket output stream; " << send << " bytes already send";
        }
        buf = (const char*)buf + ret;
        len -= ret;
        send += ret;
    }
}

void TSocketOutput::DoWriteV(const TPart* parts, size_t count) {
    const ssize_t ret = S_.SendV(parts, count);

    if (ret < 0) {
        ythrow TSystemError(-(int)ret) << "can not writev to socket output stream";
    }

    /*
     * todo for nonblocking sockets?
     */
}

namespace {
    // https://bugzilla.mozilla.org/attachment.cgi?id=503263&action=diff

    struct TLocalNames: public THashSet<TStringBuf> {
        inline TLocalNames() {
            insert("localhost");
            insert("localhost.localdomain");
            insert("localhost6");
            insert("localhost6.localdomain6");
            insert("::1");
        }

        inline bool IsLocalName(const char* name) const noexcept {
            struct sockaddr_in sa;
            memset(&sa, 0, sizeof(sa));

            if (inet_pton(AF_INET, name, &(sa.sin_addr)) == 1) {
                return (InetToHost(sa.sin_addr.s_addr) >> 24) == 127;
            }

            return contains(name);
        }
    };
}

class TNetworkAddress::TImpl: public TAtomicRefCount<TImpl> {
private:
    class TAddrInfoDeleter {
    public:
        TAddrInfoDeleter(bool useFreeAddrInfo = true)
            : UseFreeAddrInfo_(useFreeAddrInfo)
        {
        }

        void operator()(struct addrinfo* ai) noexcept {
            if (!UseFreeAddrInfo_ && ai != NULL) {
                if (ai->ai_addr != NULL) {
                    free(ai->ai_addr);
                }

                struct addrinfo* p;
                while (ai != NULL) {
                    p = ai;
                    ai = ai->ai_next;
                    free(p->ai_canonname);
                    free(p);
                }
            } else if (ai != NULL) {
                freeaddrinfo(ai);
            }
        }

    private:
        bool UseFreeAddrInfo_ = true;
    };

public:
    inline TImpl(const char* host, ui16 port, int flags)
        : Info_(nullptr, TAddrInfoDeleter{})
    {
        const TString port_st(ToString(port));
        struct addrinfo hints;

        memset(&hints, 0, sizeof(hints));

        hints.ai_flags = flags;
        hints.ai_family = PF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;

        if (!host) {
            hints.ai_flags |= AI_PASSIVE;
        } else {
            if (!Singleton<TLocalNames>()->IsLocalName(host)) {
                hints.ai_flags |= AI_ADDRCONFIG;
            }
        }

        struct addrinfo* pai = NULL;
        const int error = getaddrinfo(host, port_st.data(), &hints, &pai);

        if (error) {
            TAddrInfoDeleter()(pai);
            ythrow TNetworkResolutionError(error) << ": can not resolve " << host << ":" << port;
        }

        Info_.reset(pai);
    }

    inline TImpl(const char* path, int flags)
        : Info_(nullptr, TAddrInfoDeleter{/* useFreeAddrInfo = */ false})
    {
        THolder<struct sockaddr_un, TFree> sockAddr(
            reinterpret_cast<struct sockaddr_un*>(malloc(sizeof(struct sockaddr_un))));

        Y_ENSURE(strlen(path) < sizeof(sockAddr->sun_path), "Unix socket path more than " << sizeof(sockAddr->sun_path));
        sockAddr->sun_family = AF_UNIX;
        strcpy(sockAddr->sun_path, path);

        TAddrInfoPtr hints(reinterpret_cast<struct addrinfo*>(malloc(sizeof(struct addrinfo))), TAddrInfoDeleter{/* useFreeAddrInfo = */ false});
        memset(hints.get(), 0, sizeof(*hints));

        hints->ai_flags = flags;
        hints->ai_family = AF_UNIX;
        hints->ai_socktype = SOCK_STREAM;
        hints->ai_addrlen = sizeof(*sockAddr);
        hints->ai_addr = (struct sockaddr*)sockAddr.Release();

        Info_.reset(hints.release());
    }

    inline struct addrinfo* Info() const noexcept {
        return Info_.get();
    }

private:
    using TAddrInfoPtr = std::unique_ptr<struct addrinfo, TAddrInfoDeleter>;

    TAddrInfoPtr Info_;
};

TNetworkAddress::TNetworkAddress(const TUnixSocketPath& unixSocketPath, int flags)
    : Impl_(new TImpl(unixSocketPath.Path.data(), flags))
{
}

TNetworkAddress::TNetworkAddress(const TString& host, ui16 port, int flags)
    : Impl_(new TImpl(host.data(), port, flags))
{
}

TNetworkAddress::TNetworkAddress(const TString& host, ui16 port)
    : Impl_(new TImpl(host.data(), port, 0))
{
}

TNetworkAddress::TNetworkAddress(ui16 port)
    : Impl_(new TImpl(nullptr, port, 0))
{
}

TNetworkAddress::~TNetworkAddress() = default;

struct addrinfo* TNetworkAddress::Info() const noexcept {
    return Impl_->Info();
}

TNetworkResolutionError::TNetworkResolutionError(int error) {
    const char* errMsg = nullptr;
#ifdef _win_
    errMsg = LastSystemErrorText(error); // gai_strerror is not thread-safe on Windows
#else
    errMsg = gai_strerror(error);
#endif
    (*this) << errMsg << "(" << error;

#if defined(_unix_)
    if (error == EAI_SYSTEM) {
        (*this) << "; errno=" << LastSystemError();
    }
#endif

    (*this) << "): ";
}

#if defined(_unix_)
static inline int GetFlags(int fd) {
    const int ret = fcntl(fd, F_GETFL);

    if (ret == -1) {
        ythrow TSystemError() << "can not get fd flags";
    }

    return ret;
}

static inline void SetFlags(int fd, int flags) {
    if (fcntl(fd, F_SETFL, flags) == -1) {
        ythrow TSystemError() << "can not set fd flags";
    }
}

static inline void EnableFlag(int fd, int flag) {
    const int oldf = GetFlags(fd);
    const int newf = oldf | flag;

    if (oldf != newf) {
        SetFlags(fd, newf);
    }
}

static inline void DisableFlag(int fd, int flag) {
    const int oldf = GetFlags(fd);
    const int newf = oldf & (~flag);

    if (oldf != newf) {
        SetFlags(fd, newf);
    }
}

static inline void SetFlag(int fd, int flag, bool value) {
    if (value) {
        EnableFlag(fd, flag);
    } else {
        DisableFlag(fd, flag);
    }
}

static inline bool FlagsAreEnabled(int fd, int flags) {
    return GetFlags(fd) & flags;
}
#endif

#if defined(_win_)
static inline void SetNonBlockSocket(SOCKET fd, int value) {
    unsigned long inbuf = value;
    unsigned long outbuf = 0;
    DWORD written = 0;

    if (!inbuf) {
        WSAEventSelect(fd, nullptr, 0);
    }

    if (WSAIoctl(fd, FIONBIO, &inbuf, sizeof(inbuf), &outbuf, sizeof(outbuf), &written, 0, 0) == SOCKET_ERROR) {
        ythrow TSystemError() << "can not set non block socket state";
    }
}

static inline bool IsNonBlockSocket(SOCKET fd) {
    unsigned long buf = 0;

    if (WSAIoctl(fd, FIONBIO, 0, 0, &buf, sizeof(buf), 0, 0, 0) == SOCKET_ERROR) {
        ythrow TSystemError() << "can not get non block socket state";
    }

    return buf;
}
#endif

void SetNonBlock(SOCKET fd, bool value) {
#if defined(_unix_)
    #if defined(FIONBIO)
    Y_UNUSED(SetFlag); // shut up clang about unused function
    int nb = value;

    if (ioctl(fd, FIONBIO, &nb) < 0) {
        ythrow TSystemError() << "ioctl failed";
    }
    #else
    SetFlag(fd, O_NONBLOCK, value);
    #endif
#elif defined(_win_)
    SetNonBlockSocket(fd, value);
#else
    #error todo
#endif
}

bool IsNonBlock(SOCKET fd) {
#if defined(_unix_)
    return FlagsAreEnabled(fd, O_NONBLOCK);
#elif defined(_win_)
    return IsNonBlockSocket(fd);
#else
    #error todo
#endif
}

void SetDeferAccept(SOCKET s) {
    (void)s;

#if defined(TCP_DEFER_ACCEPT)
    CheckedSetSockOpt(s, IPPROTO_TCP, TCP_DEFER_ACCEPT, 10, "defer accept");
#endif

#if defined(SO_ACCEPTFILTER)
    struct accept_filter_arg afa;

    Zero(afa);
    strcpy(afa.af_name, "dataready");
    SetSockOpt(s, SOL_SOCKET, SO_ACCEPTFILTER, afa);
#endif
}

ssize_t PollD(struct pollfd fds[], nfds_t nfds, const TInstant& deadLine) noexcept {
    TInstant now = TInstant::Now();

    do {
        const TDuration toWait = PollStep(deadLine, now);
        const int res = poll(fds, nfds, MicroToMilli(toWait.MicroSeconds()));

        if (res > 0) {
            return res;
        }

        if (res < 0) {
            const int err = LastSystemError();

            if (err != ETIMEDOUT && err != EINTR) {
                return -err;
            }
        }
    } while ((now = TInstant::Now()) < deadLine);

    return -ETIMEDOUT;
}

void ShutDown(SOCKET s, int mode) {
    if (shutdown(s, mode)) {
        ythrow TSystemError() << "shutdown socket error";
    }
}
