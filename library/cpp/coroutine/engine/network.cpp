#include "impl.h"
#include "network.h"

#include <util/generic/scope.h>
#include <util/generic/xrange.h>

#include <sys/uio.h>

#if defined(_bionic_)
#   define IOV_MAX 1024
#endif


namespace NCoro {
    namespace {
        bool IsBlocked(int lasterr) noexcept {
            return lasterr == EAGAIN || lasterr == EWOULDBLOCK;
        }

        ssize_t DoReadVector(SOCKET fd, TContIOVector* vec) noexcept {
            return readv(fd, (const iovec*) vec->Parts(), Min(IOV_MAX, (int) vec->Count()));
        }

        ssize_t DoWriteVector(SOCKET fd, TContIOVector* vec) noexcept {
            return writev(fd, (const iovec*) vec->Parts(), Min(IOV_MAX, (int) vec->Count()));
        }
    }


    int SelectD(TCont* cont, SOCKET fds[], int what[], size_t nfds, SOCKET* outfd, TInstant deadline) noexcept {
        if (cont->Cancelled()) {
            return ECANCELED;
        }

        if (nfds == 0) {
            return 0;
        }

        TTempArray<TFdEvent> events(nfds);

        for (auto i : xrange(nfds)) {
            new(events.Data() + i) TFdEvent(cont, fds[i], (ui16) what[i], deadline);
        }

        Y_DEFER {
            for (auto i : xrange(nfds)) {
                (events.Data() + i)->~TFdEvent();
            }
        };

        for (auto i : xrange(nfds)) {
            cont->Executor()->ScheduleIoWait(events.Data() + i);
        }
        cont->Switch();

        if (cont->Cancelled()) {
            return ECANCELED;
        }

        TFdEvent* ret = nullptr;
        int status = EINPROGRESS;

        for (auto i : xrange(nfds)) {
            auto& ev = *(events.Data() + i);
            switch (ev.Status()) {
            case EINPROGRESS:
                break;
            case ETIMEDOUT:
                if (status != EINPROGRESS) {
                    break;
                }
                [[fallthrough]];
            default:
                status = ev.Status();
                ret = &ev;
            }
        }

        if (ret) {
            if (outfd) {
                *outfd = ret->Fd();
            }
            return ret->Status();
        }

        return EINPROGRESS;
    }

    int SelectT(TCont* cont, SOCKET fds[], int what[], size_t nfds, SOCKET* outfd, TDuration timeout) noexcept {
        return SelectD(cont, fds, what, nfds, outfd, timeout.ToDeadLine());
    }

    int SelectI(TCont* cont, SOCKET fds[], int what[], size_t nfds, SOCKET* outfd) {
        return SelectD(cont, fds, what, nfds, outfd, TInstant::Max());
    }


    int PollD(TCont* cont, SOCKET fd, int what, TInstant deadline) noexcept {
        TFdEvent event(cont, fd, (ui16)what, deadline);
        return ExecuteEvent(&event);
    }

    int PollT(TCont* cont, SOCKET fd, int what, TDuration timeout) noexcept {
        return PollD(cont, fd, what, timeout.ToDeadLine());
    }

    int PollI(TCont* cont, SOCKET fd, int what) noexcept {
        return PollD(cont, fd, what, TInstant::Max());
    }


    TContIOStatus ReadVectorD(TCont* cont, SOCKET fd, TContIOVector* vec, TInstant deadline) noexcept {
        while (true) {
            ssize_t res = DoReadVector(fd, vec);

            if (res >= 0) {
                return TContIOStatus::Success((size_t) res);
            }

            {
                const int err = LastSystemError();

                if (!IsBlocked(err)) {
                    return TContIOStatus::Error(err);
                }
            }

            if ((res = PollD(cont, fd, CONT_POLL_READ, deadline)) != 0) {
                return TContIOStatus::Error((int) res);
            }
        }
    }

    TContIOStatus ReadVectorT(TCont* cont, SOCKET fd, TContIOVector* vec, TDuration timeOut) noexcept {
        return ReadVectorD(cont, fd, vec, timeOut.ToDeadLine());
    }

    TContIOStatus ReadVectorI(TCont* cont, SOCKET fd, TContIOVector* vec) noexcept {
        return ReadVectorD(cont, fd, vec, TInstant::Max());
    }


    TContIOStatus ReadD(TCont* cont, SOCKET fd, void* buf, size_t len, TInstant deadline) noexcept {
        IOutputStream::TPart part(buf, len);
        TContIOVector vec(&part, 1);
        return ReadVectorD(cont, fd, &vec, deadline);
    }

    TContIOStatus ReadT(TCont* cont, SOCKET fd, void* buf, size_t len, TDuration timeout) noexcept {
        return ReadD(cont, fd, buf, len, timeout.ToDeadLine());
    }

    TContIOStatus ReadI(TCont* cont, SOCKET fd, void* buf, size_t len) noexcept {
        return ReadD(cont, fd, buf, len, TInstant::Max());
    }


    TContIOStatus WriteVectorD(TCont* cont, SOCKET fd, TContIOVector* vec, TInstant deadline) noexcept {
        size_t written = 0;

        while (!vec->Complete()) {
            ssize_t res = DoWriteVector(fd, vec);

            if (res >= 0) {
                written += res;

                vec->Proceed((size_t) res);
            } else {
                {
                    const int err = LastSystemError();

                    if (!IsBlocked(err)) {
                        return TContIOStatus(written, err);
                    }
                }

                if ((res = PollD(cont, fd, CONT_POLL_WRITE, deadline)) != 0) {
                    return TContIOStatus(written, (int) res);
                }
            }
        }

        return TContIOStatus::Success(written);
    }

    TContIOStatus WriteVectorT(TCont* cont, SOCKET fd, TContIOVector* vec, TDuration timeOut) noexcept {
        return WriteVectorD(cont, fd, vec, timeOut.ToDeadLine());
    }

    TContIOStatus WriteVectorI(TCont* cont, SOCKET fd, TContIOVector* vec) noexcept {
        return WriteVectorD(cont, fd, vec, TInstant::Max());
    }


    TContIOStatus WriteD(TCont* cont, SOCKET fd, const void* buf, size_t len, TInstant deadline) noexcept {
        IOutputStream::TPart part(buf, len);
        TContIOVector vec(&part, 1);
        return WriteVectorD(cont, fd, &vec, deadline);
    }

    TContIOStatus WriteT(TCont* cont, SOCKET fd, const void* buf, size_t len, TDuration timeout) noexcept {
        return WriteD(cont, fd, buf, len, timeout.ToDeadLine());
    }

    TContIOStatus WriteI(TCont* cont, SOCKET fd, const void* buf, size_t len) noexcept {
        return WriteD(cont, fd, buf, len, TInstant::Max());
    }


    int ConnectD(TCont* cont, TSocketHolder& s, const struct addrinfo& ai, TInstant deadline) noexcept {
        TSocketHolder res(Socket(ai));

        if (res.Closed()) {
            return LastSystemError();
        }

        const int ret = ConnectD(cont, res, ai.ai_addr, (socklen_t) ai.ai_addrlen, deadline);

        if (!ret) {
            s.Swap(res);
        }

        return ret;
    }

    int ConnectD(TCont* cont, TSocketHolder& s, const TNetworkAddress& addr, TInstant deadline) noexcept {
        int ret = EHOSTUNREACH;

        for (auto it = addr.Begin(); it != addr.End(); ++it) {
            ret = ConnectD(cont, s, *it, deadline);

            if (ret == 0 || ret == ETIMEDOUT) {
                return ret;
            }
        }

        return ret;
    }

    int ConnectT(TCont* cont, TSocketHolder& s, const TNetworkAddress& addr, TDuration timeout) noexcept {
        return ConnectD(cont, s, addr, timeout.ToDeadLine());
    }

    int ConnectI(TCont* cont, TSocketHolder& s, const TNetworkAddress& addr) noexcept {
        return ConnectD(cont, s, addr, TInstant::Max());
    }

    int ConnectD(TCont* cont, SOCKET s, const struct sockaddr* name, socklen_t namelen, TInstant deadline) noexcept {
        if (connect(s, name, namelen)) {
            const int err = LastSystemError();

            if (!IsBlocked(err) && err != EINPROGRESS) {
                return err;
            }

            int ret = PollD(cont, s, CONT_POLL_WRITE, deadline);

            if (ret) {
                return ret;
            }

            // check if we really connected
            // FIXME: Unportable ??
            int serr = 0;
            socklen_t slen = sizeof(serr);

            ret = getsockopt(s, SOL_SOCKET, SO_ERROR, (char*) &serr, &slen);

            if (ret) {
                return LastSystemError();
            }

            if (serr) {
                return serr;
            }
        }

        return 0;
    }

    int ConnectT(TCont* cont, SOCKET s, const struct sockaddr* name, socklen_t namelen, TDuration timeout) noexcept {
        return ConnectD(cont, s, name, namelen, timeout.ToDeadLine());
    }

    int ConnectI(TCont* cont, SOCKET s, const struct sockaddr* name, socklen_t namelen) noexcept {
        return ConnectD(cont, s, name, namelen, TInstant::Max());
    }


    int AcceptD(TCont* cont, SOCKET s, struct sockaddr* addr, socklen_t* addrlen, TInstant deadline) noexcept {
        SOCKET ret;

        while ((ret = Accept4(s, addr, addrlen)) == INVALID_SOCKET) {
            int err = LastSystemError();

            if (!IsBlocked(err)) {
                return -err;
            }

            err = PollD(cont, s, CONT_POLL_READ, deadline);

            if (err) {
                return -err;
            }
        }

        return (int) ret;
    }

    int AcceptT(TCont* cont, SOCKET s, struct sockaddr* addr, socklen_t* addrlen, TDuration timeout) noexcept {
        return AcceptD(cont, s, addr, addrlen, timeout.ToDeadLine());
    }

    int AcceptI(TCont* cont, SOCKET s, struct sockaddr* addr, socklen_t* addrlen) noexcept {
        return AcceptD(cont, s, addr, addrlen, TInstant::Max());
    }

    SOCKET Socket(int domain, int type, int protocol) noexcept {
        return Socket4(domain, type, protocol);
    }

    SOCKET Socket(const struct addrinfo& ai) noexcept {
        return Socket(ai.ai_family, ai.ai_socktype, ai.ai_protocol);
    }
}
