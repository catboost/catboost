#pragma once

#include "iostatus.h"

#include <util/datetime/base.h>
#include <util/network/init.h>
#include <util/network/iovec.h>
#include <util/network/nonblock.h>
#include <util/network/socket.h>

class TCont;

namespace NCoro {

    int SelectD(TCont* cont, SOCKET fds[], int what[], size_t nfds, SOCKET* outfd, TInstant deadline) noexcept;
    int SelectT(TCont* cont, SOCKET fds[], int what[], size_t nfds, SOCKET* outfd, TDuration timeout) noexcept;
    int SelectI(TCont* cont, SOCKET fds[], int what[], size_t nfds, SOCKET* outfd);

    int PollD(TCont* cont, SOCKET fd, int what, TInstant deadline) noexcept;
    int PollT(TCont* cont, SOCKET fd, int what, TDuration timeout) noexcept;
    int PollI(TCont* cont, SOCKET fd, int what) noexcept;

    TContIOStatus ReadVectorD(TCont* cont, SOCKET fd, TContIOVector* vec, TInstant deadline) noexcept;
    TContIOStatus ReadVectorT(TCont* cont, SOCKET fd, TContIOVector* vec, TDuration timeOut) noexcept;
    TContIOStatus ReadVectorI(TCont* cont, SOCKET fd, TContIOVector* vec) noexcept;

    TContIOStatus ReadD(TCont* cont, SOCKET fd, void* buf, size_t len, TInstant deadline) noexcept;
    TContIOStatus ReadT(TCont* cont, SOCKET fd, void* buf, size_t len, TDuration timeout) noexcept;
    TContIOStatus ReadI(TCont* cont, SOCKET fd, void* buf, size_t len) noexcept;

    TContIOStatus WriteVectorD(TCont* cont, SOCKET fd, TContIOVector* vec, TInstant deadline) noexcept;
    TContIOStatus WriteVectorT(TCont* cont, SOCKET fd, TContIOVector* vec, TDuration timeOut) noexcept;
    TContIOStatus WriteVectorI(TCont* cont, SOCKET fd, TContIOVector* vec) noexcept;

    TContIOStatus WriteD(TCont* cont, SOCKET fd, const void* buf, size_t len, TInstant deadline) noexcept;
    TContIOStatus WriteT(TCont* cont, SOCKET fd, const void* buf, size_t len, TDuration timeout) noexcept;
    TContIOStatus WriteI(TCont* cont, SOCKET fd, const void* buf, size_t len) noexcept;

    int ConnectD(TCont* cont, TSocketHolder& s, const struct addrinfo& ai, TInstant deadline) noexcept;

    int ConnectD(TCont* cont, TSocketHolder& s, const TNetworkAddress& addr, TInstant deadline) noexcept;
    int ConnectT(TCont* cont, TSocketHolder& s, const TNetworkAddress& addr, TDuration timeout) noexcept;
    int ConnectI(TCont* cont, TSocketHolder& s, const TNetworkAddress& addr) noexcept;

    int ConnectD(TCont* cont, SOCKET s, const struct sockaddr* name, socklen_t namelen, TInstant deadline) noexcept;
    int ConnectT(TCont* cont, SOCKET s, const struct sockaddr* name, socklen_t namelen, TDuration timeout) noexcept;
    int ConnectI(TCont* cont, SOCKET s, const struct sockaddr* name, socklen_t namelen) noexcept;

    int AcceptD(TCont* cont, SOCKET s, struct sockaddr* addr, socklen_t* addrlen, TInstant deadline) noexcept;
    int AcceptT(TCont* cont, SOCKET s, struct sockaddr* addr, socklen_t* addrlen, TDuration timeout) noexcept;
    int AcceptI(TCont* cont, SOCKET s, struct sockaddr* addr, socklen_t* addrlen) noexcept;

    SOCKET Socket(int domain, int type, int protocol) noexcept;
    SOCKET Socket(const struct addrinfo& ai) noexcept;
}
