#include "pair.h"

int SocketPair(SOCKET socks[2], bool overlapped, bool cloexec) {
#if defined(_win_)
    struct sockaddr_in addr;
    SOCKET listener;
    int e;
    int addrlen = sizeof(addr);
    DWORD flags = (overlapped ? WSA_FLAG_OVERLAPPED : 0) | (cloexec ? WSA_FLAG_NO_HANDLE_INHERIT : 0);

    if (socks == 0) {
        WSASetLastError(WSAEINVAL);

        return SOCKET_ERROR;
    }

    socks[0] = INVALID_SOCKET;
    socks[1] = INVALID_SOCKET;

    if ((listener = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        return SOCKET_ERROR;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(0x7f000001);
    addr.sin_port = 0;

    e = bind(listener, (const struct sockaddr*)&addr, sizeof(addr));

    if (e == SOCKET_ERROR) {
        e = WSAGetLastError();
        closesocket(listener);
        WSASetLastError(e);

        return SOCKET_ERROR;
    }

    e = getsockname(listener, (struct sockaddr*)&addr, &addrlen);

    if (e == SOCKET_ERROR) {
        e = WSAGetLastError();
        closesocket(listener);
        WSASetLastError(e);

        return SOCKET_ERROR;
    }

    do {
        if (listen(listener, 1) == SOCKET_ERROR) {
            break;
        }

        if ((socks[0] = WSASocket(AF_INET, SOCK_STREAM, 0, nullptr, 0, flags)) == INVALID_SOCKET) {
            break;
        }

        if (connect(socks[0], (const struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
            break;
        }

        if ((socks[1] = accept(listener, nullptr, nullptr)) == INVALID_SOCKET) {
            break;
        }

        closesocket(listener);

        return 0;
    } while (0);

    e = WSAGetLastError();
    closesocket(listener);
    closesocket(socks[0]);
    closesocket(socks[1]);
    WSASetLastError(e);

    return SOCKET_ERROR;
#else
    (void)overlapped;

    #if defined(_linux_)
    return socketpair(AF_LOCAL, SOCK_STREAM | (cloexec ? SOCK_CLOEXEC : 0), 0, socks);
    #else
    int r = socketpair(AF_LOCAL, SOCK_STREAM, 0, socks);
    // Non-atomic wrt exec
    if (r == 0 && cloexec) {
        for (int i = 0; i < 2; ++i) {
            int flags = fcntl(socks[i], F_GETFD, 0);
            if (flags < 0) {
                return flags;
            }
            r = fcntl(socks[i], F_SETFD, flags | FD_CLOEXEC);
            if (r < 0) {
                return r;
            }
        }
    }
    return r;
    #endif
#endif
}
