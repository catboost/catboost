#include <contrib/libs/libc_compat/include/windows/sys/uio.h>

#include <Windows.h>
#include <winsock2.h>
#include <malloc.h>

ssize_t readv(SOCKET sock, struct iovec const* iov, int iovcnt) {
    WSABUF* wsabuf = (WSABUF*)alloca(iovcnt * sizeof(WSABUF));
    for (int i = 0; i < iovcnt; ++i) {
        wsabuf[i].buf = iov[i].iov_base;
        wsabuf[i].len = (u_long)iov[i].iov_len;
    }
    DWORD numberOfBytesRecv;
    DWORD flags = 0;
    int res = WSARecv(sock, wsabuf, iovcnt, &numberOfBytesRecv, &flags, NULL, NULL);
    if (res == SOCKET_ERROR) {
        errno = EIO;
        return -1;
    }
    return numberOfBytesRecv;
}

ssize_t writev(SOCKET sock, struct iovec const* iov, int iovcnt) {
    WSABUF* wsabuf = (WSABUF*)alloca(iovcnt * sizeof(WSABUF));
    for (int i = 0; i < iovcnt; ++i) {
        wsabuf[i].buf = iov[i].iov_base;
        wsabuf[i].len = (u_long)iov[i].iov_len;
    }
    DWORD numberOfBytesSent;
    int res = WSASend(sock, wsabuf, iovcnt, &numberOfBytesSent, 0, NULL, NULL);
    if (res == SOCKET_ERROR) {
        errno = EIO;
        return -1;
    }
    return numberOfBytesSent;
}
