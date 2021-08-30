#pragma once

#include <stddef.h>
#include <BaseTsd.h>
#include <WinSock2.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IOV_MAX INT_MAX

typedef SSIZE_T ssize_t;

struct iovec {
    char* iov_base;
    size_t iov_len;
};

ssize_t readv(SOCKET sock, struct iovec const* iov, int nvecs);
ssize_t writev(SOCKET sock, struct iovec const* iov, int nvecs);

#ifdef __cplusplus
}
#endif
