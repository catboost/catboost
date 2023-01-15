#include "netloss.h"

#include <util/system/yassert.h>

#ifdef _linux_
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#endif

#include <stdlib.h>
#include <errno.h>

#if defined(__linux__)
#ifndef TCP_TXCNT_ENABLE
#define TCP_TXCNT_ENABLE INT_MAX
#endif
#ifndef TCP_RXCNT
#define TCP_RXCNT (INT_MAX - 1)
#endif

ui32 GetRXPacketsLostCounter(int sock) noexcept {
    unsigned value;
    socklen_t length = sizeof(value);
    if (getsockopt(sock, SOL_TCP, TCP_RXCNT, &value, &length) != 0 || length != sizeof(value)) {
        return 0;
    } else {
        return value;
    }
}

i32 GetNumRXPacketsLost(int sock, ui32 initialValue) noexcept {
    unsigned value;
    socklen_t length = sizeof(value);
    if (getsockopt(sock, SOL_TCP, TCP_RXCNT, &value, &length) == 0 && length == sizeof(value)) {
        if (value < initialValue) {
            return (i32)(value - initialValue) * 10;
        }
        return (i32)(value - initialValue);
    }
    return -1;
}

void EnableTXPacketsCounter(int sock) noexcept {
    int value = 1;
    // Ignore the possible errors
    setsockopt(sock, SOL_TCP, TCP_TXCNT_ENABLE, &value, sizeof(value));
}
#else
ui32 GetRXPacketsLostCounter(int) noexcept {
    return 0;
}
i32 GetNumRXPacketsLost(int, ui32) noexcept {
    return -1;
}
void EnableTXPacketsCounter(int) noexcept {
}
#endif
