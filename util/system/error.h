#pragma once

#include "defaults.h"

#if defined(_win_)
    #include <winerror.h>
    #include <errno.h>

    #undef E_FAIL
    #undef ERROR_TIMEOUT

    #if defined(_MSC_VER)
        #undef EADDRINUSE
        #undef EADDRNOTAVAIL
        #undef EAFNOSUPPORT
        #undef EALREADY
        #undef ECANCELED
        #undef ECONNABORTED
        #undef ECONNREFUSED
        #undef ECONNRESET
        #undef EDESTADDRREQ
        #undef EHOSTUNREACH
        #undef EINPROGRESS
        #undef EISCONN
        #undef ELOOP
        #undef EMSGSIZE
        #undef ENETDOWN
        #undef ENETRESET
        #undef ENETUNREACH
        #undef ENOBUFS
        #undef ENOPROTOOPT
        #undef ENOTCONN
        #undef ENOTSOCK
        #undef EOPNOTSUPP
        #undef EPROTONOSUPPORT
        #undef EPROTOTYPE
        #undef ETIMEDOUT
        #undef EWOULDBLOCK
        #undef ENAMETOOLONG
        #undef ENOTEMPTY

        #define EWOULDBLOCK WSAEWOULDBLOCK
        #define EINPROGRESS WSAEINPROGRESS
        #define EALREADY WSAEALREADY
        #define ENOTSOCK WSAENOTSOCK
        #define EDESTADDRREQ WSAEDESTADDRREQ
        #define EMSGSIZE WSAEMSGSIZE
        #define EPROTOTYPE WSAEPROTOTYPE
        #define ENOPROTOOPT WSAENOPROTOOPT
        #define EPROTONOSUPPORT WSAEPROTONOSUPPORT
        #define ESOCKTNOSUPPORT WSAESOCKTNOSUPPORT
        #define EOPNOTSUPP WSAEOPNOTSUPP
        #define EPFNOSUPPORT WSAEPFNOSUPPORT
        #define EAFNOSUPPORT WSAEAFNOSUPPORT
        #define EADDRINUSE WSAEADDRINUSE
        #define EADDRNOTAVAIL WSAEADDRNOTAVAIL
        #define ENETDOWN WSAENETDOWN
        #define ENETUNREACH WSAENETUNREACH
        #define ENETRESET WSAENETRESET
        #define ECONNABORTED WSAECONNABORTED
        #define ECONNRESET WSAECONNRESET
        #define ENOBUFS WSAENOBUFS
        #define EISCONN WSAEISCONN
        #define ENOTCONN WSAENOTCONN
        #define ESHUTDOWN WSAESHUTDOWN
        #define ETOOMANYREFS WSAETOOMANYREFS
        #define ETIMEDOUT WSAETIMEDOUT
        #define ECONNREFUSED WSAECONNREFUSED
        #define ELOOP WSAELOOP
        #define ENAMETOOLONG WSAENAMETOOLONG
        #define EHOSTDOWN WSAEHOSTDOWN
        #define EHOSTUNREACH WSAEHOSTUNREACH
        #define ENOTEMPTY WSAENOTEMPTY
        #define EPROCLIM WSAEPROCLIM
        #define EUSERS WSAEUSERS
        #define ESTALE WSAESTALE
        #define EREMOTE WSAEREMOTE
        #define ECANCELED WSAECANCELLED
    #endif

    #define EDQUOT WSAEDQUOT
#endif

void ClearLastSystemError();
int LastSystemError();
void LastSystemErrorText(char* str, size_t size, int code);
const char* LastSystemErrorText(int code);

inline const char* LastSystemErrorText() {
    return LastSystemErrorText(LastSystemError());
}

inline void LastSystemErrorText(char* str, size_t size) {
    LastSystemErrorText(str, size, LastSystemError());
}
