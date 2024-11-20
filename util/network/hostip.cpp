#include "socket.h"
#include "hostip.h"

#include <util/system/defaults.h>
#include <util/system/byteorder.h>

#if defined(_unix_) || defined(_cygwin_)
    #include <netdb.h>
#endif

#if !defined(BIND_LIB)
    #if !defined(__FreeBSD__) && !defined(_win32_) && !defined(_cygwin_)
        #define AGENT_USE_GETADDRINFO
    #endif

    #if defined(__FreeBSD__)
        #define AGENT_USE_GETADDRINFO
    #endif
#endif

int NResolver::GetHostIP(const char* hostname, ui32* ip, size_t* slots) {
    size_t i = 0;
    size_t ipsFound = 0;

#ifdef AGENT_USE_GETADDRINFO
    int ret = 0;
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* gai_res = nullptr;
    int gai_ret = getaddrinfo(hostname, nullptr, &hints, &gai_res);
    if (gai_ret == 0 && gai_res->ai_addr) {
        struct addrinfo* cur = gai_res;
        for (i = 0; i < *slots && cur; i++, cur = cur->ai_next, ipsFound++) {
            ip[i] = *(ui32*)(&((sockaddr_in*)(cur->ai_addr))->sin_addr);
        }
    } else {
        if (gai_ret == EAI_NONAME || gai_ret == EAI_SERVICE) {
            ret = HOST_NOT_FOUND;
        } else {
            ret = GetDnsError();
        }
    }

    if (gai_res) {
        freeaddrinfo(gai_res);
    }

    if (ret) {
        return ret;
    }
#else
    hostent* hostent = gethostbyname(hostname);

    if (!hostent) {
        return GetDnsError();
    }

    if (hostent->h_addrtype != AF_INET || (unsigned)hostent->h_length < sizeof(ui32)) {
        return HOST_NOT_FOUND;
    }

    char** cur = hostent->h_addr_list;
    for (i = 0; i < *slots && *cur; i++, cur++, ipsFound++) {
        ip[i] = *(ui32*)*cur;
    }
#endif
    for (i = 0; i < ipsFound; i++) {
        ip[i] = InetToHost(ip[i]);
    }
    *slots = ipsFound;

    return 0;
}

int NResolver::GetDnsError() {
    return h_errno;
}
