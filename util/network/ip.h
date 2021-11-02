#pragma once

#include "socket.h"
#include "hostip.h"

#include <util/system/error.h>
#include <util/system/byteorder.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>

/// IPv4 address in network format
using TIpHost = ui32;

/// Port number in host format
using TIpPort = ui16;

/*
 * ipStr is in 'ddd.ddd.ddd.ddd' format
 * returns IPv4 address in inet format
 */
static inline TIpHost IpFromString(const char* ipStr) {
    in_addr ia;

    if (inet_aton(ipStr, &ia) == 0) {
        ythrow TSystemError() << "Failed to convert (" << ipStr << ") to ip address";
    }

    return (ui32)ia.s_addr;
}

static inline char* IpToString(TIpHost ip, char* buf, size_t len) {
    if (!inet_ntop(AF_INET, (void*)&ip, buf, (socklen_t)len)) {
        ythrow TSystemError() << "Failed to get ip address string";
    }

    return buf;
}

static inline TString IpToString(TIpHost ip) {
    char buf[INET_ADDRSTRLEN];

    return TString(IpToString(ip, buf, sizeof(buf)));
}

static inline TIpHost ResolveHost(const char* data, size_t len) {
    TIpHost ret;
    const TString s(data, len);

    if (NResolver::GetHostIP(s.data(), &ret) != 0) {
        ythrow TSystemError(NResolver::GetDnsError()) << "can not resolve(" << s << ")";
    }

    return HostToInet(ret);
}

/// socket address
struct TIpAddress: public sockaddr_in {
    inline TIpAddress() noexcept {
        Clear();
    }

    inline TIpAddress(const sockaddr_in& addr) noexcept
        : sockaddr_in(addr)
        , tmp(0)
    {
    }

    inline TIpAddress(TIpHost ip, TIpPort port) noexcept {
        Set(ip, port);
    }

    inline TIpAddress(TStringBuf ip, TIpPort port) {
        Set(ResolveHost(ip.data(), ip.size()), port);
    }

    inline TIpAddress(const char* ip, TIpPort port) {
        Set(ResolveHost(ip, strlen(ip)), port);
    }

    inline operator sockaddr*() const noexcept {
        return (sockaddr*)(sockaddr_in*)this;
    }

    inline operator socklen_t*() const noexcept {
        tmp = sizeof(sockaddr_in);

        return (socklen_t*)&tmp;
    }

    inline operator socklen_t() const noexcept {
        tmp = sizeof(sockaddr_in);

        return tmp;
    }

    inline void Clear() noexcept {
        Zero((sockaddr_in&)(*this));
    }

    inline void Set(TIpHost ip, TIpPort port) noexcept {
        Clear();

        sin_family = AF_INET;
        sin_addr.s_addr = ip;
        sin_port = HostToInet(port);
    }

    inline TIpHost Host() const noexcept {
        return sin_addr.s_addr;
    }

    inline TIpPort Port() const noexcept {
        return InetToHost(sin_port);
    }

private:
    // required for "operator socklen_t*()"
    mutable socklen_t tmp;
};
