#pragma once

#include <util/digest/murmur.h>

#include <util/network/ip.h>

#include <util/str_stl.h>
#include <util/generic/maybe.h>
#include <util/generic/variant.h>

#ifdef _unix_
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <arpa/inet.h>
#endif // _unix_

#include <string.h>

#ifndef INET6_ADDRSTRLEN
    #define INET6_ADDRSTRLEN 46
#endif

// Network (big-endian) byte order
using TIp4 = TIpHost;

// Network (big-endian) byte order
struct TIp6 {
    char Data[16];

    bool operator==(const TIp6& rhs) const {
        return memcmp(Data, rhs.Data, sizeof(Data)) == 0;
    }

    bool operator<(const TIp6& rhs) const {
        return memcmp(Data, rhs.Data, sizeof(Data)) < 0;
    }
};

template <>
struct THash<TIp6> {
    inline size_t operator()(const TIp6& ip) const {
        return MurmurHash<size_t>((const void*)ip.Data, 16);
    }
};

static inline TIp6 Ip6FromIp4(TIp4 addr) {
    TIp6 res;
    memset(res.Data, 0, sizeof(res.Data));
    res.Data[10] = '\xFF';
    res.Data[11] = '\xFF';
    memcpy(res.Data + 12, &addr, 4);
    return res;
}

static inline TIp6 Ip6FromString(const char* ipStr) {
    TIp6 res;

    if (inet_pton(AF_INET6, ipStr, &res.Data) == 0) {
        ythrow TSystemError() << "Failed to convert (" << ipStr << ") to ipv6 address";
    }

    return res;
}

static inline TMaybe<TIp6> TryParseIp6FromString(const char* ipStr) {
    TIp6 res;

    if (inet_pton(AF_INET6, ipStr, &res.Data) == 0) {
        return Nothing();
    }

    return res;
}

static inline char* Ip6ToString(const TIp6& ip, char* buf, size_t len) {
    if (!inet_ntop(AF_INET6, (void*)&ip.Data, buf, (socklen_t)len)) {
        ythrow TSystemError() << "Failed to get ipv6 address string";
    }

    return buf;
}

static inline TString Ip6ToString(const TIp6& ip) {
    char buf[INET6_ADDRSTRLEN];

    return TString(Ip6ToString(ip, buf, sizeof(buf)));
}

template <>
inline void Out<TIp6>(IOutputStream& os, const TIp6& a) {
    os << Ip6ToString(a);
}

using TIp4Or6 = std::variant<TIp4, TIp6>;

static inline TIp4Or6 Ip4Or6FromString(const char* ipStr) {
    const char* c = ipStr;
    for (; *c; ++c) {
        if (*c == '.') {
            return IpFromString(ipStr);
        }
        if (*c == ':') {
            return Ip6FromString(ipStr);
        }
    }
    ythrow TSystemError() << "Failed to convert (" << ipStr << ") to ipv4 or ipv6 address";
}

static inline TString Ip4Or6ToString(const TIp4Or6& ip) {
    if (std::holds_alternative<TIp6>(ip)) {
        return Ip6ToString(std::get<TIp6>(ip));
    } else {
        return IpToString(std::get<TIp4>(ip));
    }
}

// for TIp4 or TIp6, not TIp4Or6
template <class TIp>
struct TIpCompare {
    bool Less(const TIp& l, const TIp& r) const {
        return memcmp(&l, &r, sizeof(TIp)) < 0;
    }

    bool LessEqual(const TIp& l, const TIp& r) const {
        return memcmp(&l, &r, sizeof(TIp)) <= 0;
    }

    bool operator()(const TIp& l, const TIp& r) const {
        return Less(l, r);
    }
};
