#include "stdafx.h"
#include "udp_address.h"

#include <util/system/mutex.h>
#include <util/system/spinlock.h>
#include <util/draft/ip.h>

#ifdef _win_
#include <iphlpapi.h>
#pragma comment(lib, "Iphlpapi.lib")
#else
#include <ifaddrs.h>
#endif

namespace NNetliba_v12 {
    bool IsValidIPv6(const char* str) {
        const char* cur = str;
        while (*cur && *cur != '%')
            ++cur;
        TString tmp(str, cur);
        return TryParseIp6FromString(tmp.c_str()).Defined();
    }

    static bool ParseInetName(TUdpAddress* pRes, const char* name, int nDefaultPort, EUdpAddressType addressType) {
        int nPort = nDefaultPort;

        TString host;
        if (name[0] == '[') {
            ++name;
            const char* nameFin = name;
            for (; *nameFin; ++nameFin) {
                if (nameFin[0] == ']')
                    break;
            }
            host.assign(name, nameFin);
            Y_ASSERT(IsValidIPv6(host.c_str()));
            name = *nameFin ? nameFin + 1 : nameFin;
            if (name[0] == ':') {
                char* endPtr = nullptr;
                nPort = strtol(name + 1, &endPtr, 10);
                if (!endPtr || *endPtr != '\0')
                    return false;
            }
        } else {
            host = name;
            if (!IsValidIPv6(name)) {
                size_t nIdx = host.find(':');
                if (nIdx != (size_t)TString::npos) {
                    const char* pszPort = host.c_str() + nIdx + 1;
                    char* endPtr = nullptr;
                    nPort = strtol(pszPort, &endPtr, 10);
                    if (!endPtr || *endPtr != '\0')
                        return false;
                    host.resize(nIdx);
                }
            }
        }

        addrinfo aiHints;
        Zero(aiHints);
        aiHints.ai_family = AF_UNSPEC;
        aiHints.ai_socktype = SOCK_DGRAM;
        aiHints.ai_protocol = IPPROTO_UDP;

        // Do not use TMutex here: it has a non-trivial destructor which will be called before
        // destruction of current thread, if its TThread declared as global/static variable.
        static TAdaptiveLock cs;
        TGuard lock(cs);

        addrinfo* aiList = nullptr;
        for (int attempt = 0; attempt < 1000; ++attempt) {
            int rv = getaddrinfo(host.c_str(), "1313", &aiHints, &aiList);
            if (rv == 0)
                break;
            if (aiList) {
                freeaddrinfo(aiList);
            }
            if (rv != EAI_AGAIN) {
                return false;
            }
            usleep(100 * 1000);
        }
        for (addrinfo* ptr = aiList; ptr; ptr = ptr->ai_next) {
            sockaddr* addr = ptr->ai_addr;
            if (addr == nullptr)
                continue;
            switch (addressType) {
                case UAT_ANY: {
                    if (addr->sa_family != AF_INET && addr->sa_family != AF_INET6)
                        continue;
                    break;
                }
                case UAT_IPV4: {
                    if (addr->sa_family != AF_INET)
                        continue;
                    break;
                }
                case UAT_IPV6: {
                    if (addr->sa_family != AF_INET6)
                        continue;
                    break;
                }
            }

            GetUdpAddress(pRes, *(sockaddr_in6*)addr);
            pRes->Port = nPort;
            freeaddrinfo(aiList);
            return true;
        }
        freeaddrinfo(aiList);
        return false;
    }

    bool GetLocalAddresses(TVector<TUdpAddress>* addrs) {
#ifdef _win_
        TVector<char> buf;
        buf.resize(1000000);
        PIP_ADAPTER_ADDRESSES adapterBuf = (PIP_ADAPTER_ADDRESSES)&buf[0];
        ULONG bufSize = buf.ysize();

        ULONG rv = GetAdaptersAddresses(AF_UNSPEC, 0, NULL, adapterBuf, &bufSize);
        if (rv != ERROR_SUCCESS)
            return false;
        for (PIP_ADAPTER_ADDRESSES ptr = adapterBuf; ptr; ptr = ptr->Next) {
            if ((ptr->Flags & (IP_ADAPTER_IPV4_ENABLED | IP_ADAPTER_IPV6_ENABLED)) == 0) {
                continue;
            }
            if (ptr->IfType == IF_TYPE_TUNNEL) {
                // ignore tunnels
                continue;
            }
            if (ptr->OperStatus != IfOperStatusUp) {
                // ignore disable adapters
                continue;
            }
            if (ptr->Mtu < 1280) {
                fprintf(stderr, "WARNING: MTU %lu is less then ipv6 minimum", ptr->Mtu);
            }
            for (IP_ADAPTER_UNICAST_ADDRESS* addr = ptr->FirstUnicastAddress; addr; addr = addr->Next) {
                sockaddr* x = (sockaddr*)addr->Address.lpSockaddr;
                if (x == 0)
                    continue;
                if (x->sa_family == AF_INET || x->sa_family == AF_INET6) {
                    TUdpAddress address;
                    sockaddr_in6* xx = (sockaddr_in6*)x;
                    GetUdpAddress(&address, *xx);
                    addrs->push_back(address);
                }
            }
        }
        return true;
#else
        ifaddrs* ifap;
        if (getifaddrs(&ifap) != -1) {
            for (ifaddrs* ifa = ifap; ifa; ifa = ifa->ifa_next) {
                sockaddr* sa = (sockaddr*)ifa->ifa_addr;
                if (sa == nullptr)
                    continue;
                if (sa->sa_family == AF_INET || sa->sa_family == AF_INET6) {
                    TUdpAddress address;
                    sockaddr_in6* xx = (sockaddr_in6*)sa;
                    GetUdpAddress(&address, *xx);
                    addrs->push_back(address);
                }
            }
            freeifaddrs(ifap);
            return true;
        }
        return false;
#endif
    }

    void GetUdpAddress(TUdpAddress* res, const sockaddr_in6& addr) {
        if (addr.sin6_family == AF_INET) {
            const sockaddr_in& addr4 = *(const sockaddr_in*)&addr;
            res->Network = 0;
            res->Interface = 0xffff0000ll + (((ui64)(ui32)addr4.sin_addr.s_addr) << 32);
            res->Scope = 0;
            res->Port = ntohs(addr4.sin_port);
        } else if (addr.sin6_family == AF_INET6) {
            res->Network = *BreakAliasing<ui64>(addr.sin6_addr.s6_addr + 0);
            res->Interface = *BreakAliasing<ui64>(addr.sin6_addr.s6_addr + 8);
            res->Scope = addr.sin6_scope_id;
            res->Port = ntohs(addr.sin6_port);
        }
    }

    TUdpAddress GetUdpAddress(const sockaddr_in6& addr) {
        TUdpAddress result;
        GetUdpAddress(&result, addr);
        return result;
    }

    void GetWinsockAddr(sockaddr_in6* res, const TUdpAddress& addr) {
        if (0) { //addr.IsIPv4()) {
            // use ipv4 to ipv6 mapping
            //// ipv4
            //sockaddr_in &toAddress = *(sockaddr_in*)res;
            //Zero(toAddress);
            //toAddress.sin_family = AF_INET;
            //toAddress.sin_addr.s_addr = addr.GetIPv4();
            //toAddress.sin_port = htons((u_short)addr.Port);
        } else {
            // ipv6
            sockaddr_in6& toAddress = *(sockaddr_in6*)res;
            Zero(toAddress);
            toAddress.sin6_family = AF_INET6;
            *BreakAliasing<ui64>(toAddress.sin6_addr.s6_addr + 0) = addr.Network;
            *BreakAliasing<ui64>(toAddress.sin6_addr.s6_addr + 8) = addr.Interface;
            toAddress.sin6_scope_id = addr.Scope;
            toAddress.sin6_port = htons((u_short)addr.Port);
        }
    }

    void GetWinsockAddrForMLNX(sockaddr_in6* res, const TUdpAddress& addr) {
        if (addr.IsIPv4()) {
            // ipv4
            sockaddr_in& toAddress = *(sockaddr_in*)res;
            Zero(toAddress);
            toAddress.sin_family = AF_INET;
            toAddress.sin_addr.s_addr = addr.GetIPv4();
            toAddress.sin_port = htons((u_short)addr.Port);
        } else {
            // ipv6
            sockaddr_in6& toAddress = *(sockaddr_in6*)res;
            Zero(toAddress);
            toAddress.sin6_family = AF_INET6;
            *BreakAliasing<ui64>(toAddress.sin6_addr.s6_addr + 0) = addr.Network;
            *BreakAliasing<ui64>(toAddress.sin6_addr.s6_addr + 8) = addr.Interface;
            toAddress.sin6_scope_id = addr.Scope;
            toAddress.sin6_port = htons((u_short)addr.Port);
        }
    }

    TUdpAddress CreateAddress(const TString& server, int defaultPort, EUdpAddressType addressType) {
        TUdpAddress res;
        ParseInetName(&res, server.c_str(), defaultPort, addressType);
        return res;
    }

    TString GetAddressAsString(const TUdpAddress& addr) {
        char buf[1000];
        if (addr.IsIPv4()) {
            int ip = addr.GetIPv4();
            snprintf(buf, sizeof(buf), "%d.%d.%d.%d:%d",
                    (ip >> 0) & 0xff, (ip >> 8) & 0xff,
                    (ip >> 16) & 0xff, (ip >> 24) & 0xff,
                    addr.Port);
        } else {
            ui16 ipv6[8];
            *BreakAliasing<ui64>(ipv6) = addr.Network;
            *BreakAliasing<ui64>(ipv6 + 4) = addr.Interface;
            char suffix[100] = "";
            if (addr.Scope != 0) {
                snprintf(suffix, sizeof(suffix), "%%%d", addr.Scope);
            }
            snprintf(buf, sizeof(buf), "[%x:%x:%x:%x:%x:%x:%x:%x%s]:%d",
                    ntohs(ipv6[0]), ntohs(ipv6[1]), ntohs(ipv6[2]), ntohs(ipv6[3]),
                    ntohs(ipv6[4]), ntohs(ipv6[5]), ntohs(ipv6[6]), ntohs(ipv6[7]),
                    suffix, addr.Port);
        }
        return buf;
    }
}
