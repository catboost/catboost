#include "interface.h"

#include <util/string/ascii.h>

#if defined(_unix_)
    #include <ifaddrs.h>
#endif

#ifdef _win_
    #include <iphlpapi.h>
    #pragma comment(lib, "Iphlpapi.lib")
#endif

namespace NAddr {
    static bool IsInetAddress(sockaddr* addr) {
        return (addr != nullptr) && ((addr->sa_family == AF_INET) || (addr->sa_family == AF_INET6));
    }

    TNetworkInterfaceList GetNetworkInterfaces() {
        TNetworkInterfaceList result;

#ifdef _win_
        TVector<char> buf;
        buf.resize(1000000);
        PIP_ADAPTER_ADDRESSES adapterBuf = (PIP_ADAPTER_ADDRESSES)&buf[0];
        ULONG bufSize = buf.ysize();

        if (GetAdaptersAddresses(AF_UNSPEC, 0, nullptr, adapterBuf, &bufSize) == ERROR_SUCCESS) {
            for (PIP_ADAPTER_ADDRESSES ptr = adapterBuf; ptr != 0; ptr = ptr->Next) {
                // The check below makes code working on Vista+
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

                for (IP_ADAPTER_UNICAST_ADDRESS* addr = ptr->FirstUnicastAddress; addr != 0; addr = addr->Next) {
                    sockaddr* a = (sockaddr*)addr->Address.lpSockaddr;
                    if (IsInetAddress(a)) {
                        TNetworkInterface networkInterface;

                        // Not very efficient but straightforward
                        wchar_t* it = ptr->FriendlyName;
                        while (*it != '\0') {
                            networkInterface.Name += IsAscii(*it) ? static_cast<char>(*it) : '?';
                            ++it;
                        }

                        networkInterface.Address = new TOpaqueAddr(a);
                        result.push_back(networkInterface);
                    }
                }
            }
        }
#else
        ifaddrs* ifap;
        if (getifaddrs(&ifap) != -1) {
            for (ifaddrs* ifa = ifap; ifa != nullptr; ifa = ifa->ifa_next) {
                if (IsInetAddress(ifa->ifa_addr)) {
                    TNetworkInterface interface;
                    interface.Name = ifa->ifa_name;
                    interface.Address = new TOpaqueAddr(ifa->ifa_addr);
                    if (IsInetAddress(ifa->ifa_netmask)) {
                        interface.Mask = new TOpaqueAddr(ifa->ifa_netmask);
                    }
                    result.push_back(interface);
                }
            }
            freeifaddrs(ifap);
        }
#endif

        return result;
    }
}
