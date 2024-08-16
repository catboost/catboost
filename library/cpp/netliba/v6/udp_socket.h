#pragma once

#include <util/generic/ptr.h>
#include <util/generic/utility.h>
#include <library/cpp/netliba/socket/socket.h>

namespace NNetliba {
    bool IsLocalIPv4(ui32 ip);
    bool IsLocalIPv6(ui64 network, ui64 iface);
    bool InitLocalIPList();

    constexpr int UDP_LOW_LEVEL_HEADER_SIZE = 4;

    using NNetlibaSocket::EFragFlag;
    using NNetlibaSocket::FF_ALLOW_FRAG;
    using NNetlibaSocket::FF_DONT_FRAG;

    class TNetSocket: public TNonCopyable {
        TIntrusivePtr<NNetlibaSocket::ISocket> s;
        ui32 PortCrc;
        mutable int Tos;

    public:
        enum ESendError {
            SEND_OK,
            SEND_BUFFER_OVERFLOW,
            SEND_NO_ROUTE_TO_HOST,
        };
        TNetSocket()
            : PortCrc(0)
            , Tos(0)
        {
        }
        ~TNetSocket() {
        }

        void Open(int port);
        void Open(const TIntrusivePtr<NNetlibaSocket::ISocket>& socket);
        void Close();
        void SendSelfFakePacket() const;
        bool IsValid() const {
            return s.Get() ? s->IsValid() : false;
        }
        int GetNetworkOrderPort() const {
            return s->GetNetworkOrderPort();
        }
        ESendError SendTo(const char* buf, int size, const sockaddr_in6& toAddress, const EFragFlag frag) const;
        bool RecvFrom(char* buf, int* size, sockaddr_in6* fromAddress) const;
        void Wait(float timeoutSec) const;
        void SetTOS(int n) const;

        // obtaining icmp host unreachable in convoluted way
        bool Connect(const sockaddr_in6& addr);
        void SendEmptyPacket();
        bool IsHostUnreachable();
    };
}
