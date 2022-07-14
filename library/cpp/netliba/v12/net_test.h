#pragma once

#include "socket.h"
#include <util/generic/ptr.h>
#include <library/cpp/deprecated/atomic/atomic.h>

namespace NNetliba_v12 {
    struct TUdpAddress;

    // needed to limit simultaneous port testers to avoid limit on open handles count
    extern TAtomic ActivePortTestersCount;

    class TConnectedSocket {
        TIntrusivePtr<ISocket> S;

    public:
        TConnectedSocket();

        void Open(int port);
        void Close();
        bool IsValid() const;

        // obtaining icmp host unreachable in convoluted way
        bool Connect(const sockaddr_in6& addr);
        void SendEmptyPacket();
        bool IsHostUnreachable();
    };

    // need separate socket for each destination
    // FreeBSD can not return port unreachable error for unconnected socket
    class TPortUnreachableTester: public TThrRefBase {
        TConnectedSocket S;
        float TimePassed;
        bool ConnectOk;

        ~TPortUnreachableTester() override;

    public:
        TPortUnreachableTester();
        bool IsValid() const {
            return S.IsValid();
        }
        void Connect(const TUdpAddress& addr);
        bool Test(float deltaT);
    };
}
