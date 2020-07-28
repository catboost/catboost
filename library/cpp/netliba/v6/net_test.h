#pragma once

#include "udp_socket.h"

namespace NNetliba {
    struct TUdpAddress;

    // needed to limit simultaneous port testers to avoid limit on open handles count
    extern TAtomic ActivePortTestersCount;

    // need separate socket for each destination
    // FreeBSD can not return port unreachable error for unconnected socket
    class TPortUnreachableTester: public TThrRefBase {
        TNetSocket s;
        float TimePassed;
        bool ConnectOk;

        ~TPortUnreachableTester() override;

    public:
        TPortUnreachableTester();
        bool IsValid() const {
            return s.IsValid();
        }
        void Connect(const TUdpAddress& addr);
        bool Test(float deltaT);
    };
}
