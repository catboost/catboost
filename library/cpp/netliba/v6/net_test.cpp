#include "stdafx.h"
#include "net_test.h"
#include "udp_address.h"

#ifndef _win_
#include <errno.h>
#endif

namespace NNetliba {
    TAtomic ActivePortTestersCount;

    const float PING_DELAY = 0.5f;

    TPortUnreachableTester::TPortUnreachableTester()
        : TimePassed(0)
        , ConnectOk(false)

    {
        s.Open(0);
        if (s.IsValid()) {
            AtomicAdd(ActivePortTestersCount, 1);
        }
    }

    void TPortUnreachableTester::Connect(const TUdpAddress& addr) {
        Y_ASSERT(IsValid());
        sockaddr_in6 toAddress;
        GetWinsockAddr(&toAddress, addr);
        ConnectOk = s.Connect(toAddress);
        TimePassed = 0;
    }

    TPortUnreachableTester::~TPortUnreachableTester() {
        if (s.IsValid())
            AtomicAdd(ActivePortTestersCount, -1);
    }

    bool TPortUnreachableTester::Test(float deltaT) {
        if (!ConnectOk)
            return false;
        if (s.IsHostUnreachable())
            return false;
        TimePassed += deltaT;
        if (TimePassed > PING_DELAY) {
            TimePassed = 0;
            s.SendEmptyPacket();
        }
        return true;
    }
}
