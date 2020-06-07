#include "stdafx.h"
#include "net_test.h"
#include "udp_address.h"
#include "socket.h"

#ifndef _win_
#include <errno.h>
#endif

namespace NNetliba_v12 {
    TConnectedSocket::TConnectedSocket() {
        S = CreateSocket();
    }

    bool TConnectedSocket::IsValid() const {
        return S->IsValid();
    }

    void TConnectedSocket::Open(int port) {
        S->Open(port);
    }
    void TConnectedSocket::Close() {
        S->Close();
    }

    bool TConnectedSocket::Connect(const sockaddr_in6& addr) {
        // "connect" - meaningless operation
        // needed since port unreachable is routed only to "connected" udp sockets in ingenious FreeBSD
        if (S->Connect((sockaddr*)&addr, sizeof(addr)) < 0) {
            if (LastSystemError() == EHOSTUNREACH || LastSystemError() == ENETUNREACH) {
                return false;
            } else {
                Y_ASSERT(0);
            }
        }
        return true;
    }

    void TConnectedSocket::SendEmptyPacket() {
        TIoVec v;
        Zero(v);

        // darwin ignores packets with msg_iovlen == 0, also windows implementation uses sendto of first iovec.
        TMsgHdr hdr;
        Zero(hdr);
        hdr.msg_iov = &v;
        hdr.msg_iovlen = 1;

        S->SendMsg(&hdr, 0, FF_ALLOW_FRAG); // sends empty packet to connected address
    }

    bool TConnectedSocket::IsHostUnreachable() {
#ifdef _win_
        {
            char buf[10000];
            sockaddr_in6 fromAddress;

            const NNetlibaSocket::TIoVec v = NNetlibaSocket::CreateIoVec(buf, Y_ARRAY_SIZE(buf));
            NNetlibaSocket::TMsgHdr hdr = NNetlibaSocket::CreateRecvMsgHdr(&fromAddress, v);

            const ssize_t rv = S->RecvMsg(&hdr, 0);
            if (rv < 0) {
                int err = WSAGetLastError();
                if (err == WSAECONNRESET)
                    return true;
            }
        }
#else
        int err = 0;
        socklen_t bufSize = sizeof(err);
        S->GetSockOpt(SOL_SOCKET, SO_ERROR, (char*)&err, &bufSize);
        if (err == ECONNREFUSED)
            return true;
#endif
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////////

    TAtomic ActivePortTestersCount;

    const float PING_DELAY = 0.5f;

    TPortUnreachableTester::TPortUnreachableTester()
        : TimePassed(0)
        , ConnectOk(false)

    {
        S.Open(0);
        if (S.IsValid()) {
            AtomicAdd(ActivePortTestersCount, 1);
        }
    }

    void TPortUnreachableTester::Connect(const TUdpAddress& addr) {
        Y_ASSERT(IsValid());
        sockaddr_in6 toAddress;
        GetWinsockAddr(&toAddress, addr);
        ConnectOk = S.Connect(toAddress);
        TimePassed = 0;
    }

    TPortUnreachableTester::~TPortUnreachableTester() {
        if (S.IsValid())
            AtomicAdd(ActivePortTestersCount, -1);
    }

    bool TPortUnreachableTester::Test(float deltaT) {
        if (!ConnectOk)
            return false;
        if (S.IsHostUnreachable())
            return false;
        TimePassed += deltaT;
        if (TimePassed > PING_DELAY) {
            TimePassed = 0;
            S.SendEmptyPacket();
        }
        return true;
    }

}
