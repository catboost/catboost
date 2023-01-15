#include "sockpool.h"

void SetCommonSockOpts(SOCKET sock, const struct sockaddr* sa) {
    SetSockOpt(sock, SOL_SOCKET, SO_REUSEADDR, 1);

    if (!sa || sa->sa_family == AF_INET) {
        sockaddr_in s_in;
        s_in.sin_family = AF_INET;
        s_in.sin_addr.s_addr = INADDR_ANY;
        s_in.sin_port = 0;

        if (bind(sock, (struct sockaddr*)&s_in, sizeof(s_in)) == -1) {
            warn("bind");
        }
    } else if (sa->sa_family == AF_INET6) {
        sockaddr_in6 s_in6(*(const sockaddr_in6*)sa);
        Zero(s_in6.sin6_addr);
        s_in6.sin6_port = 0;

        if (bind(sock, (const struct sockaddr*)&s_in6, sizeof s_in6) == -1) {
            warn("bind6");
        }
    } else {
        Y_ASSERT(0);
    }

    SetNoDelay(sock, true);
}

TPooledSocket TSocketPool::AllocateMore(TConnectData* conn) {
    TCont* cont = conn->Cont;

    while (true) {
        TSocketHolder s(NCoro::Socket(Addr_->Addr()->sa_family, SOCK_STREAM, 0));

        if (s == INVALID_SOCKET) {
            ythrow TSystemError(errno) << TStringBuf("can not create socket");
        }

        SetCommonSockOpts(s, Addr_->Addr());
        SetZeroLinger(s);

        const int ret = NCoro::ConnectD(cont, s, Addr_->Addr(), Addr_->Len(), conn->DeadLine);

        if (ret == EINTR) {
            continue;
        } else if (ret) {
            ythrow TSystemError(ret) << TStringBuf("can not connect(") << cont->Name() << ')';
        }

        THolder<TPooledSocket::TImpl> res(new TPooledSocket::TImpl(s, this));
        s.Release();

        if (res->IsOpen()) {
            return res.Release();
        }
    }
}
