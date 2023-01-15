#pragma once

#include "asio.h"

#include "tcp_socket_impl.h"

namespace NAsio {
    class TOperationAccept: public TFdOperation {
    public:
        TOperationAccept(SOCKET fd, TTcpSocket::TImpl& newSocket, TTcpAcceptor::TAcceptHandler h, TInstant deadline)
            : TFdOperation(fd, PollRead, deadline)
            , H_(h)
            , NS_(newSocket)
        {
        }

        bool Execute(int errorCode) override;

        TTcpAcceptor::TAcceptHandler H_;
        TTcpSocket::TImpl& NS_;
    };

    class TTcpAcceptor::TImpl: public TThrRefBase {
    public:
        TImpl(TIOService::TImpl& srv) noexcept
            : Srv_(srv)
        {
        }

        inline void Bind(TEndpoint& ep, TErrorCode& ec) noexcept {
            TSocketHolder s(socket(ep.SockAddr()->sa_family, SOCK_STREAM, 0));

            if (s == INVALID_SOCKET) {
                ec.Assign(LastSystemError());
            }

            FixIPv6ListenSocket(s);
            CheckedSetSockOpt(s, SOL_SOCKET, SO_REUSEADDR, 1, "reuse addr");
            SetNonBlock(s);

            if (::bind(s, ep.SockAddr(), ep.SockAddrLen())) {
                ec.Assign(LastSystemError());
                return;
            }

            S_.Swap(s);
        }

        inline void Listen(int backlog, TErrorCode& ec) noexcept {
            if (::listen(S_, backlog)) {
                ec.Assign(LastSystemError());
                return;
            }
        }

        inline void AsyncAccept(TTcpSocket& s, TTcpAcceptor::TAcceptHandler h, TInstant deadline) {
            Srv_.ScheduleOp(new TOperationAccept((SOCKET)S_, s.GetImpl(), h, deadline)); //set callback
        }

        inline void AsyncCancel() {
            Srv_.ScheduleOp(new TOperationCancel<TTcpAcceptor::TImpl>(this));
        }

        inline TIOService::TImpl& GetIOServiceImpl() const noexcept {
            return Srv_;
        }

        inline SOCKET Fd() const noexcept {
            return S_;
        }

    private:
        TIOService::TImpl& Srv_;
        TSocketHolder S_;
    };
}
