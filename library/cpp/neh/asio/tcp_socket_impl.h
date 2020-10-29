#pragma once

#include "asio.h"
#include "io_service_impl.h"

#include <sys/uio.h>

#if defined(_bionic_)
#   define IOV_MAX 1024
#endif

namespace NAsio {
    // ownership/keep-alive references:
    // Handlers <- TOperation...(TFdOperation) <- TPollFdEventHandler <- TIOService

    class TSocketOperation: public TFdOperation {
    public:
        TSocketOperation(TTcpSocket::TImpl& s, TPollType pt, TInstant deadline);

    protected:
        TTcpSocket::TImpl& S_;
    };

    class TOperationConnect: public TSocketOperation {
    public:
        TOperationConnect(TTcpSocket::TImpl& s, TTcpSocket::TConnectHandler h, TInstant deadline)
            : TSocketOperation(s, PollWrite, deadline)
            , H_(h)
        {
        }

        bool Execute(int errorCode) override {
            H_(errorCode, *this);

            return true;
        }

        TTcpSocket::TConnectHandler H_;
    };

    class TOperationConnectFailed: public TSocketOperation {
    public:
        TOperationConnectFailed(TTcpSocket::TImpl& s, TTcpSocket::TConnectHandler h, int errorCode, TInstant deadline)
            : TSocketOperation(s, PollWrite, deadline)
            , H_(h)
            , ErrorCode_(errorCode)
        {
            Speculative_ = true;
        }

        bool Execute(int errorCode) override {
            Y_UNUSED(errorCode);
            H_(ErrorCode_, *this);

            return true;
        }

        TTcpSocket::TConnectHandler H_;
        int ErrorCode_;
    };

    class TOperationWrite: public TSocketOperation {
    public:
        TOperationWrite(TTcpSocket::TImpl& s, NAsio::TTcpSocket::TSendedData& buffs, TTcpSocket::TWriteHandler h, TInstant deadline)
            : TSocketOperation(s, PollWrite, deadline)
            , H_(h)
            , Buffs_(buffs)
            , Written_(0)
        {
            Speculative_ = true;
        }

        //return true, if not need write more data
        bool Execute(int errorCode) override;

    private:
        TTcpSocket::TWriteHandler H_;
        NAsio::TTcpSocket::TSendedData Buffs_;
        size_t Written_;
    };

    class TOperationWriteVector: public TSocketOperation {
    public:
        TOperationWriteVector(TTcpSocket::TImpl& s, TContIOVector* v, TTcpSocket::TWriteHandler h, TInstant deadline)
            : TSocketOperation(s, PollWrite, deadline)
            , H_(h)
            , V_(*v)
            , Written_(0)
        {
            Speculative_ = true;
        }

        //return true, if not need write more data
        bool Execute(int errorCode) override;

    private:
        TTcpSocket::TWriteHandler H_;
        TContIOVector& V_;
        size_t Written_;
    };

    class TOperationReadSome: public TSocketOperation {
    public:
        TOperationReadSome(TTcpSocket::TImpl& s, void* buff, size_t size, TTcpSocket::TReadHandler h, TInstant deadline)
            : TSocketOperation(s, PollRead, deadline)
            , H_(h)
            , Buff_(static_cast<char*>(buff))
            , Size_(size)
        {
        }

        //return true, if not need read more data
        bool Execute(int errorCode) override;

    protected:
        TTcpSocket::TReadHandler H_;
        char* Buff_;
        size_t Size_;
    };

    class TOperationRead: public TOperationReadSome {
    public:
        TOperationRead(TTcpSocket::TImpl& s, void* buff, size_t size, TTcpSocket::TReadHandler h, TInstant deadline)
            : TOperationReadSome(s, buff, size, h, deadline)
            , Read_(0)
        {
        }

        bool Execute(int errorCode) override;

    private:
        size_t Read_;
    };

    class TOperationPoll: public TSocketOperation {
    public:
        TOperationPoll(TTcpSocket::TImpl& s, TPollType pt, TTcpSocket::TPollHandler h, TInstant deadline)
            : TSocketOperation(s, pt, deadline)
            , H_(h)
        {
        }

        bool Execute(int errorCode) override {
            H_(errorCode, *this);

            return true;
        }

    private:
        TTcpSocket::TPollHandler H_;
    };

    template <class T>
    class TOperationCancel: public TNoneOperation {
    public:
        TOperationCancel(T* s)
            : TNoneOperation()
            , S_(s)
        {
            Speculative_ = true;
        }

        ~TOperationCancel() override {
        }

    private:
        bool Execute(int errorCode) override {
            Y_UNUSED(errorCode);
            if (!errorCode && S_->Fd() != INVALID_SOCKET) {
                S_->GetIOServiceImpl().CancelFdOp(S_->Fd());
            }
            return true;
        }

        TIntrusivePtr<T> S_;
    };

    class TTcpSocket::TImpl: public TNonCopyable, public TThrRefBase {
    public:
        typedef TTcpSocket::TSendedData TSendedData;

        TImpl(TIOService::TImpl& srv) noexcept
            : Srv_(srv)
        {
        }

        ~TImpl() override {
            DBGOUT("TSocket::~TImpl()");
        }

        void Assign(SOCKET fd, TEndpoint ep) {
            TSocketHolder(fd).Swap(S_);
            RemoteEndpoint_ = ep;
        }

        void AsyncConnect(const TEndpoint& ep, TTcpSocket::TConnectHandler h, TInstant deadline) {
            TSocketHolder s(socket(ep.SockAddr()->sa_family, SOCK_STREAM, 0));

            if (Y_UNLIKELY(s == INVALID_SOCKET || Srv_.HasAbort())) {
                throw TSystemError() << TStringBuf("can't create socket");
            }

            SetNonBlock(s);

            int err;
            do {
                err = connect(s, ep.SockAddr(), (int)ep.SockAddrLen());
                if (Y_LIKELY(err)) {
                    err = LastSystemError();
                }
#if defined(_freebsd_)
                if (Y_UNLIKELY(err == EINTR)) {
                    err = EINPROGRESS;
                }
            } while (0);
#elif defined(_linux_)
            } while (Y_UNLIKELY(err == EINTR));
#else
            } while (0);
#endif

            RemoteEndpoint_ = ep;
            S_.Swap(s);

            DBGOUT("AsyncConnect(): " << err);
            if (Y_LIKELY(err == EINPROGRESS || err == EWOULDBLOCK || err == 0)) {
                Srv_.ScheduleOp(new TOperationConnect(*this, h, deadline)); //set callback
            } else {
                Srv_.ScheduleOp(new TOperationConnectFailed(*this, h, err, deadline)); //set callback
            }
        }

        inline void AsyncWrite(TSendedData& d, TTcpSocket::TWriteHandler h, TInstant deadline) {
            Srv_.ScheduleOp(new TOperationWrite(*this, d, h, deadline));
        }

        inline void AsyncWrite(TContIOVector* v, TTcpSocket::TWriteHandler h, TInstant deadline) {
            Srv_.ScheduleOp(new TOperationWriteVector(*this, v, h, deadline));
        }

        inline void AsyncRead(void* buff, size_t size, TTcpSocket::TReadHandler h, TInstant deadline) {
            Srv_.ScheduleOp(new TOperationRead(*this, buff, size, h, deadline));
        }

        inline void AsyncReadSome(void* buff, size_t size, TTcpSocket::TReadHandler h, TInstant deadline) {
            Srv_.ScheduleOp(new TOperationReadSome(*this, buff, size, h, deadline));
        }

        inline void AsyncPollWrite(TTcpSocket::TPollHandler h, TInstant deadline) {
            Srv_.ScheduleOp(new TOperationPoll(*this, TOperationPoll::PollWrite, h, deadline));
        }

        inline void AsyncPollRead(TTcpSocket::TPollHandler h, TInstant deadline) {
            Srv_.ScheduleOp(new TOperationPoll(*this, TOperationPoll::PollRead, h, deadline));
        }

        inline void AsyncCancel() {
            if (Y_UNLIKELY(Srv_.HasAbort())) {
                return;
            }
            Srv_.ScheduleOp(new TOperationCancel<TTcpSocket::TImpl>(this));
        }

        inline bool SysCallHasResult(ssize_t& n, TErrorCode& ec) noexcept {
            if (n >= 0) {
                return true;
            }

            int errn = LastSystemError();
            if (errn == EINTR) {
                return false;
            }

            ec.Assign(errn);
            n = 0;
            return true;
        }

        size_t WriteSome(TContIOVector& iov, TErrorCode& ec) noexcept {
            for (;;) {
                ssize_t n = writev(S_, (const iovec*)iov.Parts(), Min(IOV_MAX, (int)iov.Count()));
                DBGOUT("WriteSome(): n=" << n);
                if (SysCallHasResult(n, ec)) {
                    return n;
                }
            }
        }

        size_t WriteSome(const void* buff, size_t size, TErrorCode& ec) noexcept {
            for (;;) {
                ssize_t n = send(S_, (char*)buff, size, 0);
                DBGOUT("WriteSome(): n=" << n);
                if (SysCallHasResult(n, ec)) {
                    return n;
                }
            }
        }

        size_t ReadSome(void* buff, size_t size, TErrorCode& ec) noexcept {
            for (;;) {
                ssize_t n = recv(S_, (char*)buff, size, 0);
                DBGOUT("ReadSome(): n=" << n);
                if (SysCallHasResult(n, ec)) {
                    return n;
                }
            }
        }

        inline void Shutdown(TTcpSocket::TShutdownMode mode, TErrorCode& ec) {
            if (shutdown(S_, mode)) {
                ec.Assign(LastSystemError());
            }
        }

        TIOService::TImpl& GetIOServiceImpl() const noexcept {
            return Srv_;
        }

        inline SOCKET Fd() const noexcept {
            return S_;
        }

        TEndpoint RemoteEndpoint() const {
            return RemoteEndpoint_;
        }

    private:
        TIOService::TImpl& Srv_;
        TSocketHolder S_;
        TEndpoint RemoteEndpoint_;
    };
}
