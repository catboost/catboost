#pragma once

//
//primary header for work with asio
//

#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/network/socket.h>
#include <util/network/endpoint.h>
#include <util/system/error.h>
#include <util/stream/output.h>
#include <functional>

#include <library/cpp/dns/cache.h>

//#define DEBUG_ASIO

class TContIOVector;

namespace NAsio {
    class TErrorCode {
    public:
        inline TErrorCode(int val = 0) noexcept
            : Val_(val)
        {
        }

        typedef void (*TUnspecifiedBoolType)();

        static void UnspecifiedBoolTrue() {
        }

        //safe cast to bool value
        operator TUnspecifiedBoolType() const noexcept { // true if error
            return Val_ == 0 ? nullptr : UnspecifiedBoolTrue;
        }

        bool operator!() const noexcept {
            return Val_ == 0;
        }

        void Assign(int val) noexcept {
            Val_ = val;
        }

        int Value() const noexcept {
            return Val_;
        }

        TString Text() const {
            if (!Val_) {
                return TString();
            }
            return LastSystemErrorText(Val_);
        }

        void Check() {
            if (Val_) {
                throw TSystemError(Val_);
            }
        }

    private:
        int Val_;
    };

    //wrapper for TInstant, for enabling use TDuration (+TInstant::Now()) as deadline
    class TDeadline: public TInstant {
    public:
        TDeadline()
            : TInstant(TInstant::Max())
        {
        }

        TDeadline(const TInstant& t)
            : TInstant(t)
        {
        }

        TDeadline(const TDuration& d)
            : TInstant(TInstant::Now() + d)
        {
        }
    };

    class IHandlingContext {
    public:
        virtual ~IHandlingContext() {
        }

        //if handler throw exception, call this function be ignored
        virtual void ContinueUseHandler(TDeadline deadline = TDeadline()) = 0;
    };

    typedef std::function<void()> TCompletionHandler;

    class TIOService: public TNonCopyable {
    public:
        TIOService();
        ~TIOService();

        void Run();
        void Post(TCompletionHandler); //call handler in Run() thread-executor
        void Abort();                  //in Run() all exist async i/o operations + timers receive error = ECANCELED, Run() exited

        // not const since internal queue is lockfree and needs to increment and decrement its reference counters
        size_t GetOpQueueSize() noexcept;

        //counterpart boost::asio::io_service::work
        class TWork {
        public:
            TWork(TWork&);
            TWork(TIOService&);
            ~TWork();

        private:
            void operator=(const TWork&); //disable

            TIOService& Srv_;
        };

        class TImpl;

        TImpl& GetImpl() noexcept {
            return *Impl_;
        }

    private:
        THolder<TImpl> Impl_;
    };

    class TDeadlineTimer: public TNonCopyable {
    public:
        typedef std::function<void(const TErrorCode& err, IHandlingContext&)> THandler;

        TDeadlineTimer(TIOService&) noexcept;
        ~TDeadlineTimer();

        void AsyncWaitExpireAt(TDeadline, THandler);
        void Cancel();

        TIOService& GetIOService() const noexcept {
            return Srv_;
        }

        class TImpl;

    private:
        TIOService& Srv_;
        TImpl* Impl_;
    };

    class TTcpSocket: public TNonCopyable {
    public:
        class IBuffers {
        public:
            virtual ~IBuffers() {
            }
            virtual TContIOVector* GetIOvec() = 0;
        };
        typedef TAutoPtr<IBuffers> TSendedData;

        typedef std::function<void(const TErrorCode& err, IHandlingContext&)> THandler;
        typedef THandler TConnectHandler;
        typedef std::function<void(const TErrorCode& err, size_t amount, IHandlingContext&)> TWriteHandler;
        typedef std::function<void(const TErrorCode& err, size_t amount, IHandlingContext&)> TReadHandler;
        typedef THandler TPollHandler;

        enum TShutdownMode {
            ShutdownReceive = SHUT_RD,
            ShutdownSend = SHUT_WR,
            ShutdownBoth = SHUT_RDWR
        };

        TTcpSocket(TIOService&) noexcept;
        ~TTcpSocket();

        void AsyncConnect(const TEndpoint& ep, TConnectHandler, TDeadline deadline = TDeadline());
        void AsyncWrite(TSendedData&, TWriteHandler, TDeadline deadline = TDeadline());
        void AsyncWrite(TContIOVector* buff, TWriteHandler, TDeadline deadline = TDeadline());
        void AsyncWrite(const void* buff, size_t size, TWriteHandler, TDeadline deadline = TDeadline());
        void AsyncRead(void* buff, size_t size, TReadHandler, TDeadline deadline = TDeadline());
        void AsyncReadSome(void* buff, size_t size, TReadHandler, TDeadline deadline = TDeadline());
        void AsyncPollWrite(TPollHandler, TDeadline deadline = TDeadline());
        void AsyncPollRead(TPollHandler, TDeadline deadline = TDeadline());
        void AsyncCancel();

        //sync, but non blocked methods
        size_t WriteSome(TContIOVector&, TErrorCode&) noexcept;
        size_t WriteSome(const void* buff, size_t size, TErrorCode&) noexcept;
        size_t ReadSome(void* buff, size_t size, TErrorCode&) noexcept;

        bool IsOpen() const noexcept;
        void Shutdown(TShutdownMode mode, TErrorCode& ec);

        TIOService& GetIOService() const noexcept {
            return Srv_;
        }

        SOCKET Native() const noexcept;

        TEndpoint RemoteEndpoint() const;

        inline size_t WriteSome(TContIOVector& v) {
            TErrorCode ec;
            size_t n = WriteSome(v, ec);
            ec.Check();
            return n;
        }

        inline size_t WriteSome(const void* buff, size_t size) {
            TErrorCode ec;
            size_t n = WriteSome(buff, size, ec);
            ec.Check();
            return n;
        }

        inline size_t ReadSome(void* buff, size_t size) {
            TErrorCode ec;
            size_t n = ReadSome(buff, size, ec);
            ec.Check();
            return n;
        }

        void Shutdown(TShutdownMode mode) {
            TErrorCode ec;
            Shutdown(mode, ec);
            ec.Check();
        }

        class TImpl;

        TImpl& GetImpl() const noexcept {
            return *Impl_;
        }

    private:
        TIOService& Srv_;
        TIntrusivePtr<TImpl> Impl_;
    };

    class TTcpAcceptor: public TNonCopyable {
    public:
        typedef std::function<void(const TErrorCode& err, IHandlingContext&)> TAcceptHandler;

        TTcpAcceptor(TIOService&) noexcept;
        ~TTcpAcceptor();

        void Bind(TEndpoint&, TErrorCode&) noexcept;
        void Listen(int backlog, TErrorCode&) noexcept;

        void AsyncAccept(TTcpSocket&, TAcceptHandler, TDeadline deadline = TDeadline());

        void AsyncCancel();

        inline void Bind(TEndpoint& ep) {
            TErrorCode ec;
            Bind(ep, ec);
            ec.Check();
        }
        inline void Listen(int backlog) {
            TErrorCode ec;
            Listen(backlog, ec);
            ec.Check();
        }

        TIOService& GetIOService() const noexcept {
            return Srv_;
        }

        class TImpl;

        TImpl& GetImpl() const noexcept {
            return *Impl_;
        }

    private:
        TIOService& Srv_;
        TIntrusivePtr<TImpl> Impl_;
    };
}
