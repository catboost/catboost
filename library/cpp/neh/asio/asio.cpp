#include "io_service_impl.h"
#include "deadline_timer_impl.h"
#include "tcp_socket_impl.h"
#include "tcp_acceptor_impl.h"

using namespace NDns;
using namespace NAsio;

namespace NAsio {
    TIOService::TWork::TWork(TWork& w)
        : Srv_(w.Srv_)
    {
        Srv_.GetImpl().WorkStarted();
    }

    TIOService::TWork::TWork(TIOService& srv)
        : Srv_(srv)
    {
        Srv_.GetImpl().WorkStarted();
    }

    TIOService::TWork::~TWork() {
        Srv_.GetImpl().WorkFinished();
    }

    TIOService::TIOService()
        : Impl_(new TImpl())
    {
    }

    TIOService::~TIOService() {
    }

    void TIOService::Run() {
        Impl_->Run();
    }

    size_t TIOService::GetOpQueueSize() noexcept {
        return Impl_->GetOpQueueSize();
    }

    void TIOService::Post(TCompletionHandler h) {
        Impl_->Post(std::move(h));
    }

    void TIOService::Abort() {
        Impl_->Abort();
    }

    TDeadlineTimer::TDeadlineTimer(TIOService& srv) noexcept
        : Srv_(srv)
        , Impl_(nullptr)
    {
    }

    TDeadlineTimer::~TDeadlineTimer() {
        if (Impl_) {
            Srv_.GetImpl().ScheduleOp(new TUnregisterTimerOperation(Impl_));
        }
    }

    void TDeadlineTimer::AsyncWaitExpireAt(TDeadline deadline, THandler h) {
        if (!Impl_) {
            Impl_ = new TDeadlineTimer::TImpl(Srv_.GetImpl());
            Srv_.GetImpl().ScheduleOp(new TRegisterTimerOperation(Impl_));
        }
        Impl_->AsyncWaitExpireAt(deadline, h);
    }

    void TDeadlineTimer::Cancel() {
        Impl_->Cancel();
    }

    TTcpSocket::TTcpSocket(TIOService& srv) noexcept
        : Srv_(srv)
        , Impl_(new TImpl(srv.GetImpl()))
    {
    }

    TTcpSocket::~TTcpSocket() {
    }

    void TTcpSocket::AsyncConnect(const TEndpoint& ep, TTcpSocket::TConnectHandler h, TDeadline deadline) {
        Impl_->AsyncConnect(ep, h, deadline);
    }

    void TTcpSocket::AsyncWrite(TSendedData& d, TTcpSocket::TWriteHandler h, TDeadline deadline) {
        Impl_->AsyncWrite(d, h, deadline);
    }

    void TTcpSocket::AsyncWrite(TContIOVector* vec, TWriteHandler h, TDeadline deadline) {
        Impl_->AsyncWrite(vec, h, deadline);
    }

    void TTcpSocket::AsyncWrite(const void* data, size_t size, TWriteHandler h, TDeadline deadline) {
        class TBuffers: public IBuffers {
        public:
            TBuffers(const void* theData, size_t theSize)
                : Part(theData, theSize)
                , IOVec(&Part, 1)
            {
            }

            TContIOVector* GetIOvec() override {
                return &IOVec;
            }

            IOutputStream::TPart Part;
            TContIOVector IOVec;
        };

        TSendedData d(new TBuffers(data, size));
        Impl_->AsyncWrite(d, h, deadline);
    }

    void TTcpSocket::AsyncRead(void* buff, size_t size, TTcpSocket::TReadHandler h, TDeadline deadline) {
        Impl_->AsyncRead(buff, size, h, deadline);
    }

    void TTcpSocket::AsyncReadSome(void* buff, size_t size, TTcpSocket::TReadHandler h, TDeadline deadline) {
        Impl_->AsyncReadSome(buff, size, h, deadline);
    }

    void TTcpSocket::AsyncPollRead(TTcpSocket::TPollHandler h, TDeadline deadline) {
        Impl_->AsyncPollRead(h, deadline);
    }

    void TTcpSocket::AsyncPollWrite(TTcpSocket::TPollHandler h, TDeadline deadline) {
        Impl_->AsyncPollWrite(h, deadline);
    }

    void TTcpSocket::AsyncCancel() {
        return Impl_->AsyncCancel();
    }

    size_t TTcpSocket::WriteSome(TContIOVector& d, TErrorCode& ec) noexcept {
        return Impl_->WriteSome(d, ec);
    }

    size_t TTcpSocket::WriteSome(const void* buff, size_t size, TErrorCode& ec) noexcept {
        return Impl_->WriteSome(buff, size, ec);
    }

    size_t TTcpSocket::ReadSome(void* buff, size_t size, TErrorCode& ec) noexcept {
        return Impl_->ReadSome(buff, size, ec);
    }

    bool TTcpSocket::IsOpen() const noexcept {
        return Native() != INVALID_SOCKET;
    }

    void TTcpSocket::Shutdown(TShutdownMode what, TErrorCode& ec) {
        return Impl_->Shutdown(what, ec);
    }

    SOCKET TTcpSocket::Native() const noexcept {
        return Impl_->Fd();
    }

    TEndpoint TTcpSocket::RemoteEndpoint() const {
        return Impl_->RemoteEndpoint();
    }

    //////////////////////////////////

    TTcpAcceptor::TTcpAcceptor(TIOService& srv) noexcept
        : Srv_(srv)
        , Impl_(new TImpl(srv.GetImpl()))
    {
    }

    TTcpAcceptor::~TTcpAcceptor() {
    }

    void TTcpAcceptor::Bind(TEndpoint& ep, TErrorCode& ec) noexcept {
        return Impl_->Bind(ep, ec);
    }

    void TTcpAcceptor::Listen(int backlog, TErrorCode& ec) noexcept {
        return Impl_->Listen(backlog, ec);
    }

    void TTcpAcceptor::AsyncAccept(TTcpSocket& s, TTcpAcceptor::TAcceptHandler h, TDeadline deadline) {
        return Impl_->AsyncAccept(s, h, deadline);
    }

    void TTcpAcceptor::AsyncCancel() {
        Impl_->AsyncCancel();
    }

}
