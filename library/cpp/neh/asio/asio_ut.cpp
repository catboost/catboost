#include "asio.h"
#include "tcp_acceptor_impl.h"
#include <util/network/address.h>

#include <library/cpp/unittest/registar.h>
#include <library/cpp/unittest/tests_data.h>

#include <util/stream/str.h>

using namespace NAsio;
using namespace std::placeholders;

Y_UNIT_TEST_SUITE(TAsio) {
    struct TTestSession {
        static const size_t TrDataSize = 20;
        TTestSession(bool throwExcept = false)
            : SIn(Srv)
            , SOut(Srv)
            , AcceptOk(false)
            , ConnectOk(false)
            , WriteOk(false)
            , ReadOk(false)
            , ThrowExcept(throwExcept)
        {
            for (size_t i = 0; i < TrDataSize; ++i) {
                BufferOut[i] = i;
            }
        }

        void OnAccept(const TErrorCode& ec, IHandlingContext&) {
            UNIT_ASSERT_VALUES_EQUAL_C(ec.Value(), 0, "accept");
            AcceptOk = true;
            SIn.AsyncRead(BufferIn, TrDataSize, std::bind(&TTestSession::OnRead, this, _1, _2, _3), TInstant::Now() + TDuration::Seconds(3));
            if (ThrowExcept) {
                throw yexception() << "accept";
            }
        }

        void OnConnect(const TErrorCode& ec, IHandlingContext&) {
            UNIT_ASSERT_VALUES_EQUAL_C(ec.Value(), 0, "connect");
            ConnectOk = true;
            SOut.AsyncWrite(BufferOut, TrDataSize, std::bind(&TTestSession::OnWrite, this, _1, _2, _3), TInstant::Now() + TDuration::Seconds(3));
            if (ThrowExcept) {
                throw yexception() << "connect";
            }
        }

        void OnWrite(const TErrorCode& ec, size_t amount, IHandlingContext&) {
            UNIT_ASSERT_VALUES_EQUAL_C(ec.Value(), 0, "write");
            UNIT_ASSERT_VALUES_EQUAL_C(amount, 20u, "write amount");
            WriteOk = true;
            if (ThrowExcept) {
                throw yexception() << "write";
            }
        }

        void OnRead(const TErrorCode& ec, size_t amount, IHandlingContext&) {
            UNIT_ASSERT_VALUES_EQUAL_C(ec.Value(), 0, "read");
            UNIT_ASSERT_VALUES_EQUAL_C(amount, 20u, "read amount");

            UNIT_ASSERT_VALUES_EQUAL(TStringBuf(BufferIn, TrDataSize), TStringBuf(BufferOut, TrDataSize));
            ReadOk = true;
            if (ThrowExcept) {
                throw yexception() << "read";
            }
        }

        TIOService Srv;
        TTcpSocket SIn;
        TTcpSocket SOut;
        char BufferOut[TrDataSize];
        char BufferIn[TrDataSize];
        bool AcceptOk;
        bool ConnectOk;
        bool WriteOk;
        bool ReadOk;
        bool ThrowExcept;
    };

    Y_UNIT_TEST(TTcpSocket_Bind_Listen_Connect_Accept_Write_Read) {
        TTestSession sess;
        TTcpAcceptor a(sess.Srv);

        TEndpoint ep; //ip v4
        ui16 bindedPort = 0;

        for (ui16 port = 10000; port < 40000; ++port) {
            ep.SetPort(port);

            TErrorCode ec;
            a.Bind(ep, ec);
            if (!ec) {
                TErrorCode ecL;
                a.Listen(100, ecL);
                if (!ecL) {
                    bindedPort = port;
                    break;
                }
            }
        }

        UNIT_ASSERT(bindedPort);

        a.AsyncAccept(sess.SIn, std::bind(&TTestSession::OnAccept, std::ref(sess), _1, _2), TDuration::Seconds(3));

        TEndpoint dest(new NAddr::TIPv4Addr(TIpAddress(InetToHost(INADDR_LOOPBACK), bindedPort)));

        sess.SOut.AsyncConnect(dest, std::bind(&TTestSession::OnConnect, std::ref(sess), _1, _2), TDuration::Seconds(3));

        sess.Srv.Run();

        UNIT_ASSERT(sess.AcceptOk);
        UNIT_ASSERT(sess.ConnectOk);
        UNIT_ASSERT(sess.WriteOk);
        UNIT_ASSERT(sess.ReadOk);
    }

    class TTestErrorSession {
    public:
        TTestErrorSession()
            : SOut(Srv)
            , ConnectErrorCode(0)
        {}

        ~TTestErrorSession() {
            close(SocketFd_);
        }

        void OnConnect(const TErrorCode& ec, IHandlingContext&) {
            ConnectErrorCode = ec.Value();
        }

        ui16 AllocateTcpPort() {
            return PortManager_.GetPort();
        }

    #ifdef __unix__
        void PrepareUnixSocket(const TUnixSocketPath& unixSocketPath) {
            SocketFd_ = socket(AF_UNIX, SOCK_STREAM, 0);
            UNIT_ASSERT(SocketFd_ != -1);

            struct sockaddr_un sockAddr;
            sockAddr.sun_family = AF_UNIX;
            strcpy(sockAddr.sun_path, unixSocketPath.Path.data());
            unlink(unixSocketPath.Path.data());

            UNIT_ASSERT(bind(SocketFd_, (struct sockaddr*)&sockAddr, sizeof(sockAddr)) != -1);

            NetworkAddress_ = new TNetworkAddress(unixSocketPath);
        }
    #endif

        TEndpoint::TAddrRef GetFirstAddress() {
            Y_ENSURE(NetworkAddress_);

            THolder<NAddr::TAddrInfo> addrInfo = nullptr;
            for (TNetworkAddress::TIterator ai = NetworkAddress_->Begin(); ai != NetworkAddress_->End(); ai++) {
                addrInfo = new NAddr::TAddrInfo(&*ai);
            }
            Y_ENSURE(addrInfo);

            return TEndpoint::TAddrRef(addrInfo.Release());
        }

    public:
        TIOService Srv;
        TTcpSocket SOut;

        int ConnectErrorCode;

    private:
        TPortManager PortManager_;

        SOCKET SocketFd_;
        THolder<TNetworkAddress> NetworkAddress_;
    };

    Y_UNIT_TEST(TTcpSocket_ConnectionRefused) {
        TTestErrorSession sess;

        ui16 bindedPort = sess.AllocateTcpPort();
        UNIT_ASSERT(bindedPort);

        TEndpoint dest(new NAddr::TIPv4Addr(TIpAddress(InetToHost(INADDR_LOOPBACK), bindedPort)));

        sess.SOut.AsyncConnect(dest, std::bind(&TTestErrorSession::OnConnect, std::ref(sess), _1, _2), TDuration::Seconds(3));

        sess.Srv.Run();

        UNIT_ASSERT_VALUES_EQUAL(sess.ConnectErrorCode, ECONNREFUSED);
    }

#ifdef __unix__
    Y_UNIT_TEST(TTcpSocket_UnixSocket_ConnectionRefused) {
        TTestErrorSession sess;

        TUnixSocketPath unixSocketPath("./unixsocket");
        sess.PrepareUnixSocket(unixSocketPath);

        TEndpoint dest(sess.GetFirstAddress());

        sess.SOut.AsyncConnect(dest, std::bind(&TTestErrorSession::OnConnect, std::ref(sess), _1, _2), TDuration::Seconds(3));

        sess.Srv.Run();

        UNIT_ASSERT_VALUES_EQUAL(sess.ConnectErrorCode, ECONNREFUSED);
    }
#endif

    struct TTestTimer {
        TTestTimer()
            : Srv()
            , CreateTime(TInstant::Now())
            , Dt1(Srv)
            , Dt2(Srv)
        {
            Ec1.Assign(666);
        }

        void OnTimeout1(const TErrorCode& ec, IHandlingContext&) {
            Ec1 = ec;
            Timeout1Time = TInstant::Now();
            Dt2.Cancel();
        }

        void OnTimeout2(const TErrorCode& ec, IHandlingContext&) {
            Ec2 = ec;
        }

        static void ThrowExcept() {
            throw yexception() << "test exception";
        }

        TIOService Srv;
        TInstant CreateTime;
        TInstant Timeout1Time;
        TDeadlineTimer Dt1;
        TDeadlineTimer Dt2;
        TErrorCode Ec1;
        TErrorCode Ec2;
    };

    Y_UNIT_TEST(TTimer) {
        TTestTimer test;
        test.Dt1.AsyncWaitExpireAt(TDuration::MilliSeconds(500), std::bind(&TTestTimer::OnTimeout1, std::ref(test), _1, _2));

        test.Dt2.AsyncWaitExpireAt(TDuration::Seconds(10), std::bind(&TTestTimer::OnTimeout2, std::ref(test), _1, _2));

        test.Srv.Run();

        UNIT_ASSERT_VALUES_EQUAL(test.Ec1.Value(), 0);
        UNIT_ASSERT_VALUES_EQUAL(test.Ec2.Value(), ECANCELED);
        UNIT_ASSERT(test.CreateTime + TDuration::MilliSeconds(450) < test.Timeout1Time);
        UNIT_ASSERT(test.CreateTime + TDuration::MilliSeconds(2000) > test.Timeout1Time);
    }

    Y_UNIT_TEST(TRestartAfterThrow) {
        {
            TTestTimer test;

            test.Dt1.AsyncWaitExpireAt(TDuration::MilliSeconds(500), std::bind(&TTestTimer::OnTimeout1, std::ref(test), _1, _2));

            test.Dt2.AsyncWaitExpireAt(TDuration::Seconds(10), std::bind(&TTestTimer::OnTimeout2, std::ref(test), _1, _2));

            test.Srv.Post(&TTestTimer::ThrowExcept);

            bool catchExcept = false;

            try {
                try {
                    test.Srv.Run();
                } catch (...) {
                    catchExcept = true;
                    UNIT_ASSERT_STRING_CONTAINS(CurrentExceptionMessage(), AsStringBuf("test exception"));
                    test.Srv.Run();
                }

                UNIT_ASSERT_VALUES_EQUAL(catchExcept, true);
                UNIT_ASSERT_VALUES_EQUAL(test.Ec1.Value(), 0);
                UNIT_ASSERT_VALUES_EQUAL(test.Ec2.Value(), ECANCELED);
                UNIT_ASSERT(test.CreateTime + TDuration::MilliSeconds(450) < test.Timeout1Time);
                UNIT_ASSERT(test.CreateTime + TDuration::MilliSeconds(2000) > test.Timeout1Time);
            } catch (...) {
                test.Dt1.Cancel();
                test.Dt2.Cancel();
                test.Srv.Run();
                Sleep(TDuration::Seconds(1));
                throw;
            }
        }

        {
            TTestSession sess(true);
            TTcpAcceptor a(sess.Srv);

            TEndpoint ep; //ip v4
            ui16 bindedPort = 0;

            for (ui16 port = 10000; port < 40000; ++port) {
                ep.SetPort(port);
                TErrorCode ec1;
                a.Bind(ep, ec1);
                if (!ec1) {
                    bindedPort = port;
                    {
                        TErrorCode ec2;
                        a.Listen(100, ec2);
                        if (!ec2) {
                            break;
                        }
                    }
                }
            }

            UNIT_ASSERT(bindedPort);

            a.AsyncAccept(sess.SIn, std::bind(&TTestSession::OnAccept, std::ref(sess), _1, _2), TDuration::Seconds(3));

            TEndpoint dest(new NAddr::TIPv4Addr(TIpAddress(InetToHost(INADDR_LOOPBACK), bindedPort)));

            sess.SOut.AsyncConnect(dest, std::bind(&TTestSession::OnConnect, std::ref(sess), _1, _2), TDuration::Seconds(3));

            size_t exceptCnt = 0;
            while (true) {
                try {
                    sess.Srv.Run();
                    break;
                } catch (...) {
                    ++exceptCnt;
                }
            }

            UNIT_ASSERT_VALUES_EQUAL(exceptCnt, 4u);
            UNIT_ASSERT(sess.AcceptOk);
            UNIT_ASSERT(sess.ConnectOk);
            UNIT_ASSERT(sess.WriteOk);
            UNIT_ASSERT(sess.ReadOk);
        }
    }

    struct TTestAcceptorLifespan: public TThrRefBase {
        TTestAcceptorLifespan(TIOService& srv)
            : A_(srv)
        {
        }

        ~TTestAcceptorLifespan() override {
            Destroyed = true;
        }

        void StartAccept() {
            TSimpleSharedPtr<TTcpSocket> s(new TTcpSocket(A_.GetIOService())); //socket for accepting
            A_.AsyncAccept(*s, std::bind(&TTestAcceptorLifespan::OnAccept, TIntrusivePtr<TTestAcceptorLifespan>(this), s, _1, _2));
        }

        void OnAccept(TSimpleSharedPtr<TTcpSocket>, const TErrorCode& err, IHandlingContext&) {
            if (!err) {
                StartAccept();
            }
        }

        TTcpAcceptor A_;

        static void ShutdownSocket(SOCKET fd, const TErrorCode&, IHandlingContext&) {
            shutdown(fd, SHUT_RDWR);
        }

        static bool Destroyed;
    };

    bool TTestAcceptorLifespan::Destroyed = false;

    Y_UNIT_TEST(TCheckTcpAcceptorLifespan) {
        TIOService srv;
        TIntrusivePtr<TTestAcceptorLifespan> a(new TTestAcceptorLifespan(srv));

        ui16 bindedPort = 0;
        for (ui16 port = 10000; port < 40000; ++port) {
            TEndpoint ep; //ip v4
            ep.SetPort(port);

            TErrorCode ec;
            a->A_.Bind(ep, ec);
            if (!ec) {
                bindedPort = port;
                break;
            }
        }
        UNIT_ASSERT(bindedPort);
        a->StartAccept();
        SOCKET fd = a->A_.GetImpl().Fd();
        a.Drop(); //now only asio reference to acceptor (via TTestAcceptorLifespan)
        UNIT_ASSERT(!TTestAcceptorLifespan::Destroyed);

        TDeadlineTimer dt(srv);
        dt.AsyncWaitExpireAt(TDuration::MilliSeconds(10), std::bind(&TTestAcceptorLifespan::ShutdownSocket, fd, _1, _2));
        srv.Run();
        UNIT_ASSERT(TTestAcceptorLifespan::Destroyed);
    }
}
