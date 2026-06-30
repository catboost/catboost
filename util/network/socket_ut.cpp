#include "socket.h"

#include "pair.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/string/builder.h>
#include <util/generic/vector.h>

#include <ctime>

#ifdef _linux_
    #include <linux/version.h>
    #include <sys/utsname.h>
#endif

class TSockTest: public TTestBase {
    UNIT_TEST_SUITE(TSockTest);
    UNIT_TEST(TestSock);
    UNIT_TEST(TestTimeout);
#ifndef _win_ // Test hangs on Windows
    UNIT_TEST_EXCEPTION(TestConnectionRefused, yexception);
#endif
    UNIT_TEST(TestNetworkResolutionError);
    UNIT_TEST(TestNetworkResolutionErrorMessage);
    UNIT_TEST(TestBrokenPipe);
    UNIT_TEST(TestClose);
    UNIT_TEST_SUITE_END();

public:
    void TestSock();
    void TestTimeout();
    void TestConnectionRefused();
    void TestNetworkResolutionError();
    void TestNetworkResolutionErrorMessage();
    void TestBrokenPipe();
    void TestClose();
};

UNIT_TEST_SUITE_REGISTRATION(TSockTest);

void TSockTest::TestSock() {
    TNetworkAddress addr("yandex.ru", 80);
    TSocket s(addr);
    TSocketOutput so(s);
    TSocketInput si(s);
    const TStringBuf req = "GET / HTTP/1.1\r\nHost: yandex.ru\r\n\r\n";

    so.Write(req.data(), req.size());

    UNIT_ASSERT(!si.ReadLine().empty());
}

void TSockTest::TestTimeout() {
    static const int timeout = 1000;
    i64 startTime = millisec();
    try {
        TNetworkAddress addr("localhost", 1313);
        TSocket s(addr, TDuration::MilliSeconds(timeout));
    } catch (const yexception&) {
    }
    int realTimeout = (int)(millisec() - startTime);
    if (realTimeout > timeout + 2000) {
        TString err = TStringBuilder() << "Timeout exceeded: " << realTimeout << " ms (expected " << timeout << " ms)";
        UNIT_FAIL(err);
    }
}

void TSockTest::TestConnectionRefused() {
    TNetworkAddress addr("localhost", 1313);
    TSocket s(addr);
}

void TSockTest::TestNetworkResolutionError() {
    TString errMsg;
    try {
        TNetworkAddress addr("", 0);
    } catch (const TNetworkResolutionError& e) {
        errMsg = e.what();
    }

    if (errMsg.empty()) {
        return; // on Windows getaddrinfo("", 0, ...) returns "OK"
    }

    int expectedErr = EAI_NONAME;
    TString expectedErrMsg = gai_strerror(expectedErr);
    if (errMsg.find(expectedErrMsg) == TString::npos) {
        UNIT_FAIL("TNetworkResolutionError contains\nInvalid msg: " + errMsg + "\nExpected msg: " + expectedErrMsg + "\n");
    }
}

void TSockTest::TestNetworkResolutionErrorMessage() {
#ifdef _unix_
    auto str = [](int code) -> TString {
        return TNetworkResolutionError(code).what();
    };

    auto expected = [](int code) -> TString {
        return gai_strerror(code);
    };

    struct TErrnoGuard {
        TErrnoGuard()
            : PrevValue_(errno)
        {
        }

        ~TErrnoGuard() {
            errno = PrevValue_;
        }

    private:
        int PrevValue_;
    } g;

    UNIT_ASSERT_VALUES_EQUAL(expected(0) + "(0): ", str(0));
    UNIT_ASSERT_VALUES_EQUAL(expected(-9) + "(-9): ", str(-9));

    errno = 0;
    UNIT_ASSERT_VALUES_EQUAL(expected(EAI_SYSTEM) + "(" + IntToString<10>(EAI_SYSTEM) + "; errno=0): ",
                             str(EAI_SYSTEM));
    errno = 110;
    UNIT_ASSERT_VALUES_EQUAL(expected(EAI_SYSTEM) + "(" + IntToString<10>(EAI_SYSTEM) + "; errno=110): ",
                             str(EAI_SYSTEM));
#endif
}

class TTempEnableSigPipe {
public:
    TTempEnableSigPipe() {
        OriginalSigHandler_ = signal(SIGPIPE, SIG_DFL);
        Y_ABORT_UNLESS(OriginalSigHandler_ != SIG_ERR);
    }

    ~TTempEnableSigPipe() {
        auto ret = signal(SIGPIPE, OriginalSigHandler_);
        Y_ABORT_UNLESS(ret != SIG_ERR);
    }

private:
    void (*OriginalSigHandler_)(int);
};

void TSockTest::TestBrokenPipe() {
    TTempEnableSigPipe guard;

    SOCKET socks[2];

    int ret = SocketPair(socks);
    UNIT_ASSERT_VALUES_EQUAL(ret, 0);

    TSocket sender(socks[0]);
    TSocket receiver(socks[1]);
    receiver.ShutDown(SHUT_RDWR);
    int sent = sender.Send("FOO", 3);
    UNIT_ASSERT(sent < 0);

    IOutputStream::TPart parts[] = {
        {"foo", 3},
        {"bar", 3},
    };
    sent = sender.SendV(parts, 2);
    UNIT_ASSERT(sent < 0);
}

void TSockTest::TestClose() {
    SOCKET socks[2];

    UNIT_ASSERT_EQUAL(SocketPair(socks), 0);
    TSocket receiver(socks[1]);

    UNIT_ASSERT_EQUAL(static_cast<SOCKET>(receiver), socks[1]);

#if defined _linux_
    UNIT_ASSERT_GE(fcntl(socks[1], F_GETFD), 0);
    receiver.Close();
    UNIT_ASSERT_EQUAL(fcntl(socks[1], F_GETFD), -1);
#else
    receiver.Close();
#endif

    UNIT_ASSERT_EQUAL(static_cast<SOCKET>(receiver), INVALID_SOCKET);
}

class TPollTest: public TTestBase {
    UNIT_TEST_SUITE(TPollTest);
    UNIT_TEST(TestPollInOut);
    UNIT_TEST_SUITE_END();

public:
    inline TPollTest() {
        srand(static_cast<unsigned int>(time(nullptr)));
    }

    void TestPollInOut();

private:
    sockaddr_in GetAddress(ui32 ip, ui16 port);
    SOCKET CreateSocket();
    SOCKET StartServerSocket(ui16 port, int backlog);
    SOCKET StartClientSocket(ui32 ip, ui16 port);
    SOCKET AcceptConnection(SOCKET serverSocket);
};

UNIT_TEST_SUITE_REGISTRATION(TPollTest);

sockaddr_in TPollTest::GetAddress(ui32 ip, ui16 port) {
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(ip);
    return addr;
}

SOCKET TPollTest::CreateSocket() {
    SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET) {
        ythrow yexception() << "Can not create socket (" << LastSystemErrorText() << ")";
    }
    return s;
}

SOCKET TPollTest::StartServerSocket(ui16 port, int backlog) {
    TSocketHolder s(CreateSocket());
    sockaddr_in addr = GetAddress(ntohl(INADDR_ANY), port);
    if (bind(s, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        ythrow yexception() << "Can not bind server socket (" << LastSystemErrorText() << ")";
    }
    if (listen(s, backlog) == SOCKET_ERROR) {
        ythrow yexception() << "Can not listen on server socket (" << LastSystemErrorText() << ")";
    }
    return s.Release();
}

SOCKET TPollTest::StartClientSocket(ui32 ip, ui16 port) {
    TSocketHolder s(CreateSocket());
    sockaddr_in addr = GetAddress(ip, port);
    if (connect(s, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        ythrow yexception() << "Can not connect client socket (" << LastSystemErrorText() << ")";
    }
    return s.Release();
}

SOCKET TPollTest::AcceptConnection(SOCKET serverSocket) {
    SOCKET connectedSocket = accept(serverSocket, nullptr, nullptr);
    if (connectedSocket == INVALID_SOCKET) {
        ythrow yexception() << "Can not accept connection on server socket (" << LastSystemErrorText() << ")";
    }
    return connectedSocket;
}

void TPollTest::TestPollInOut() {
#ifdef _win_
    const size_t socketCount = 1000;

    ui16 port = static_cast<ui16>(1300 + rand() % 97);
    TSocketHolder serverSocket = StartServerSocket(port, socketCount);

    ui32 localIp = ntohl(inet_addr("127.0.0.1"));

    TVector<TSimpleSharedPtr<TSocketHolder>> clientSockets;
    TVector<TSimpleSharedPtr<TSocketHolder>> connectedSockets;
    TVector<pollfd> fds;

    for (size_t i = 0; i < socketCount; ++i) {
        TSimpleSharedPtr<TSocketHolder> clientSocket(new TSocketHolder(StartClientSocket(localIp, port)));
        clientSockets.push_back(clientSocket);

        if (i % 5 == 0 || i % 5 == 2) {
            char buffer = 'c';
            if (send(*clientSocket, &buffer, 1, 0) == -1) {
                ythrow yexception() << "Can not send (" << LastSystemErrorText() << ")";
            }
        }

        TSimpleSharedPtr<TSocketHolder> connectedSocket(new TSocketHolder(AcceptConnection(serverSocket)));
        connectedSockets.push_back(connectedSocket);

        if (i % 5 == 2 || i % 5 == 3) {
            closesocket(*clientSocket);
            shutdown(*clientSocket, SD_BOTH);
        }
    }

    int expectedCount = 0;
    for (size_t i = 0; i < connectedSockets.size(); ++i) {
        pollfd fd = {(i % 5 == 4) ? INVALID_SOCKET : static_cast<SOCKET>(*connectedSockets[i]), POLLIN | POLLOUT, 0};
        fds.push_back(fd);
        if (i % 5 != 4) {
            ++expectedCount;
        }
    }

    int polledCount = poll(&fds[0], fds.size(), INFTIM);
    UNIT_ASSERT_EQUAL(expectedCount, polledCount);

    for (size_t i = 0; i < connectedSockets.size(); ++i) {
        short revents = fds[i].revents;
        if (i % 5 == 0) {
            UNIT_ASSERT_EQUAL(static_cast<short>(POLLRDNORM | POLLWRNORM), revents);
        } else if (i % 5 == 1) {
            UNIT_ASSERT_EQUAL(static_cast<short>(POLLOUT | POLLWRNORM), revents);
        } else if (i % 5 == 2) {
            UNIT_ASSERT_EQUAL(static_cast<short>(POLLHUP | POLLRDNORM | POLLWRNORM), revents);
        } else if (i % 5 == 3) {
            UNIT_ASSERT_EQUAL(static_cast<short>(POLLHUP | POLLWRNORM), revents);
        } else if (i % 5 == 4) {
            UNIT_ASSERT_EQUAL(static_cast<short>(POLLNVAL), revents);
        }
    }
#endif
}
