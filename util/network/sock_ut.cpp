#include "sock.h"

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/threading/future/legacy_future.h>

#include <util/string/split.h>
#include <util/system/fs.h>

Y_UNIT_TEST_SUITE(TSocketTest) {
    Y_UNIT_TEST(InetDgramTest) {
        char buf[256];
        TSockAddrInetDgram servAddr(IpFromString("127.0.0.1"), 0);
        TSockAddrInetDgram cliAddr(IpFromString("127.0.0.1"), 0);
        TSockAddrInetDgram servFromAddr;
        TSockAddrInetDgram cliFromAddr;
        TInetDgramSocket cliSock;
        TInetDgramSocket servSock;
        cliSock.CheckSock();
        servSock.CheckSock();

        TBaseSocket::Check(cliSock.Bind(&cliAddr));
        TBaseSocket::Check(servSock.Bind(&servAddr));

        // client
        const char reqStr[] = "Hello, world!!!";
        TBaseSocket::Check(cliSock.SendTo(reqStr, sizeof(reqStr), &servAddr));

        // server
        TBaseSocket::Check(servSock.RecvFrom(buf, 256, &servFromAddr));
        UNIT_ASSERT(strcmp(reqStr, buf) == 0);
        const char repStr[] = "The World's greatings to you";
        TBaseSocket::Check(servSock.SendTo(repStr, sizeof(repStr), &servFromAddr));

        // client
        TBaseSocket::Check(cliSock.RecvFrom(buf, 256, &cliFromAddr));
        UNIT_ASSERT(strcmp(repStr, buf) == 0);
    }

    void RunLocalDgramTest(const char* localServerSockName, const char* localClientSockName) {
        char buf[256];
        TSockAddrLocalDgram servAddr(localServerSockName);
        TSockAddrLocalDgram cliAddr(localClientSockName);
        TSockAddrLocalDgram servFromAddr;
        TSockAddrLocalDgram cliFromAddr;
        TLocalDgramSocket cliSock;
        TLocalDgramSocket servSock;
        cliSock.CheckSock();
        servSock.CheckSock();

        TBaseSocket::Check(cliSock.Bind(&cliAddr), "bind client");
        TBaseSocket::Check(servSock.Bind(&servAddr), "bind server");

        // client
        const char reqStr[] = "Hello, world!!!";
        TBaseSocket::Check(cliSock.SendTo(reqStr, sizeof(reqStr), &servAddr), "send from client");

        // server
        TBaseSocket::Check(servSock.RecvFrom(buf, 256, &servFromAddr), "receive from client");
        UNIT_ASSERT(strcmp(reqStr, buf) == 0);
        const char repStr[] = "The World's greatings to you";
        TBaseSocket::Check(servSock.SendTo(repStr, sizeof(repStr), &servFromAddr), "send to client");

        // client
        TBaseSocket::Check(cliSock.RecvFrom(buf, 256, &cliFromAddr), "receive from server");
        UNIT_ASSERT(strcmp(repStr, buf) == 0);
    }

    Y_UNIT_TEST(LocalDgramTest) {
        const char* localServerSockName = "./serv_sock";
        const char* localClientSockName = "./cli_sock";
        RunLocalDgramTest(localServerSockName, localClientSockName);
        NFs::Remove(localServerSockName);
        NFs::Remove(localClientSockName);
    }

    template <class A, class S>
    void RunInetStreamTest(const char* ip) {
        char buf[256];
        A servAddr(ip, 0);
        A newAddr;
        S cliSock;
        S servSock;
        S newSock;
        cliSock.CheckSock();
        servSock.CheckSock();
        newSock.CheckSock();

        // server
        int yes = 1;
        CheckedSetSockOpt(servSock, SOL_SOCKET, SO_REUSEADDR, yes, "servSock, SO_REUSEADDR");
        TBaseSocket::Check(servSock.Bind(&servAddr), "bind");
        TBaseSocket::Check(servSock.Listen(10), "listen");

        // client
        TBaseSocket::Check(cliSock.Connect(&servAddr), "connect");

        // server
        TBaseSocket::Check(servSock.Accept(&newSock, &newAddr), "accept");

        // client
        const char reqStr[] = "Hello, world!!!";
        TBaseSocket::Check(cliSock.Send(reqStr, sizeof(reqStr)), "send");

        // server - new
        TBaseSocket::Check(newSock.Recv(buf, 256), "recv");
        UNIT_ASSERT(strcmp(reqStr, buf) == 0);
        const char repStr[] = "The World's greatings to you";
        TBaseSocket::Check(newSock.Send(repStr, sizeof(repStr)), "send");

        // client
        TBaseSocket::Check(cliSock.Recv(buf, 256), "recv");
        UNIT_ASSERT(strcmp(repStr, buf) == 0);
    }

    Y_UNIT_TEST(InetStreamTest) {
        RunInetStreamTest<TSockAddrInetStream, TInetStreamSocket>("127.0.0.1");
    }

    Y_UNIT_TEST(Inet6StreamTest) {
        RunInetStreamTest<TSockAddrInet6Stream, TInet6StreamSocket>("::1");
    }

    void RunLocalStreamTest(const char* localServerSockName) {
        char buf[256];
        TSockAddrLocalStream servAddr(localServerSockName);
        TSockAddrLocalStream newAddr;
        TLocalStreamSocket cliSock;
        TLocalStreamSocket servSock;
        TLocalStreamSocket newSock;
        cliSock.CheckSock();
        servSock.CheckSock();
        newSock.CheckSock();

        // server
        TBaseSocket::Check(servSock.Bind(&servAddr), "bind");
        TBaseSocket::Check(servSock.Listen(10), "listen");

        NThreading::TLegacyFuture<void> f([&]() {
            // server
            TBaseSocket::Check(servSock.Accept(&newSock, &newAddr), "accept");
        });

        // client
        TBaseSocket::Check(cliSock.Connect(&servAddr), "connect");

        f.Get();

        // client
        const char reqStr[] = "Hello, world!!!";
        TBaseSocket::Check(cliSock.Send(reqStr, sizeof(reqStr)), "send");

        // server - new
        TBaseSocket::Check(newSock.Recv(buf, 256), "recv");
        UNIT_ASSERT(strcmp(reqStr, buf) == 0);
        const char repStr[] = "The World's greatings to you";
        TBaseSocket::Check(newSock.Send(repStr, sizeof(repStr)), "send");

        // client
        TBaseSocket::Check(cliSock.Recv(buf, 256), "recv");
        UNIT_ASSERT(strcmp(repStr, buf) == 0);
    }

    Y_UNIT_TEST(LocalStreamTest) {
        const char* localServerSockName = "./serv_sock2";
        RunLocalStreamTest(localServerSockName);
        NFs::Remove(localServerSockName);
    }

    Y_UNIT_TEST(DetermingPath) {
        const TString connectionString = "/var/run/some.sock http://localhost/endpoint";

        TStringBuf sockPath, endpoint;
        StringSplitter(connectionString).Split(' ').SkipEmpty().CollectInto(&sockPath, &endpoint);

        TSockAddrLocal sal;
        sal.Set(sockPath);
        UNIT_ASSERT_STRINGS_EQUAL(sal.ToString(), "/var/run/some.sock");
    }
} // Y_UNIT_TEST_SUITE(TSocketTest)
