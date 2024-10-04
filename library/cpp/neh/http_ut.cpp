#include "http2.h"
#include "http_common.h"
#include "neh.h"
#include "rpc.h"

#include "https.h"

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/testing/unittest/tests_data.h>

#include <util/generic/buffer.h>
#include <util/network/endpoint.h>
#include <util/network/socket.h>
#include <util/stream/str.h>
#include <util/string/builder.h>
#include <util/generic/scope.h>

using namespace NNeh;

Y_UNIT_TEST_SUITE(NehHttp) {
    /**
        @brief  class with a method that responses to requests.
     */
    class TRequestServer {
    public:
        TRequestServer(std::function<void(const IRequestRef&)> f = [](const IRequestRef& req) {
            TDataSaver responseData;
            Sleep(TDuration::MilliSeconds(FromString<int>(req->Data())));
            responseData << req->Data();
            auto* httpReq = dynamic_cast<IHttpRequest*>(req.Get());
            TString headers = "\r\nContent-Type: text/plain";
            httpReq->SendReply(responseData, headers);
        })
            : F_(f)
        {
        }

        void ServeRequest(const IRequestRef& req) {
            F_(req);
            // Cerr << "SendReply:" << req->Data() << Endl;
        }
    private:
        std::function<void(const IRequestRef&)> F_;
    };

    /**
        @brief Auxiliary struct for tests with info about running services.
     */
    struct TServ {
        IServicesRef Services;
        ui16 ServerPort;
    };

    /**
        Creates service for serving request.

        @return ptr to IServices, port and error if occured. Tests failes if server could not be created.
     */
    TServ CreateServices(const TServiceFunction& f =
        [](const IRequestRef& req) {
            TDataSaver responseData;
            Sleep(TDuration::MilliSeconds(FromString<int>(req->Data())));
            responseData << req->Data();
            auto* httpReq = dynamic_cast<IHttpRequest*>(req.Get());
            TString headers = "\r\nContent-Type: text/plain";
            httpReq->SendReply(responseData, headers);
        }
    ) {
        TServ serv;
        TString err;

        //in loop try run service (bind port)
        for (ui16 basePort = 20000; basePort < 40000; basePort += 100) {
            serv.Services = CreateLoop();
            try {
                serv.ServerPort = basePort;
                TStringStream addr;
                addr << "http://localhost:" << serv.ServerPort << "/pipeline";

                serv.Services->Add(addr.Str(), f);
                serv.Services->ForkLoop(16); //<< throw exception, if can not bind port
                break;
            } catch (...) {
                serv.Services.Destroy();
                err = CurrentExceptionMessage();
            }
        }

        UNIT_ASSERT_C(serv.Services.Get(), err.data());
        return serv;
    }

    TString Request(SOCKET s, const TString& request) {
        const ssize_t nSend = send(s, request.data(), request.size(), 0);
        UNIT_ASSERT_C(nSend == ssize_t(request.size()), "can't write request to socket.");

        TVector<char> responseChars(1024, '\0');
        const ssize_t nRecv = recv(s, &responseChars[0], responseChars.size(), 0);
        UNIT_ASSERT_C(nRecv > 0, "can't read from socket.");

        TString response = "";
        for (ssize_t i = 0; i < nRecv; ++i) {
            response += responseChars[i];
        }

        return response;
    }

    Y_UNIT_TEST(TTestAnyHttpCodeIsAccepted) {
        auto responseWith523 = [](const IRequestRef& req) {
            auto* httpReq = dynamic_cast<IHttpRequest*>(req.Get());

            TData data;
            httpReq->SendReply(data, {}, 523);
        };

        TServ serv = CreateServices(responseWith523);
        NNeh::THandleRef handle = NNeh::Request(TStringBuilder() << "http://localhost:" << serv.ServerPort << "/pipeline?", nullptr);
        auto resp = handle->Wait();

        UNIT_ASSERT(resp);
        UNIT_ASSERT(resp->IsError());
        UNIT_ASSERT_EQUAL(resp->GetErrorCode(), 523);
    }

    Y_UNIT_TEST(TPipelineRequests) {
        TServ serv = CreateServices();

        // const TResolvedHost* host = CachedResolve(TResolveInfo("localhost", serv.ServerPort));
        TNetworkAddress addr("localhost", serv.ServerPort);
        TEndpoint ep(new NAddr::TAddrInfo(&*addr.Begin()));
        TSocketHolder s(socket(ep.SockAddr()->sa_family, SOCK_STREAM, 0));
        UNIT_ASSERT_C(s != INVALID_SOCKET, "can't create socket");
        const int errConnect = connect(s, ep.SockAddr(), (int)ep.SockAddrLen());
        UNIT_ASSERT_C(!errConnect, "can't connect socket");

        // build http requests/expected_responses
        TStringStream reqs;
        TStringStream expectedResponses;
        // first requests must has most big delay with respoding
        // (but server side must return responses in right order)
        for (int i = 500; i >= 0; i -= 50) {
            TString delay = ToString<int>(i); // response delay (millseconds)
            reqs << "GET /pipeline?" << delay << " HTTP/1.1\r\n"
                 << "\r\n";
            expectedResponses << "HTTP/1.1 200 Ok\r\n"
                              << "Content-Length: " << delay.size() << "\r\n"
                              << "Connection: Keep-Alive\r\n"
                              << "Content-Type: text/plain\r\n"
                              << "\r\n"
                              << delay;
        }
        // send requests compare responses with expected responses
        const ssize_t nSend = send(s, reqs.Data(), reqs.Size(), 0);
        UNIT_ASSERT_C(nSend == ssize_t(+reqs.Size()), "can't write reqs to socket");
        TVector<char> resp(expectedResponses.Size());
        size_t expectedCntBytes = expectedResponses.Size();
        SetSocketTimeout(s, 10, 0); // use as watchdog
        while (expectedCntBytes) {
            const ssize_t nRecv = recv(s, &resp[expectedResponses.Size() - expectedCntBytes], expectedCntBytes, 0);
            UNIT_ASSERT_C(nRecv > 0, "can't read data from socket");
            expectedCntBytes -= nRecv;
        }
        const TStringBuf responseBuf(resp.data(), resp.size());
        UNIT_ASSERT_C(responseBuf == expectedResponses.Str(), TString("has unexpected responses: ") + responseBuf);
    }

    /**
        @brief  Tests that neh closes http/1.0 connection after repling to it.
     */
    Y_UNIT_TEST(TClosedHttp10Connection) {
        TServ serv = CreateServices();

        // const TResolvedHost* host = CachedResolve(TResolveInfo("localhost", serv.ServerPort));
        const TNetworkAddress addr("localhost", serv.ServerPort);
        const TEndpoint ep(new NAddr::TAddrInfo(&*addr.Begin()));

        // form request.
        TStringStream request;
        request << "GET /pipeline?0 HTTP/1.0\r\n"
                << "\r\n";

        // form etalon response.
        TStringStream expectedResponse;
        expectedResponse << "HTTP/1.0 200 Ok\r\n"
                         << "Content-Length: 1\r\n"
                         << "Connection: close\r\n"
                         << "Content-Type: text/plain\r\n"
                         << "\r\n"
                         << "0";

        TSocketHolder s(socket(ep.SockAddr()->sa_family, SOCK_STREAM, 0));
        UNIT_ASSERT_C(s != INVALID_SOCKET, "can't create socket");
        SetSocketTimeout(s, 10, 0); // use as watchdog

        const int errConnect = connect(s, ep.SockAddr(), (int)ep.SockAddrLen());
        UNIT_ASSERT_C(!errConnect, "can't connect socket");

        const ssize_t nSend = send(s, request.Data(), request.Size(), 0);
        UNIT_ASSERT_C(nSend == ssize_t(request.Size()), "can't write request to socket.");
        TVector<char> response(expectedResponse.Size());
        size_t expectedCntBytes = expectedResponse.Size();
        while (expectedCntBytes) {
            const ssize_t nRecv = recv(s, &response[expectedResponse.Size() - expectedCntBytes], expectedCntBytes, 0);
            UNIT_ASSERT_C(nRecv > 0, "can't read data from socket.");
            expectedCntBytes -= nRecv;
        }
        const TStringBuf responseBuf(response.data(), response.size());
        UNIT_ASSERT_C(responseBuf == expectedResponse.Str(), TString("bad response: ") + responseBuf);

        /// Try to write to socket after waiting for a while to check that it's closed.
        Sleep(TDuration::MilliSeconds(500));

        // this test works fine.
        const int socket_fd = static_cast<int>(s);
        TBuffer buf(1);
        fd_set readset;
        FD_ZERO(&readset);
        FD_SET(static_cast<int>(s), &readset);
        timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 500000;
        const int selret = select(socket_fd + 1, &readset, nullptr, nullptr, &tv);
        UNIT_ASSERT_C(selret != -1, "select failed");
        UNIT_ASSERT_C(selret == 1, "select should return one fd that is ready to return 0 on recv call");
        const ssize_t nReadBytes = recv(s, buf.Data(), buf.Size(), 0);
        UNIT_ASSERT_C(nReadBytes == 0, "connection must be closed, but we did not get 0 as return value from it.");
    }

    Y_UNIT_TEST(TInputOutputDeadline) {
        // InputDeadline test
        {
            TServ serv = CreateServices();

            static constexpr size_t inputDeadline = 500;

            THttp2Options::Set("InputDeadline", ToString(inputDeadline) + "ms");
            UNIT_ASSERT_C(THttp2Options::InputDeadline == TDuration::MilliSeconds(inputDeadline), "InputDeadline was not set");

            // Request to immediate-answering server
            static constexpr size_t serverDelay1 = 0;
            TMessage noDelayMessage = TMessage::FromString("http://localhost:" + ToString(serv.ServerPort) + "/pipeline?" + ToString(serverDelay1));
            TResponseRef noDelayResponse = Request(noDelayMessage)->Wait();
            UNIT_ASSERT_C(!noDelayResponse->IsError(), noDelayResponse->GetErrorText());
            UNIT_ASSERT_VALUES_EQUAL(noDelayResponse->Data, ToString(serverDelay1));

            // Request to delayed-answering server
            static constexpr size_t serverDelay2 = 2500;
            TMessage delayMessage = TMessage::FromString("http://localhost:" + ToString(serv.ServerPort) + "/pipeline?" + ToString(serverDelay2));
            TResponseRef delayResponse = Request(delayMessage)->Wait();
            UNIT_ASSERT_C(delayResponse->IsError(), "Request was finished successfuly after deadline");
            // On windows error text is empty (?)
            if (!delayResponse->GetErrorText().empty()) {
                UNIT_ASSERT_VALUES_EQUAL(delayResponse->GetErrorText(), "Connection timed out");
            }

            THttp2Options::InputDeadline = TDuration::Max();
        }

        // OutputDeadline test
        {
            // Create dummy socket that will listen but not read
            ui16 serverPort = TPortManager().GetTcpPort();
            TNetworkAddress serverAddr("localhost", serverPort);
            TEndpoint serverEp(new NAddr::TAddrInfo(&*serverAddr.Begin()));
            TSocketHolder serverSocket(socket(serverEp.SockAddr()->sa_family, SOCK_STREAM, 0));
            UNIT_ASSERT_C(serverSocket != INVALID_SOCKET, "can't create server socket");
            {
                SOCKET sock = serverSocket;
                const int yes = 1;
                ::setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));
            }
            UNIT_ASSERT_C(bind(serverSocket, serverEp.SockAddr(), serverEp.SockAddrLen()) != SOCKET_ERROR, "can't bind server socket");
            UNIT_ASSERT_C(listen(serverSocket, 2) != SOCKET_ERROR, "can't listen on server socket");

            // Test system for max send buf size
            TSocket testSocket(socket(serverEp.SockAddr()->sa_family, SOCK_STREAM, 0));
            UNIT_ASSERT_C(testSocket != INVALID_SOCKET, "can't create test socket");
            UNIT_ASSERT_C(connect(testSocket, serverEp.SockAddr(), serverEp.SockAddrLen()) != -1, "can't establish test connection");
            SetNonBlock(testSocket);
            TString bigMessage(32 * 1024 * 1024, 'A');
            size_t bigMessageOffset = 0;
            ssize_t bytesSent = 0;
            do {
                bigMessageOffset += bytesSent;
                if (bigMessageOffset >= bigMessage.size()) {
                    bigMessage += bigMessage;
                }
                bytesSent = testSocket.Send(bigMessage.c_str() + bigMessageOffset, bigMessage.size() - bigMessageOffset);
            } while (bytesSent >= 0);
            // Double it's size to be sure that it will cause blocking write/send syscall
            bigMessage += bigMessage;
            testSocket.Close();

            static constexpr size_t outputDeadline = 500;

            THttp2Options::Set("OutputDeadline", ToString(outputDeadline) + "ms");
            UNIT_ASSERT_C(THttp2Options::OutputDeadline == TDuration::MilliSeconds(outputDeadline), "OutputDeadline was not set");
            TMessage msg("http://localhost:" + ToString(serverPort), bigMessage);
            TInstant requestStartTime = Now();
            TResponseRef res = Request(msg)->Wait();
            TInstant requestEndTime = Now();
            UNIT_ASSERT_C((requestEndTime - requestStartTime) >= TDuration::MilliSeconds(outputDeadline), "Request was finished before deadline");
            UNIT_ASSERT_C(res->IsError(), "Request was finished successfuly after deadline");
            // On windows error text is empty (?)
            if (!res->GetErrorText().empty()) {
                UNIT_ASSERT_VALUES_EQUAL(res->GetErrorText(), "Connection timed out");
            }

            THttp2Options::OutputDeadline = TDuration::Max();
        }
    }

    Y_UNIT_TEST(TTestLimitRequestsPerConnection) {
        const i32 LimitRequestsPerConnection = 4;
        NNeh::THttp2Options::Set("LimitRequestsPerConnection", ToString(LimitRequestsPerConnection));

        TServ serv = CreateServices();

        const TString testRequest = TStringBuilder()
            << "GET /pipeline?0 HTTP/1.0\r\n"
            << "Connection: Keep-Alive\r\n"
            << "\r\n";

        const TString keepAliveResponse = TStringBuilder()
            << "HTTP/1.0 200 Ok\r\n"
            << "Content-Length: 1\r\n"
            << "Content-Type: text/plain\r\n"
            << "\r\n"
            << "0";

        const TString closeResponse = TStringBuilder()
            << "HTTP/1.0 200 Ok\r\n"
            << "Content-Length: 1\r\n"
            << "Connection: close\r\n"
            << "Content-Type: text/plain\r\n"
            << "\r\n"
            << "0";

        const TNetworkAddress addr("localhost", serv.ServerPort);
        const TEndpoint ep(new NAddr::TAddrInfo(&*addr.Begin()));

        for (size_t cycle = 0; cycle < 3; ++cycle) {
            TSocketHolder s(socket(ep.SockAddr()->sa_family, SOCK_STREAM, 0));
            UNIT_ASSERT_C(s != INVALID_SOCKET, "can't create socket");

            const int errConnect = connect(s, ep.SockAddr(), (int)ep.SockAddrLen());
            UNIT_ASSERT_C(!errConnect, "can't connect socket");

            for (size_t reqId = 1; reqId <= LimitRequestsPerConnection; ++reqId) {
                const TString response = Request(s, testRequest);

                if (reqId % LimitRequestsPerConnection == 0) {
                    UNIT_ASSERT_C(response == closeResponse, TStringBuilder() << "Response" << reqId << ": " << response);
                } else {
                    UNIT_ASSERT_C(response == keepAliveResponse, TStringBuilder() << "Response" << reqId << ": " << response);
                }
            }

            TVector<char> buffer(1024, '\0');
            const ssize_t nRecv = recv(s, &buffer[0], buffer.size(), 0);
            UNIT_ASSERT_C(nRecv == 0, "socket is still readable.");
        }
    }

    Y_UNIT_TEST(TTestAnyResponseIsNotError) {
        auto f = [](const IRequestRef&) {
            throw yexception() << "error";
        };

        NNeh::THttp2Options::AnyResponseIsNotError = false;

        Y_DEFER {
            NNeh::THttp2Options::AnyResponseIsNotError = false;
        };

        {
            TServ serv = CreateServices(f);

            NNeh::THandleRef handle = NNeh::Request(TStringBuilder() << "http://localhost:" << serv.ServerPort << "/yandsearch?", nullptr);
            auto resp = handle->Wait();
            UNIT_ASSERT(resp);
            UNIT_ASSERT(resp->IsError());
        }

        {
            NNeh::THttp2Options::AnyResponseIsNotError = true;

            TServ serv = CreateServices(f);

            NNeh::THandleRef handle = NNeh::Request(TStringBuilder() << "http://localhost:" << serv.ServerPort << "/yandsearch?", nullptr);
            auto resp = handle->Wait();
            UNIT_ASSERT(resp);
            UNIT_ASSERT(!resp->IsError());
            UNIT_ASSERT_STRING_CONTAINS(resp->FirstLine, "404 Not found");
        }

        {
            NNeh::THttp2Options::AnyResponseIsNotError = true;

            TServ serv = CreateServices(f);

            NNeh::THandleRef handle = NNeh::Request(TStringBuilder() << "http://localhost:" << serv.ServerPort << "/pipeline?", nullptr);
            auto resp = handle->Wait();
            UNIT_ASSERT(resp);
            UNIT_ASSERT(!resp->IsError());
            UNIT_ASSERT_STRING_CONTAINS(resp->FirstLine, "HTTP/1.1 503 service unavailable");
        }
    }

    Y_UNIT_TEST(TTestRespectHostInHttpServer)  {
        TServ serv;
        TString err;
        serv.Services = CreateLoop();
        serv.ServerPort = 20000;
        TStringStream pingEndpoint;
        NNeh::THttp2Options::RespectHostInHttpServerNetworkAddress = true;
        Y_DEFER {
            NNeh::THttp2Options::RespectHostInHttpServerNetworkAddress = false;
        };
        auto responseFunc = TRequestServer([](const IRequestRef& req) {
            TDataSaver responseData;
            responseData << req->Data();
            auto* httpReq = dynamic_cast<IHttpRequest*>(req.Get());
            TString headers = "\r\nContent-Type: text/plain";
            httpReq->SendReply(responseData, headers);
        });
        pingEndpoint << "http://127.0.0.1:" << serv.ServerPort << "/ping";


        serv.Services->Add(pingEndpoint.Str(), responseFunc);
        serv.Services->ForkLoop(16); //<< throw exception, if can not bind port

        {
            NNeh::THandleRef handle = NNeh::Request(TStringBuilder() << "http://127.0.0.1:" << serv.ServerPort << "/ping?", nullptr);
            auto resp = handle->Wait();
            UNIT_ASSERT(resp);
            UNIT_ASSERT(!resp->IsError());
        }
        {
            NNeh::THandleRef handle = NNeh::Request(TStringBuilder() << "http://[::]:" << serv.ServerPort << "/ping?", nullptr);
            auto resp = handle->Wait();
            UNIT_ASSERT(resp);
            UNIT_ASSERT(resp->IsError());
        }
        UNIT_ASSERT_C(serv.Services.Get(), err.data());

    }
}
