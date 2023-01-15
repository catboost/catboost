#include <util/thread/pool.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/testing/unittest/env.h>

#if defined(_unix_)
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/un.h>
#endif

#include "http2.h"
#include "http_common.h"
#include "neh.h"
#include "rpc.h"

using namespace NNeh;

Y_UNIT_TEST_SUITE(THttpListen) {
    class TRequestServer {
    public:
        void ServeRequest(const IRequestRef& req) {
            TDataSaver responseData;
            responseData << req->Data();
            auto* httpReq = dynamic_cast<IHttpRequest*>(req.Get());
            TString headers = "\r\nContent-Type: text/plain";
            httpReq->SendReply(responseData, headers);
            // Cerr << "SendReply:" << req->Data() << Endl;
        }
    };

    struct TServ {
        IServicesRef Services;
        ui16 ServerPort;
    };

    TServ CreateServices() {
        TServ serv;
        TString err;

        for (ui16 basePort = 20000; basePort < 40000; basePort += 100) {
            serv.Services = CreateLoop();
            try {
                serv.ServerPort = basePort;
                TStringStream addr;
                addr << "http://localhost:" << serv.ServerPort << "/echo";
                TRequestServer requestServer;
                serv.Services->Add(addr.Str(), requestServer);
                serv.Services->Listen();
                break;
            } catch (...) {
                serv.Services.Destroy();
                err = CurrentExceptionMessage();
            }
        }

        UNIT_ASSERT_C(serv.Services.Get(), err.data());
        return serv;
    }

    Y_UNIT_TEST(TEchoTest) {
        TServ serv = CreateServices();
        auto handle = NNeh::Request({"http://localhost:" + ToString(serv.ServerPort) + "/echo", "sample"});
        THolder<TResponse> response = handle->Wait();
        UNIT_ASSERT(response);
        UNIT_ASSERT_VALUES_EQUAL(response->Data, "sample");
    }

    #ifdef __unix__
        class TUnixSocketServer : public IObjectInQueue {
        public:
            TUnixSocketServer(const TUnixSocketPath& unixSocketPath)
                : UnixSocketPath_(unixSocketPath)
            {}

            ~TUnixSocketServer() override {
                unlink(UnixSocketPath_.Path.data());
            }

            void Process(void*) override {
                SOCKET socketFd = socket(AF_UNIX, SOCK_STREAM, 0);
                DestructIfTrue(socketFd == -1, {socketFd}, "socket");

                struct sockaddr_un sockAddr;
                sockAddr.sun_family = AF_UNIX;
                strcpy(sockAddr.sun_path, UnixSocketPath_.Path.data());
                unlink(UnixSocketPath_.Path.data());

                DestructIfTrue(
                    bind(socketFd, (struct sockaddr*)&sockAddr, sizeof(sockAddr)) == -1,
                    {socketFd},
                    "socket"
                );
                DestructIfTrue(listen(socketFd, 10) == -1, {socketFd}, "Listen");

                int clientFd = accept(socketFd, NULL, NULL);
                DestructIfTrue(clientFd < 0, {socketFd}, "accept");

                char buffer[1024]{0};
                ssize_t bytes_read = recv(clientFd, buffer, 1024, 0);
                if(bytes_read > 0) {
                    TString response = TStringBuilder() << "HTTP/1.1 200 OK\r\n"
                        << "Content-Length: " << bytes_read << "\r\n\r\n"
                        << TStringBuf(buffer);
                    send(clientFd, response.data(), response.size(), 0);
                }

                close(clientFd);
                close(socketFd);
            }

            void Bind() {
                SOCKET socketFd = socket(AF_UNIX, SOCK_STREAM, 0);
                DestructIfTrue(socketFd == -1, {socketFd}, "socket");

                struct sockaddr_un sockAddr;
                sockAddr.sun_family = AF_UNIX;
                strcpy(sockAddr.sun_path, UnixSocketPath_.Path.data());
                unlink(UnixSocketPath_.Path.data());

                DestructIfTrue(
                    bind(socketFd, (struct sockaddr*)&sockAddr, sizeof(sockAddr)) == -1,
                    {socketFd},
                    "socket"
                );
            }

        private:
            void DestructIfTrue(bool cond, const TVector<SOCKET>& sockets, const TString& errMsg) {
                if (cond) {
                    for (int socket : sockets) {
                        close(socket);
                    }

                    Cerr << errMsg << ": " << strerror(errno);
                    ythrow yexception() << errMsg << ": " << strerror(errno);
                }
            }

        private:
            TUnixSocketPath UnixSocketPath_;
        };

        Y_UNIT_TEST(TEchoTestUnixSocket) {
            TString requestData = "sample";
            TString unixSocketBasePath = "./unixsocket";
            TUnixSocketPath unixSocketPath(unixSocketBasePath);

            // Select unix socket path if it exists, else use unixSocketBasePath
            for (size_t postfix = 0; postfix < 10000; ++postfix) {
                TString unixSocketFullPath = unixSocketBasePath + ToString(postfix);
                if (access(unixSocketFullPath.data(), F_OK) != 0) {
                    unixSocketPath = TUnixSocketPath(unixSocketFullPath);
                    break;
                }
            }

            TThreadPool pool;
            pool.Start(1);

            THolder<TUnixSocketServer> server = MakeHolder<TUnixSocketServer>(unixSocketPath);
            if(!pool.Add(server.Get())) {
                ythrow yexception() << "Can not create unix domain socket echo server";
            }
            sleep(1);

            auto handle = NNeh::Request(NNeh::TMessage{"full+unix://[" + unixSocketPath.Path + "]/echo", requestData});
            THolder<TResponse> response = handle->Wait(TDuration::MilliSeconds(5000));

            pool.Stop();

            UNIT_ASSERT(response);
            UNIT_ASSERT_VALUES_EQUAL(response->Data, requestData);
        }

        Y_UNIT_TEST(TNoSuchUnixSocketFile) {
            TString requestData = "sample";
            TUnixSocketPath unixSocketPath("./unixsocket");

            auto handle = NNeh::Request(NNeh::TMessage{"full+unix://[" + unixSocketPath.Path + "]/echo", requestData});
            THolder<TResponse> response = handle->Wait(TDuration::MilliSeconds(5000));

            UNIT_ASSERT_VALUES_EQUAL(response->GetSystemErrorCode(), ENOENT);
        }

        Y_UNIT_TEST(TConnRefusedUnixSocket) {
            TString requestData = "sample";
            TUnixSocketPath unixSocketPath("./unixsocket");

            THolder<TUnixSocketServer> server = MakeHolder<TUnixSocketServer>(unixSocketPath);
            server->Bind();

            auto handle = NNeh::Request(NNeh::TMessage{"full+unix://[" + unixSocketPath.Path + "]/echo", requestData});
            THolder<TResponse> response = handle->Wait(TDuration::MilliSeconds(5000));

            UNIT_ASSERT_VALUES_EQUAL(response->GetSystemErrorCode(), ECONNREFUSED);
        }
    #endif
}
