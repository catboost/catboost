#include <library/unittest/registar.h>
#include <library/unittest/env.h>

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
}
