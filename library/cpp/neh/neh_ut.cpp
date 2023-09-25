#include "neh.h"
#include "rpc.h"
#include "utils.h"

#include "factory.h"
#include "udp.h"
#include "netliba.h"
#include "https.h"
#include "http2.h"
#include "inproc.h"
#include "tcp.h"
#include "tcp2.h"

#include <library/cpp/http/io/headers.h>

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/testing/unittest/env.h>

#include <util/stream/str.h>
#include <util/string/builder.h>

using namespace NNeh;

namespace {
    const TString INTERNAL_ERROR_DETAILS = "internal error\t\r\nтест42";
}

Y_UNIT_TEST_SUITE(Neh) {
    static const TString HTTPS_PARAMS = TStringBuilder()
                                        << "cert=" << ArcadiaSourceRoot() << TStringBuf("/library/cpp/neh/ut/server.pem")
                                        << ';'
                                        << "key=" << ArcadiaSourceRoot() << TStringBuf("/library/cpp/neh/ut/server.pem")
                                        << '@';

    class TServer {
    public:
        TServer(const TString& response)
            : R_(response)
        {
            ResetLastReqState();
        }

        void ServeRequest(const IRequestRef& req) {
            if (req->Data() == TStringBuf("test_cancel")) {
                ReceiveTestCancel = true;
                for (size_t i = 0; i < 100; ++i) {
                    if (req->Canceled()) {
                        Canceled = true;
                        return;
                    }
                    Sleep(TDuration::MilliSeconds(10));
                }
            } else if (req->Data() == TStringBuf("test_error_InternalError")) {
                req->SendError(IRequest::InternalError, INTERNAL_ERROR_DETAILS);
            }
            TData res(R_.data(), R_.data() + R_.size());
            req->SendReply(res);
        }

        void ResetLastReqState() {
            ReceiveTestCancel = false;
            Canceled = false;
        }

        TAtomicBool ReceiveTestCancel;
        TAtomicBool Canceled;

    private:
        TString R_;
    };

    struct TServiceInfo {
        TServiceInfo(const TStringBuf& p)
            : Protocol(p)
        {
        }

        TString Protocol;
        TStringStream Addr;
    };

    Y_UNIT_TEST(TTrivialRequests) {
        const TString response = "response data";
        TServer srv(response);

        //tested protocols
        TVector<std::pair<TStringBuf, TStringBuf>> protocols;
        protocols.push_back({NetLibaProtocol()->Scheme(), ""});
        protocols.push_back({UdpProtocol()->Scheme(), ""});
        protocols.push_back({SSLGetProtocol()->Scheme(), HTTPS_PARAMS});
        protocols.push_back({SSLPostProtocol()->Scheme(), HTTPS_PARAMS});
        protocols.push_back({Http1Protocol()->Scheme(), ""});
        protocols.push_back({Post1Protocol()->Scheme(), ""});
        protocols.push_back({InProcProtocol()->Scheme(), ""});
        protocols.push_back({TcpProtocol()->Scheme(), ""});
        protocols.push_back({Tcp2Protocol()->Scheme(), ""});

        IServicesRef svs;
        TVector<TServiceInfo> svsInfo;
        TString err;

        //in loop try run services(bind ports)
        for (ui16 basePort = 20000; basePort < 40000; basePort += 100) {
            svs = CreateLoop();
            try {
                ui16 port = basePort;
                for (size_t i = 0; i < protocols.size(); ++i) {
                    TServiceInfo si(protocols[i].first);
                    si.Addr << si.Protocol << TStringBuf("://") << protocols[i].second << ("localhost:") << port++ << TStringBuf("/test");
                    svsInfo.push_back(si);
                    svs->Add(si.Addr.Str(), srv);
                }
                svs->ForkLoop(2); //<< throw exception, if can not bind port
                break;
            } catch (...) {
                svs.Destroy();
                svsInfo.clear();
                err = CurrentExceptionMessage();
            }
        }

        UNIT_ASSERT_C(svs.Get(), err.data());
        UNIT_ASSERT_VALUES_EQUAL(svsInfo.size(), protocols.size());

        const TString request = "request_data";

        //check receiving responses
        SetHttpInputConnectionsLimits(1, 2);
        for (size_t i = 0; i < svsInfo.size(); ++i) {
            // check connection limits and other state-depended checks
            for (size_t j = 0; j < 32; ++j) {
                TResponseRef res = Request(TMessage(svsInfo[i].Addr.Str(), request))->Wait(TDuration::Seconds(3));
                UNIT_ASSERT_C(!!res, svsInfo[i].Addr.Str());
                UNIT_ASSERT_C(!res->IsError(), svsInfo[i].Addr.Str() + ": " + res->GetErrorText());
                UNIT_ASSERT_VALUES_EQUAL(res->Data, response);

                if (svsInfo[i].Protocol == "http") {
                    UNIT_ASSERT(res->Headers.Count());
                    THashSet<TStringBuf> requiredHeaders = {
                        "Content-Length",
                        "Connection",
                    };
                    for (const auto& header : res->Headers) {
                        requiredHeaders.erase(header.Name());
                        Cdbg << header.Name() << Endl;
                    }
                    UNIT_ASSERT(requiredHeaders.empty());
                }
            }
        }

        //check receiving error
        for (size_t i = 0; i < svsInfo.size(); ++i) {
            const TString& protocol = svsInfo[i].Protocol;
            if (protocol != TStringBuf("udp")        //udp can't detect request with unsupported service name (url-path)
                && protocol != TStringBuf("netliba") //some for netliba, tcp and inproc
                && protocol != TStringBuf("tcp") && protocol != TStringBuf("inproc")) {
                TString badAddr = svsInfo[i].Addr.Str() + "_unexisted_service";
                TResponseRef res = Request(TMessage(badAddr, request))->Wait(TDuration::Seconds(3));
                UNIT_ASSERT_C(!!res, badAddr);
                UNIT_ASSERT_C(res->IsError(), badAddr + ": " + res->Data);
                UNIT_ASSERT_VALUES_EQUAL(res->Request.Addr, badAddr);
            }
            /*
            if (svsInfo[i].Protocol != TStringBuf("udp") //udp can't detect request to unbinded port
                    && svsInfo[i].Protocol != TStringBuf("tcp") //some for tcp & inproc
                    && svsInfo[i].Protocol != TStringBuf("inproc"))
            {
                TString badAddr = svsInfo[i].Protocol + "://localhost:4/test";
                TResponseRef res = Request(TMessage(badAddr, request))->Wait(TDuration::Seconds(3));
                UNIT_ASSERT_C(!!res, badAddr);
                UNIT_ASSERT_C(res->IsError(), badAddr + ": " + res->Data);
                UNIT_ASSERT_VALUES_EQUAL(res->Request.Addr, badAddr);
            }
            */
        }

        //check receiving http error codes/messages
        for (size_t i = 0; i < svsInfo.size(); ++i) {
            const TString& protocol = svsInfo[i].Protocol;
            if (protocol != "http") {
                continue;
            }

            {
                TString badAddr = svsInfo[i].Addr.Str() + "_unexisted_service";
                TResponseRef res = Request(TMessage(badAddr, request))->Wait(TDuration::Seconds(3));
                UNIT_ASSERT_C(!!res, badAddr);
                UNIT_ASSERT_C(res->IsError(), badAddr + ": " + res->Data);
                UNIT_ASSERT_C(res->GetErrorType() == TError::ProtocolSpecific, "lost protocol specific error");
                UNIT_ASSERT_VALUES_EQUAL(res->GetErrorCode(), 404);
                UNIT_ASSERT_VALUES_EQUAL(res->GetErrorText(), "request failed(HTTP/1.1 404 Not found)");
                UNIT_ASSERT_VALUES_EQUAL(res->Request.Addr, badAddr);
            }

            {
                TResponseRef res = Request(TMessage(svsInfo[i].Addr.Str(), "test_error_InternalError"))->Wait(TDuration::Seconds(3));
                UNIT_ASSERT(!!res);
                UNIT_ASSERT(res->IsError());
                UNIT_ASSERT_C(res->GetErrorType() == TError::ProtocolSpecific, "lost protocol specific error");
                UNIT_ASSERT_VALUES_EQUAL(res->GetErrorCode(), 500);
                UNIT_ASSERT_VALUES_EQUAL(res->GetErrorText(),
                                         "request failed(HTTP/1.1 500 internal error...........42)");
            }
        }

        //check receiving inproc errors
        for (size_t i = 0; i < svsInfo.size(); ++i) {
            const TString& protocol = svsInfo[i].Protocol;
            if (protocol != "inproc") {
                continue;
            }

            {
                TResponseRef res = Request(TMessage(svsInfo[i].Addr.Str(), "test_error_InternalError"))->Wait(TDuration::Seconds(3));
                UNIT_ASSERT(!!res);
                UNIT_ASSERT(res->IsError());
                UNIT_ASSERT_C(res->GetErrorType() == TError::ProtocolSpecific, "lost protocol specific error");
                UNIT_ASSERT_VALUES_EQUAL(res->GetErrorCode(), 1);
                UNIT_ASSERT_VALUES_EQUAL(res->GetErrorText(), INTERNAL_ERROR_DETAILS);
            }
        }

        //check canceling request
        for (size_t i = 0; i < svsInfo.size(); ++i) {
            const TString& protocol = svsInfo[i].Protocol;
            if (protocol != TStringBuf("udp") //udp & tcp not support canceling request
                && protocol != TStringBuf("tcp")) {
                TInstant begin = TInstant::Now();
                THandleRef h = Request(TMessage(svsInfo[i].Addr.Str(), "test_cancel"));
                for (size_t t = 0; t < 50; ++t) { //give time for transmit request to service side
                    if (srv.ReceiveTestCancel) {
                        break;
                    }
                    Sleep(TDuration::MilliSeconds(50));
                }
                h->Cancel();
                TResponseRef res = h->Wait(TDuration::Seconds(3));
                UNIT_ASSERT_C(res->IsError(), "canceling request not cause error");
                UNIT_ASSERT_C(res->GetErrorType() == TError::Cancelled, "lost cancelled error type: "
                                                                            << int(res->GetErrorType()) << " protocol: " << protocol);
                TInstant end = TInstant::Now();
                UNIT_ASSERT_C(end > begin || end < begin + TDuration::MicroSeconds(100), "bad timing cancel request");
                for (size_t t = 0; t < 10; ++t) {
                    if (srv.Canceled) {
                        break;
                    }
                    Sleep(TDuration::MilliSeconds(10)); //give time for transmit canceling to service side
                }
                UNIT_ASSERT_C(srv.Canceled, svsInfo[i].Addr.Str());
                srv.ResetLastReqState();
            }
        }


        //check multirequester
        {
            auto requester = CreateRequester();

            UNIT_ASSERT(requester->IsEmpty());

            for (size_t i = 0; i < svsInfo.size(); ++i) {
                auto handle = Request(TMessage(svsInfo[i].Addr.Str(), request));
                requester->Add(handle);
            }

            UNIT_ASSERT(!requester->IsEmpty());

            for (size_t i = 0; i < svsInfo.size(); ++i) {
                THandleRef tmp;
                bool waitRes = requester->Wait(tmp);
                UNIT_ASSERT(waitRes);
                UNIT_ASSERT(tmp);
            }

            UNIT_ASSERT(requester->IsEmpty());
        }
    }

    Y_UNIT_TEST(TSetProtocolsOptions) {
        UNIT_ASSERT_EXCEPTION(SetProtocolOption("http2000/ConnectTimeout", "10ms"), yexception);
        UNIT_ASSERT(!SetProtocolOption("http2/CConnectTimeout", "10ms"));
        UNIT_ASSERT(SetProtocolOption("http2/ConnectTimeout", "10ms"));
        UNIT_ASSERT_EQUAL(THttp2Options::ConnectTimeout, TDuration::MilliSeconds(10));
        UNIT_ASSERT(SetProtocolOption("http2/ConnectTimeout", "2s"));
        UNIT_ASSERT_EQUAL(THttp2Options::ConnectTimeout, TDuration::Seconds(2));
        UNIT_ASSERT(SetProtocolOption("http2/UseResponseAsErrorMessage", "1"));
        UNIT_ASSERT_EQUAL(THttp2Options::UseResponseAsErrorMessage, true);
        UNIT_ASSERT(SetProtocolOption("http2/FullHeadersAsErrorMessage", "1"));
        UNIT_ASSERT_EQUAL(THttp2Options::FullHeadersAsErrorMessage, true);
        UNIT_ASSERT(SetProtocolOption("http2/RedirectionNotError", "1"));
        UNIT_ASSERT_EQUAL(THttp2Options::RedirectionNotError, true);
        UNIT_ASSERT(SetProtocolOption("http2/TcpKeepAlive", "true"));
        UNIT_ASSERT_EQUAL(THttp2Options::TcpKeepAlive, true);
        UNIT_ASSERT(SetProtocolOption("tcp2/InputBufferSize", "4999"));
        UNIT_ASSERT_EQUAL(TTcp2Options::InputBufferSize, 4999);
        UNIT_ASSERT(SetProtocolOption("tcp2/InputBufferSize", "4888"));
        UNIT_ASSERT_EQUAL(TTcp2Options::InputBufferSize, 4888);
        UNIT_ASSERT(SetProtocolOption("tcp2/ServerUseDirectWrite", "yes"));
        UNIT_ASSERT_EQUAL(TTcp2Options::ServerUseDirectWrite, true);
        UNIT_ASSERT(SetProtocolOption("tcp2/ServerUseDirectWrite", "no"));
        UNIT_ASSERT_EQUAL(TTcp2Options::ServerUseDirectWrite, false);
        UNIT_ASSERT(SetProtocolOption("https/CAFile", "file"));
        UNIT_ASSERT_EQUAL(THttpsOptions::CAFile, "file");
        UNIT_ASSERT(SetProtocolOption("https/CAPath", "path"));
        UNIT_ASSERT_EQUAL(THttpsOptions::CAPath, "path");
        UNIT_ASSERT(SetProtocolOption("https/CheckCertificateHostname", "yes"));
        UNIT_ASSERT_EQUAL(THttpsOptions::CheckCertificateHostname, true);
        UNIT_ASSERT(SetProtocolOption("https/RedirectionNotError", "yes"));
        UNIT_ASSERT_EQUAL(THttpsOptions::RedirectionNotError, true);
    }
}
