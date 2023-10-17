#include "par.h"
#include "par_network.h"
#include "compression.h"
#include "par_log.h"
#include "par_settings.h"
#include "par_locked_hash.h"

#include <library/cpp/digest/crc32c/crc32c.h>
#include <library/cpp/neh/multiclient.h>
#include <library/cpp/neh/neh.h>
#include <library/cpp/neh/rpc.h>
#include <library/cpp/netliba/v12/ib_low.h>
#include <library/cpp/netliba/v12/udp_http.h>
#include <library/cpp/threading/atomic/bool.h>

#include <util/generic/strbuf.h>
#include <util/network/sock.h>
#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/string/split.h>
#include <library/cpp/deprecated/atomic/atomic_ops.h>
#include <util/system/mutex.h>
#include <util/thread/factory.h>

namespace NPar {
    class TNehRequester: public IRequester {
        static const int DefaultRetries = 40;

    public:
        struct TSentNetQueryInfo: public TThrRefBase {
            NNeh::TMessage NehMessage;
            TGUID SentGUID;
            TString Url;
            int RetriesRest = DefaultRetries;
            TString ToString() const {
                TStringBuilder str;
                str << "reqId: " << GetGuidAsString(SentGUID)
                    << " " << NehMessage.Addr
                    << " " << Url
                    << " retries rest: " << RetriesRest;
                return str;
            }
            ~TSentNetQueryInfo() override {
            }
        };

        struct TSyncRequestsInfo: public TThrRefBase {
            TSyncRequestsInfo() {
                Event.Reset();
            }
            TManualEvent Event;
            TAutoPtr<TNetworkResponse> Response;
        };

        explicit TNehRequester(
            int port,
            TProcessQueryCancelCallback queryCancelCallback,
            TProcessQueryCallback queryCallback,
            TProcessReplyCallback replyCallback)
            : QueryCancelCallback(std::move(queryCancelCallback))
            , QueryCallback(std::move(queryCallback))
            , ReplyCallback(std::move(replyCallback))
        {
            NNeh::SetProtocolOption("tcp2/ServerOutputDeadline", "600s");
            MultiClient = NNeh::CreateMultiClient();
            MultiClientThread = SystemThreadFactory()->Run([this]() {
                MultiClientThreadLoopFunction();
            });
            ReceiverServices = NNeh::CreateLoop();
            if (port == 0) {
                ListenPort = GetFreeTcpPort();
            } else {
                ListenPort = port;
            }
            TNetworkAddress serverAddr("*", ListenPort);
            PAR_DEBUG_LOG << "Listening as " << serverAddr.GetNehAddr() << Endl;
            ReceiverServices->Add(serverAddr.GetNehAddr(), [this](const NNeh::IRequestRef& req) {
                NehServiceQueryCallback(req);
            });
            ReceiverServices->ForkLoop(5);
            PingerThread = SystemThreadFactory()->Run([this]() {
                PingerThreadFunction();
            });
        }

        int GetListenPort() const override {
            return ListenPort;
        }

        void NehServiceQueryCallback(const NNeh::IRequestRef& req) {
            CHROMIUM_TRACE_FUNCTION();

            PAR_DEBUG_LOG << "At " << GetHostAndPort() << " incoming req: " << req->Scheme() << " " << req->RemoteHost() << " " << req->Service() << Endl;
            if (req->Canceled()) {
                PAR_DEBUG_LOG << "At " << GetHostAndPort() << " incoming req: " << req->Scheme() << " " << req->RemoteHost() << " " << req->Service() << " query is canceled in flight, don't even try to launch it" << Endl;
                // if query is canceled in flight, don't even try to launch it
                return;
            }
            auto del1pos = req->Data().find('\xff');
            auto del2pos = req->Data().find('\xff', del1pos + 1);
            if (del1pos == TStringBuf::npos || del2pos == TStringBuf::npos) {
                req->SendError(NNeh::IRequest::BadRequest, "no \\xff delimiters found");
                return;
            }

            TGUID reqId;
            if (!GetGuid(ToString(req->Data().substr(0, del1pos)), reqId)) {
                req->SendError(NNeh::IRequest::BadRequest, "incorrect guid");
                return;
            }
            auto Url = req->Data().substr(del1pos + 1, del2pos - del1pos - 1);
            auto del3pos = req->Data().find('\xff', del2pos + 1);
            TVector<char> Data;
            if (del3pos != TStringBuf::npos) {
                auto crc32ref = FromString<ui32>(TStringBuf(req->Data().begin() + del2pos + 1, del3pos - del2pos - 1));
                auto dataSize = req->Data().end() - (req->Data().begin() + del3pos + 1);
                auto crc32actual = Crc32c(req->Data().begin() + del3pos + 1, dataSize);
                if (crc32actual != crc32ref) {
                    TStringBuilder errorString;
                    errorString << "Invalid crc32 for data, expected " << crc32ref << " got " << crc32actual;

                    PAR_DEBUG_LOG << "At " << GetHostAndPort() << ": " << errorString << Endl;
                    req->SendError(NNeh::IRequest::BadRequest, errorString);
                    return;
                }
                Data.assign(req->Data().begin() + del3pos + 1, req->Data().end());
            }
            QuickLZDecompress(&Data);
            PAR_DEBUG_LOG << "At " << GetHostAndPort() << " got request " << GetGuidAsString(reqId) << " service: " << Url << " data len: " << Data.size() << Endl;
            NNeh::TData ok = {'O', 'K'};
            req->SendReply(ok);
            if (Url == "_ping_") {
                return;
            } else if (Url == "_cancel_") {
                if (!IncomingRequestsInfo.EraseValueIfPresent(reqId)) {
                    return;
                }
                QueryCancelCallback(reqId);
            } else if (Url == "_reply_") {
                if (!RequestsInfo.EraseValueIfPresent(reqId)) {
                    return;
                }
                TAutoPtr<TNetworkResponse> httpResponse = new TNetworkResponse;
                httpResponse->ReqId = reqId;
                httpResponse->Data = std::move(Data);
                httpResponse->Status = TNetworkResponse::EStatus::Ok;
                auto directReplyNotifier = [&httpResponse](TIntrusivePtr<TSyncRequestsInfo>& syncRequestInfo) {
                    syncRequestInfo->Response = std::move(httpResponse);
                    syncRequestInfo->Event.Signal();
                };
                if (DirectRequestsInfo.LockedValueModify(reqId, directReplyNotifier)) { // Got direct request reply
                    PAR_DEBUG_LOG << "At " << GetHostAndPort() << " got reply for sync query " << GetGuidAsString(reqId) << Endl;
                } else { // got regular async query reply
                    ReplyCallback(httpResponse);
                }
            } else {
                if (IncomingRequestsInfo.EraseValueIfPresent(reqId)) {
                    PAR_DEBUG_LOG << "At " << GetHostAndPort() << "Duplicate http request received with reqId = " << GetGuidAsString(reqId) << Endl;
                }
                TAutoPtr<TNetworkRequest> httpRequest = new TNetworkRequest;
                httpRequest->ReqId = reqId;
                TString realUrl;
                int port;
                Split(Url, '@', realUrl, port);
                httpRequest->Url = realUrl;
                httpRequest->Data = std::move(Data);
                IncomingRequestsInfo.EmplaceValue(reqId, TNetworkAddress(req->RemoteHost(), port));
                QueryCallback(httpRequest);
            }
        }

        void MultiClientThreadLoopFunction() {
            NNeh::IMultiClient::TEvent ev;
            while (MultiClient->Wait(ev)) {
                CHROMIUM_TRACE_FUNCTION();
                if (ev.Type == NNeh::IMultiClient::TEvent::Response) {
                    NNeh::TResponseRef resp = ev.Hndl->Get();
                    TIntrusivePtr<TSentNetQueryInfo> infoDataPtr((TSentNetQueryInfo*)ev.UserData); // harakiri holder
                    if (resp->IsError()) {
                        ERROR_LOG << "query error: " << resp->GetErrorCode() << " " << resp->GetErrorText()
                                  << " info: " << infoDataPtr->ToString() << Endl;
                        --infoDataPtr->RetriesRest;
                        if (infoDataPtr->RetriesRest < 0) {
                            Singleton<TParLogger>()->OutputLogTailToCout();
                            Y_ABORT("got unexpected network error, no retries rest");
                        }
                        NNeh::IMultiClient::TRequest request(infoDataPtr->NehMessage,
                                                             Timeout(*infoDataPtr).ToDeadLine(), infoDataPtr.Release());
                        MultiClient->Request(request);
                    } else {
                        if (resp->Data != TStringBuf{"OK"}) {
                            ERROR_LOG << "query info: " << infoDataPtr->ToString() << Endl;
                            Y_ABORT("reply isn't OK");
                        }
                    }
                } else if (ev.Type == NNeh::IMultiClient::TEvent::Timeout) {
                    ev.Hndl->Cancel();                                                             // or handle somehow
                    TIntrusivePtr<TSentNetQueryInfo> infoDataPtr((TSentNetQueryInfo*)ev.UserData); // harakiri holder
                    ERROR_LOG << "timeout(" << Timeout(*infoDataPtr) << " seconds) exceed, info:" << infoDataPtr->ToString() << Endl;
                    --infoDataPtr->RetriesRest;
                    if (infoDataPtr->RetriesRest < 0) {
                        Singleton<TParLogger>()->OutputLogTailToCout();
                        Y_ABORT("got timeout for some request :(");
                    }
                    NNeh::IMultiClient::TRequest request(infoDataPtr->NehMessage,
                                                         Timeout(*infoDataPtr).ToDeadLine(), infoDataPtr.Release());
                    MultiClient->Request(request);
                } else {
                    Y_ASSERT(0 && "unexpected event");
                }
            }
        }

        void PingerThreadFunction() {
            while (Running) {
                THashSet<TNetworkAddress> requestedHosts;
                auto collectHosts = [&requestedHosts](const TGUID&, TNetworkAddress& addr) {
                    requestedHosts.insert(addr);
                };
                RequestsInfo.LockedIterateValues(collectHosts);
                if (requestedHosts) {
                    TGUID pingGuid;
                    CreateGuid(&pingGuid);
                    PAR_DEBUG_LOG << "From " << GetHostAndPort() << "Pinging " << requestedHosts.size() << " hosts" << Endl;
                    for (const auto& addr : requestedHosts) {
                        InternalSendQuery(addr, pingGuid, "_ping_", nullptr);
                    }
                }
                Sleep(TDuration::Seconds(2));
            }
        }

        ~TNehRequester() override {
            Running = false;
            PingerThread->Join();
            MultiClient->Interrupt();
            MultiClientThread->Join();
            ReceiverServices->SyncStopFork();
        }

        void SendRequest(const TGUID& reqId, const TNetworkAddress& address, const TString& url, TVector<char>* data) override {
            CHROMIUM_TRACE_FUNCTION();

            RequestsInfo.EmplaceValue(reqId, address);
            InternalSendQuery(address, reqId, url + "@" + ToString(ListenPort), data);
        }

        void CancelRequest(const TGUID& reqId) override { //cancel request from requester side
            CHROMIUM_TRACE_FUNCTION();
            TNetworkAddress addr;
            if (!RequestsInfo.ExtractValueIfPresent(reqId, addr)) {
                return;
            }
            InternalSendQuery(addr, reqId, "_cancel_", nullptr);
            TAutoPtr<TNetworkResponse> response = new TNetworkResponse();
            response->Status = TNetworkResponse::EStatus::Canceled;
            response->ReqId = reqId;
            ReplyCallback(response);
        }

        void SendResponse(const TGUID& reqId, TVector<char>* data) override {
            CHROMIUM_TRACE_FUNCTION();
            TNetworkAddress addr;
            if (!IncomingRequestsInfo.ExtractValueIfPresent(reqId, addr)) {
                PAR_DEBUG_LOG << "At " << GetHostAndPort() << " reply to " << addr.GetNehAddr() << " already sent or cancelled" << Endl;
                return;
            }

            PAR_DEBUG_LOG << "From " << GetHostAndPort() << " sending reply for " << GetGuidAsString(reqId) << " data len: " << (data ? data->size() : 0) << Endl;
            InternalSendQuery(addr, reqId, "_reply_", data);
        }

        TAutoPtr<TNetworkResponse> Request(const TNetworkAddress& address, const TString& url, TVector<char>* data) override {
            CHROMIUM_TRACE_FUNCTION();
            TIntrusivePtr<TSyncRequestsInfo> reqInfo = new TSyncRequestsInfo;
            TGUID reqId;
            CreateGuid(&reqId);
            DirectRequestsInfo.EmplaceValue(reqId, reqInfo);
            RequestsInfo.EmplaceValue(reqId, address);
            PAR_DEBUG_LOG << "From " << GetHostAndPort() << " sending request " << GetGuidAsString(reqId) << " url: " << url << " data len: " << (data ? data->size() : 0) << Endl;
            InternalSendQuery(address, reqId, url + "@" + ToString(ListenPort), data);
            reqInfo->Event.WaitI();
            Y_ABORT_UNLESS(DirectRequestsInfo.EraseValueIfPresent(reqId));
            return std::move(reqInfo->Response);
        }

    private:
        void InternalSendQuery(const TNetworkAddress& address, const TGUID& requestId, const TString& url, TVector<char>* data) {
            auto netQueryInfo = new TSentNetQueryInfo;
            netQueryInfo->SentGUID = requestId;
            netQueryInfo->Url = url;
            netQueryInfo->NehMessage = CreateNehMessage(address, requestId, url, data);
            NNeh::IMultiClient::TRequest request(netQueryInfo->NehMessage,
                                                 Timeout(*netQueryInfo).ToDeadLine(), netQueryInfo);
            MultiClient->Request(request);
        }

        NNeh::TMessage CreateNehMessage(const TNetworkAddress& address, const TGUID& requestId, const TString& url, TVector<char>* data) {
            CHROMIUM_TRACE_FUNCTION();

            NNeh::TMessage msg;
            msg.Addr = address.GetNehAddr();
            TStringOutput messageStream(msg.Data);
            messageStream << GetGuidAsString(requestId) << '\xff';
            messageStream << url << '\xff';
            if (data) {
                const size_t origLen = data->size();
                QuickLZCompress(data);
                const size_t compressedLen = data->size();
                auto val = Crc32c(data->data(), data->size());
                messageStream << val << '\xff';
                msg.Data.AppendNoAlias(data->data(), data->size());

                TVector<char>().swap(*data);
                PAR_DEBUG_LOG << "From " << GetHostAndPort() << " sending request " << GetGuidAsString(requestId) << " to " << address.GetNehAddr() << " service " << url << " data len: " << origLen << " (compressed: " << compressedLen << ")" << Endl;
            } else {
                PAR_DEBUG_LOG << "From " << GetHostAndPort() << " sending empty request " << GetGuidAsString(requestId) << " to " << address.GetNehAddr() << " service " << url << Endl;
            }
            return msg;
        }
        template <class TAddr = TSockAddrInetStream, class TSock = TInetStreamSocket>
        static int GetFreeFamilyPort(const char* ip) {
            TSock servSock;
            TAddr servAddr(ip, 0);
            servSock.CheckSock();
            TBaseSocket::Check(servSock.Bind(&servAddr));
            return servAddr.GetPort();
        }

        static int GetFreeTcpPort() {
            try {
                return GetFreeFamilyPort<TSockAddrInet6Stream, TInet6StreamSocket>("::");
            } catch (...) {
                return GetFreeFamilyPort<TSockAddrInetStream, TInetStreamSocket>("0.0.0.0");
            }
        }

        TDuration Timeout(const TSentNetQueryInfo& query) {
            const float netBytesPerSecond = 1e6; // network transactions run at least at 1Mbyte/sec in average for large messages
            int timeForDataTransfer = (int)(query.NehMessage.Data.size() / netBytesPerSecond);
            return TDuration::Seconds(Max(defaultNehTimeOut, timeForDataTransfer) * (TNehRequester::DefaultRetries - query.RetriesRest + 1));
        }

    private:
        TProcessQueryCancelCallback QueryCancelCallback;
        TProcessQueryCallback QueryCallback;
        TProcessReplyCallback ReplyCallback;

        TSpinLockedKeyValueStorage<TGUID, TNetworkAddress, TGUIDHash> RequestsInfo;
        TSpinLockedKeyValueStorage<TGUID, TNetworkAddress, TGUIDHash> IncomingRequestsInfo;
        TSpinLockedKeyValueStorage<TGUID, TIntrusivePtr<TSyncRequestsInfo>, TGUIDHash> DirectRequestsInfo;
        NNeh::IServicesRef ReplyService;
        NNeh::TMultiClientPtr MultiClient;
        TAutoPtr<IThreadFactory::IThread> MultiClientThread;
        TAutoPtr<IThreadFactory::IThread> PingerThread;
        NNeh::IServicesRef ReceiverServices;
        ui16 ListenPort = 0;
        NAtomic::TBool Running = true;
    };

    class TNetlibaRequester: public IRequester {
    public:
        explicit TNetlibaRequester(
            int listenPort,
            TProcessQueryCancelCallback queryCancelCallback,
            TProcessQueryCallback queryCallback,
            TProcessReplyCallback replyCallback)
            : QueryCancelCallback(std::move(queryCancelCallback))
            , QueryCallback(std::move(queryCallback))
            , ReplyCallback(std::move(replyCallback))
            , Requester(NNetliba_v12::CreateHttpUdpRequester(listenPort))
        {
            PAR_DEBUG_LOG << "Created netliba httpudp requester on port " << listenPort << Endl;
            Requester->EnableReportRequestCancel();

            ReceiverThread = SystemThreadFactory()->Run([this]() {
                ReceiveLoopFunc();
            });
        }
        ~TNetlibaRequester() override {
            Stopped = true;
            Requester->GetAsyncEvent().Signal();
            ReceiverThread->Join();
        }
        TAutoPtr<TNetworkResponse> Request(const TNetworkAddress& address, const TString& url, TVector<char>* data) override {
            QuickLZCompress(data);
            TAutoPtr<NNetliba_v12::TUdpHttpResponse> answer = Requester->Request(address.GetNetlibaAddr(), url, data);
            TAutoPtr<TNetworkResponse> resp = new TNetworkResponse;
            ProcessUdpHttpResponse(answer, resp);
            return resp;
        }

        void SendRequest(const TGUID& guid, const TNetworkAddress& address, const TString& url, TVector<char>* data) override {
            QuickLZCompress(data);
            Requester->SendRequest(address.GetNetlibaAddr(), url, data, guid);
        }

        void CancelRequest(const TGUID& reqId) override {
            Requester->CancelRequest(reqId);
        }
        void SendResponse(const TGUID& reqId, TVector<char>* data) override {
            QuickLZCompress(data);
            Requester->SendResponse(reqId, data, Colors);
        }
        int GetListenPort() const override {
            return Requester->GetPort();
        }

    private:
        void ReceiveLoopFunc() {
            while (!Stopped) {
                THolder<NNetliba_v12::TUdpHttpRequest> nlReq(Requester->GetRequest());
                if (nlReq) {
                    QuickLZDecompress(&nlReq->Data);
                    PAR_DEBUG_LOG << "Got request " << nlReq->Url.data() << Endl;
                    TAutoPtr<TNetworkRequest> netReq = new TNetworkRequest;
                    netReq->ReqId = nlReq->ReqId;
                    netReq->Url = nlReq->Url;
                    netReq->Data = std::move(nlReq->Data);
                    QueryCallback(netReq);
                }
                TAutoPtr<NNetliba_v12::TUdpHttpResponse> answer = Requester->GetResponse();
                if (answer) {
                    TAutoPtr<TNetworkResponse> response = new TNetworkResponse;
                    ProcessUdpHttpResponse(answer, response);

                    ReplyCallback(response);
                }
                TGUID canceledReq;
                if (Requester->GetRequestCancel(&canceledReq)) {
                    QueryCancelCallback(canceledReq);
                }
                Requester->GetAsyncEvent().Wait();
            }
        }

        void ProcessUdpHttpResponse(TAutoPtr<NNetliba_v12::TUdpHttpResponse>& answer, TAutoPtr<TNetworkResponse>& response) {
            NNetliba_v12::AbortOnFailedRequest(answer.Get()); //LEGACY
            response->ReqId = answer->ReqId;
            if (answer->Ok == NNetliba_v12::TUdpHttpResponse::CANCELED) {
                response->Status = TNetworkResponse::EStatus::Canceled;
            } else if (answer->Ok == NNetliba_v12::TUdpHttpResponse::OK) {
                QuickLZDecompress(&answer->Data);
                response->Data = std::move(answer->Data);
                response->Status = TNetworkResponse::EStatus::Ok;
            } else {
                response->Status = TNetworkResponse::EStatus::Failed;
                Y_ASSERT(0); //LEGACY
            }
        }

    private:
        TProcessQueryCancelCallback QueryCancelCallback;
        TProcessQueryCallback QueryCallback;
        TProcessReplyCallback ReplyCallback;

        NAtomic::TBool Stopped = false;
        THolder<NNetliba_v12::IRequester> Requester;
        TAutoPtr<IThreadFactory::IThread> ReceiverThread;
        const NNetliba_v12::TColors Colors;
    };

    TIntrusivePtr<IRequester> CreateRequester(
        int listenPort,
        IRequester::TProcessQueryCancelCallback processQueryCancelCallback,
        IRequester::TProcessQueryCallback processQueryCallback,
        IRequester::TProcessReplyCallback processReplyCallback)
    {
        auto& settings = TParNetworkSettings::GetRef();
        if (settings.RequesterType == TParNetworkSettings::ERequesterType::AutoDetect) {
            TIntrusivePtr<NNetliba_v12::TIBPort> ibPort = NNetliba_v12::GetIBDevice();
            if (ibPort) {
                DEBUG_LOG << "Detected IB port, using Netliba requester" << Endl;
                settings.RequesterType = TParNetworkSettings::ERequesterType::Netliba;
            } else {
                settings.RequesterType = TParNetworkSettings::ERequesterType::NEH;
            }
        }
        switch (settings.RequesterType) {
            case TParNetworkSettings::ERequesterType::NEH:
                DEBUG_LOG << "Creating NEH requester" << Endl;
                return MakeIntrusive<TNehRequester>(
                    listenPort,
                    std::move(processQueryCancelCallback),
                    std::move(processQueryCallback),
                    std::move(processReplyCallback));
            case TParNetworkSettings::ERequesterType::Netliba:
                DEBUG_LOG << "Creating Netliba requester" << Endl;
                return MakeIntrusive<TNetlibaRequester>(
                    listenPort,
                    std::move(processQueryCancelCallback),
                    std::move(processQueryCallback),
                    std::move(processReplyCallback));
            default:
                Y_ABORT("Unknown requester type");
        }
    }
}
