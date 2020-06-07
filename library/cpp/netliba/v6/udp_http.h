#pragma once

#include "udp_address.h"
#include "udp_debug.h"
#include "net_queue_stat.h"

#include <util/network/init.h>
#include <util/generic/ptr.h>
#include <util/generic/guid.h>
#include <library/cpp/threading/mux_event/mux_event.h>
#include <library/cpp/netliba/socket/socket.h>

namespace NNetliba {
    const ui64 MAX_PACKET_SIZE = 0x70000000;

    struct TRequest;
    struct TUdpHttpRequest {
        TAutoPtr<TRequest> DataHolder;
        TGUID ReqId;
        TString Url;
        TUdpAddress PeerAddress;
        TVector<char> Data;

        ~TUdpHttpRequest();
    };

    struct TUdpHttpResponse {
        enum EResult {
            FAILED = 0,
            OK = 1,
            CANCELED = 2
        };
        TAutoPtr<TRequest> DataHolder;
        TGUID ReqId;
        TUdpAddress PeerAddress;
        TVector<char> Data;
        EResult Ok;
        TString Error;

        ~TUdpHttpResponse();
    };

    // vector<char> *data - vector will be cleared upon call
    struct IRequestOps: public TThrRefBase {
        class TWaitResponse: public TThrRefBase, public TNonCopyable {
            TGUID ReqId;
            TMuxEvent CompleteEvent;
            TUdpHttpResponse* Response;
            bool RequestSent;

            ~TWaitResponse() override {
                delete GetResponse();
            }

        public:
            TWaitResponse()
                : Response(nullptr)
                , RequestSent(false)
            {
            }
            void Wait() {
                CompleteEvent.Wait();
            }
            bool Wait(int ms) {
                return CompleteEvent.Wait(ms);
            }
            TUdpHttpResponse* GetResponse();
            bool IsRequestSent() const {
                return RequestSent;
            }
            void SetResponse(TUdpHttpResponse* r);
            void SetReqId(const TGUID& reqId) {
                ReqId = reqId;
            }
            const TGUID& GetReqId() {
                return ReqId;
            }
            void SetRequestSent() {
                RequestSent = true;
            }
        };

        // async
        virtual void SendRequest(const TUdpAddress& addr, const TString& url, TVector<char>* data, const TGUID& reqId) = 0;
        TGUID SendRequest(const TUdpAddress& addr, const TString& url, TVector<char>* data) {
            TGUID reqId;
            CreateGuid(&reqId);
            SendRequest(addr, url, data, reqId);
            return reqId;
        }
        virtual void CancelRequest(const TGUID& reqId) = 0; //cancel request from requester side
        virtual void BreakRequest(const TGUID& reqId) = 0;  //break request-response from requester side

        virtual void SendResponse(const TGUID& reqId, TVector<char>* data) = 0;
        virtual void SendResponseLowPriority(const TGUID& reqId, TVector<char>* data) = 0;
        virtual TUdpHttpRequest* GetRequest() = 0;
        virtual TUdpHttpResponse* GetResponse() = 0;
        virtual bool GetRequestCancel(TGUID* req) = 0;
        virtual bool GetSendRequestAcc(TGUID* req) = 0;
        // sync mode
        virtual TUdpHttpResponse* Request(const TUdpAddress& addr, const TString& url, TVector<char>* data) = 0;
        virtual TIntrusivePtr<TWaitResponse> WaitableRequest(const TUdpAddress& addr, const TString& url, TVector<char>* data) = 0;
        //
        virtual TMuxEvent& GetAsyncEvent() = 0;
    };

    struct IRequester: public IRequestOps {
        virtual int GetPort() = 0;
        virtual void StopNoWait() = 0;
        virtual TUdpAddress GetPeerAddress(const TGUID& reqId) = 0;
        virtual void GetPendingDataSize(TRequesterPendingDataStats* res) = 0;
        virtual bool HasRequest(const TGUID& reqId) = 0;
        virtual TString GetDebugInfo() = 0;
        virtual void GetRequestQueueSize(TRequesterQueueStats* res) = 0;
        virtual IRequestOps* CreateSubRequester() = 0;
        virtual void EnableReportRequestCancel() = 0;
        virtual void EnableReportSendRequestAcc() = 0;
        virtual TIntrusivePtr<IPeerQueueStats> GetQueueStats(const TUdpAddress& addr) = 0;

        ui64 GetPendingDataSize() {
            TRequesterPendingDataStats pds;
            GetPendingDataSize(&pds);
            return pds.InpDataSize + pds.OutDataSize;
        }
    };

    IRequester* CreateHttpUdpRequester(int port);
    IRequester* CreateHttpUdpRequester(const TIntrusivePtr<NNetlibaSocket::ISocket>& socket);

    void SetUdpMaxBandwidthPerIP(float f);
    void SetUdpSlowStart(bool enable);
    void SetCongCtrlChannelInflate(float inflate);

    void EnableUseTOSforAcks(bool enable);
    void EnableROCE(bool f);

    void AbortOnFailedRequest(TUdpHttpResponse* answer);
    TString GetDebugInfo(const TUdpAddress& addr, double timeout = 60);
    void Kill(const TUdpAddress& addr);
    void StopAllNetLibaThreads();

    // if heartbeat timeout is set and NetLibaHeartbeat() is not called for timeoutSec
    // then StopAllNetLibaThreads() will be called
    void SetNetLibaHeartbeatTimeout(double timeoutSec);
    void NetLibaHeartbeat();

    bool IsLocal(const TUdpAddress& addr);
}
