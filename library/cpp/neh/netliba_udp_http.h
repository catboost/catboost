#pragma once

#include <library/cpp/netliba/v6/net_queue_stat.h>
#include <library/cpp/netliba/v6/udp_address.h>
#include <library/cpp/netliba/v6/udp_debug.h>

#include <util/generic/guid.h>
#include <util/generic/ptr.h>
#include <util/network/init.h>
#include <util/system/event.h>

namespace NNetliba {
    struct TRequest;
}

namespace NNehNetliba {
    using namespace NNetliba;

    typedef TAutoPtr<TRequest> TRequestPtr;

    class TUdpHttpMessage {
    public:
        TUdpHttpMessage(const TGUID& reqId, const TUdpAddress& peerAddr);

        TGUID ReqId;
        TUdpAddress PeerAddress;
    };

    class TUdpHttpRequest: public TUdpHttpMessage {
    public:
        TUdpHttpRequest(TRequestPtr& dataHolder, const TGUID& reqId, const TUdpAddress& peerAddr);

        TString Url;
        TVector<char> Data;
    };

    class TUdpHttpResponse: public TUdpHttpMessage {
    public:
        enum EResult {
            FAILED = 0,
            OK = 1,
            CANCELED = 2
        };

        TUdpHttpResponse(TRequestPtr& dataHolder, const TGUID& reqId, const TUdpAddress& peerAddr, EResult result, const char* error);

        EResult Ok;
        TString Data;
        TString Error;
    };

    class IRequester: public TThrRefBase {
    public:
        virtual void EnableReportRequestCancel() = 0;
        virtual void EnableReportRequestAck() = 0;

        // vector<char> *data - vector will be cleared upon call
        virtual void SendRequest(const TUdpAddress&, const TString& url, const TString& data, const TGUID&) = 0;
        virtual void CancelRequest(const TGUID&) = 0;
        virtual void SendResponse(const TGUID&, TVector<char>* data) = 0;

        virtual void StopNoWait() = 0;
    };

    class IEventsCollector: public TThrRefBase {
    public:
        // move ownership request/response object to event collector
        virtual void AddRequest(TUdpHttpRequest*) = 0;
        virtual void AddResponse(TUdpHttpResponse*) = 0;
        virtual void AddCancel(const TGUID&) = 0;
        virtual void AddRequestAck(const TGUID&) = 0;
    };

    typedef TIntrusivePtr<IEventsCollector> IEventsCollectorRef;
    typedef TIntrusivePtr<IRequester> IRequesterRef;

    // throw exception, if can't bind port
    IRequesterRef CreateHttpUdpRequester(int port, const IEventsCollectorRef&, int physicalCpu = -1);
}
