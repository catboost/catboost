#pragma once

#include "net_queue_stat.h"
#include "settings.h"
#include "udp_address.h"
#include "udp_debug.h"

#include <util/network/init.h>
#include <util/generic/ptr.h>
#include <util/generic/guid.h>
#include <library/cpp/threading/mux_event/mux_event.h>
#include <library/cpp/netliba/socket/socket.h>

namespace NNetliba_v12 {
    const ui64 MAX_PACKET_SIZE = 0x70000000;

    enum EHttpFlags {
        HF_HP_QUEUE = 128
    };

    struct TColors {
        TColors()
            : NetlibaRequestColor(0)
            , NetlibaResponseColor(0)
            , Priority(PP_NORMAL)
        {
        }

        virtual ~TColors() {
        }

        int GetRequestDataTos() const {
            return RequestTos.GetDataTos();
        }
        int GetResponseDataTos() const {
            return ResponseTos.GetDataTos();
        }
        int GetRequestAckTos() const {
            return RequestTos.GetAckTos();
        }
        int GetResponseAckTos() const {
            return ResponseTos.GetAckTos();
        }

        ui8 GetNetlibaRequestColor() const {
            return NetlibaRequestColor;
        }
        ui8 GetNetlibaResponseColor() const {
            return NetlibaResponseColor;
        }

        // return by value for incapsulation and future implementation changes
        TTos GetRequestTos() const {
            return RequestTos;
        }
        TTos GetResponseTos() const {
            return ResponseTos;
        }

        EPacketPriority GetPriority() const {
            return Priority;
        }

        void SetRequestDataTos(const int tos) {
            RequestTos.SetDataTos(tos);
        }
        void SetResponseDataTos(const int tos) {
            ResponseTos.SetDataTos(tos);
        }
        void SetRequestAckTos(const int tos) {
            RequestTos.SetAckTos(tos);
        }
        void SetResponseAckTos(const int tos) {
            ResponseTos.SetAckTos(tos);
        }

        void SetNetlibaRequestColor(const ui8 color) {
            NetlibaRequestColor = color;
        }
        void SetNetlibaResponseColor(const ui8 color) {
            NetlibaResponseColor = color;
        }

        void SetPriority(const EPacketPriority pp) {
            Priority = pp;
        }

    private:
        TTos RequestTos;
        TTos ResponseTos;
        ui8 NetlibaRequestColor;
        ui8 NetlibaResponseColor;
        EPacketPriority Priority;
    };

    inline bool operator==(const TColors& lhv, const TColors& rhv) {
        return lhv.GetRequestTos() == rhv.GetRequestTos() &&
               lhv.GetResponseTos() == rhv.GetResponseTos() &&
               lhv.GetNetlibaRequestColor() == rhv.GetNetlibaRequestColor() &&
               lhv.GetNetlibaResponseColor() == rhv.GetNetlibaResponseColor();
    }

    inline bool operator!=(const TColors& lhv, const TColors& rhv) {
        return !(lhv == rhv);
    }

    ///////////////////////////////////////////////////////////////////////////////

    struct TConnectionAddress: public TColors, public TConnectionSettings {
        // default ctor is only for containers
        TConnectionAddress() {
        }

        // non explicit for source compatibility
        TConnectionAddress(const TUdpAddress& address)
            : Address(address)
        {
        }

        TConnectionAddress(const TUdpAddress& address,
                           const TColors& colors,
                           const TConnectionSettings& connectionSettings)
            : TColors(colors)
            , TConnectionSettings(connectionSettings)
            , Address(address)
        {
        }

        const TUdpAddress& GetAddress() const {
            return Address;
        }

    private:
        TUdpAddress Address;
    };

    inline bool operator==(const TConnectionAddress& lhv, const TConnectionAddress& rhv) {
        return (TColors&)lhv == (TColors&)rhv &&
               (TConnectionSettings&)lhv == (TConnectionSettings&)rhv &&
               lhv.GetAddress() == rhv.GetAddress();
    }

    inline bool operator!=(const TConnectionAddress& lhv, const TConnectionAddress& rhv) {
        return !(lhv == rhv);
    }

    ///////////////////////////////////////////////////////////////////////////////

    struct TUdpRequest;
    struct TUdpHttpRequest: public TWithCustomAllocator {
        TUdpHttpRequest();
        ~TUdpHttpRequest();

        TGUID ReqId;
        TUdpAddress PeerAddress;
        TAutoPtr<TUdpRequest> DataHolder;

        TColors Colors;
        TString Url;
        TVector<char> Data;
    };

    struct TUdpHttpResponse: public TWithCustomAllocator {
        enum EResult {
            FAILED = 0,
            OK = 1,
            CANCELED = 2
        };

        TUdpHttpResponse();
        ~TUdpHttpResponse();

        TAutoPtr<TUdpRequest> DataHolder;
        TGUID ReqId;
        TUdpAddress PeerAddress;
        TVector<char> Data;
        EResult Ok;
        TString Error;
        bool IsHighPriority;
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
        virtual void SendRequest(const TConnectionAddress& addr, const TString& url, TVector<char>* data, const TGUID& reqId) = 0;
        TGUID SendRequest(const TConnectionAddress& addr, const TString& url, TVector<char>* data) {
            TGUID reqId;
            CreateGuid(&reqId);
            SendRequest(addr, url, data, reqId);
            return reqId;
        }
        virtual void CancelRequest(const TGUID& reqId) = 0;                                            //cancel request from requester side
        virtual void BreakRequest(const TGUID& reqId) = 0;                                             //break request-response from requester side
        virtual void SendResponse(const TGUID& reqId, TVector<char>* data, const TColors& colors) = 0; // TODO
        virtual void SendResponseLowPriority(const TGUID& reqId, TVector<char>* data, const TColors& colors) = 0;
        virtual TUdpHttpRequest* GetRequest() = 0;
        virtual TUdpHttpResponse* GetResponse() = 0;
        virtual bool GetRequestCancel(TGUID* req) = 0;
        virtual bool GetSendRequestAcc(TGUID* req) = 0;
        // sync mode
        virtual TUdpHttpResponse* Request(const TConnectionAddress& addr, const TString& url, TVector<char>* data) = 0;
        virtual TIntrusivePtr<TWaitResponse> WaitableRequest(const TConnectionAddress& addr, const TString& url, TVector<char>* data) = 0;
        //
        virtual TMuxEvent& GetAsyncEvent() = 0;
    };

    struct IRequester: public IRequestOps {
        virtual int GetPort() = 0;
        virtual void StopNoWait() = 0;
        virtual TUdpAddress GetPeerAddress(const TGUID& reqId) = 0;
        virtual void GetPendingDataSize(TRequesterPendingDataStats* res) = 0;
        virtual void GetAllPendingDataSize(TRequesterPendingDataStats* total, TColoredRequesterPendingDataStats* colored) = 0;
        virtual bool HasRequest(const TGUID& reqId) = 0;
        virtual TString GetDebugInfo() = 0;
        virtual void GetRequestQueueSize(TRequesterQueueStats* res) = 0;
        virtual IRequestOps* CreateSubRequester() = 0;
        virtual void EnableReportRequestCancel() = 0;
        virtual void EnableReportSendRequestAcc() = 0;
        virtual float GetPacketFailRate() const = 0;

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

    void EnableUseTOSforAcks(bool enable);
    void EnableROCE(bool f);

    void AbortOnFailedRequest(TUdpHttpResponse* answer);
    TString GetDebugInfo(const TUdpAddress& addr, double timeout = 60);
    void StopAllNetLibaThreads();

    // if heartbeat timeout is set and NetLibaHeartbeat() is not called for timeoutSec
    // then StopAllNetLibaThreads() will be called
    void SetNetLibaHeartbeatTimeout(double timeoutSec);
    void NetLibaHeartbeat();

    bool IsLocal(const TUdpAddress& addr);
}
