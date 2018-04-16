#pragma once

#include <library/binsaver/bin_saver.h>
#include <library/netliba/v12/udp_address.h>

#include <util/system/hostname.h>
#include <util/generic/guid.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/ptr.h>
#include <util/string/builder.h>
#include <util/generic/maybe.h>

#include <functional>

namespace NPar {
    static const int defaultNehTimeOut = 10;

    class TNetworkAddress {
    public:
        using TPortNum = unsigned short;

        TNetworkAddress() = default;

        TNetworkAddress(const TString& address, TPortNum port)
            : Address(address)
            , Port(port)
        {
        }

        bool operator==(const TNetworkAddress& other) const {
            return this->Address == other.Address && this->Port == other.Port;
        }

        bool operator!=(const TNetworkAddress& other) const {
            return !(*this == other);
        }

        const TString& GetNehAddr() const {
            if (!CachedNehAddr) {
                CachedNehAddr = TStringBuilder() << "tcp2://" << Address << ":" << Port << "/matrixnet";
            }
            return CachedNehAddr;
        }

        const NNetliba_v12::TUdpAddress& GetNetlibaAddr() const {
            if (NetlibaAddr.Empty()) {
                NetlibaAddr = NNetliba_v12::CreateAddress(Address, Port);
            }
            return NetlibaAddr.GetRef();
        }

        size_t Hash() const {
            return Address.hash() ^ IntHash(Port);
        }

        SAVELOAD(Address, Port);

    private:
        TString Address;
        TPortNum Port;
        mutable TString CachedNehAddr;
        mutable TMaybe<NNetliba_v12::TUdpAddress> NetlibaAddr;
    };

    struct TNetworkRequest {
        TGUID ReqId;
        TString Url;
        TVector<char> Data;
    };

    struct TNetworkResponse {
        TGUID ReqId;
        TVector<char> Data;
        enum class EStatus {
            Canceled,
            Ok,
            Failed
        };
        EStatus Status;
    };

    class IRequester: public TThrRefBase {
    public:
        using ProcessQueryCancelCallback = std::function<void(const TGUID& canceledReqId)>;
        using ProcessQueryCallback = std::function<void(TAutoPtr<TNetworkRequest>& networkRequest)>;
        using ProcessReplyCallback = std::function<void(TAutoPtr<TNetworkResponse> response)>;

        virtual TAutoPtr<TNetworkResponse> Request(const TNetworkAddress& address, const TString& url, TVector<char>* data) = 0;   // send and wait for reply
        virtual void SendRequest(const TGUID& reqId, const TNetworkAddress& address, const TString& url, TVector<char>* data) = 0; // async send
        virtual void CancelRequest(const TGUID& reqId) = 0;                                                                        //cancel request from requester side
        virtual void SendResponse(const TGUID& reqId, TVector<char>* data) = 0;
        virtual int GetListenPort() const = 0;
        void SetQueryCancelCallback(ProcessQueryCancelCallback func) {
            QueryCancelCallback = func;
        }
        void SetQueryCallback(ProcessQueryCallback func) {
            QueryCallback = func;
        }
        void SetReplyCallback(ProcessReplyCallback func) {
            ReplyCallback = func;
        }
        virtual ~IRequester() = default;

        TString GetHostAndPort() {
            if (!hostAndPort) {
                TNetworkAddress myAddress(HostName(), GetListenPort());
                hostAndPort = myAddress.GetNehAddr();
            }
            return hostAndPort;
        }

    protected:
        ProcessQueryCancelCallback QueryCancelCallback;
        ProcessQueryCallback QueryCallback;
        ProcessReplyCallback ReplyCallback;

    private:
        TString hostAndPort;
    };

    TIntrusivePtr<IRequester> CreateRequester(int listenPort);
}

template <>
struct THash<NPar::TNetworkAddress> {
    inline size_t operator()(const NPar::TNetworkAddress& netAddr) const {
        return netAddr.Hash();
    }
};
