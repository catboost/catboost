#pragma once

#include <library/cpp/binsaver/bin_saver.h>
#include <library/cpp/netliba/v12/udp_address.h>

#include <util/generic/guid.h>
#include <util/generic/hash.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/string/builder.h>
#include <util/system/hostname.h>
#include <util/system/spinlock.h>

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
            , CachedNehAddr(TStringBuilder() << "tcp2://" << Address << ":" << Port << "/matrixnet")
        {
        }

        bool operator==(const TNetworkAddress& other) const {
            return this->Address == other.Address && this->Port == other.Port;
        }

        bool operator!=(const TNetworkAddress& other) const {
            return !(*this == other);
        }

        const TString& GetNehAddr() const {
            return CachedNehAddr;
        }

        const NNetliba_v12::TUdpAddress& GetNetlibaAddr() const {
            with_lock (NetlibaAddrLock) {
                if (NetlibaAddr.Empty()) {
                    NetlibaAddr = NNetliba_v12::CreateAddress(Address, Port);
                }
                return NetlibaAddr.GetRef();
            }
        }

        size_t Hash() const {
            return ComputeHash(Address) ^ IntHash(Port);
        }

        SAVELOAD(Address, Port, CachedNehAddr);

    private:
        TString Address;
        TPortNum Port;
        TString CachedNehAddr;

        mutable TAdaptiveLock NetlibaAddrLock;
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
        using TProcessQueryCancelCallback = std::function<void(const TGUID& canceledReqId)>;
        using TProcessQueryCallback = std::function<void(TAutoPtr<TNetworkRequest>& networkRequest)>;
        using TProcessReplyCallback = std::function<void(TAutoPtr<TNetworkResponse> response)>;

        virtual TAutoPtr<TNetworkResponse> Request(const TNetworkAddress& address, const TString& url, TVector<char>* data) = 0;   // send and wait for reply
        virtual void SendRequest(const TGUID& reqId, const TNetworkAddress& address, const TString& url, TVector<char>* data) = 0; // async send
        virtual void CancelRequest(const TGUID& reqId) = 0;                                                                        //cancel request from requester side
        virtual void SendResponse(const TGUID& reqId, TVector<char>* data) = 0;
        virtual int GetListenPort() const = 0;
        virtual ~IRequester() = default;

        TString GetHostAndPort() {
            TNetworkAddress myAddress(HostName(), GetListenPort());
            return myAddress.GetNehAddr();
        }
    };

    TIntrusivePtr<IRequester> CreateRequester(
        int listenPort,
        IRequester::TProcessQueryCancelCallback queryCancelCallback,
        IRequester::TProcessQueryCallback queryCallback,
        IRequester::TProcessReplyCallback replyCallback);
}

template <>
struct THash<NPar::TNetworkAddress> {
    inline size_t operator()(const NPar::TNetworkAddress& netAddr) const {
        return netAddr.Hash();
    }
};
