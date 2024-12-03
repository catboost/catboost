#pragma once

#include <util/generic/ptr.h>
#include <util/generic/guid.h>
#include <util/str_stl.h>

#include "net_request.h"
#include "settings.h"
#include "socket.h"
#include "udp_address.h"
#include "udp_debug.h"

namespace NNetliba_v12 {
    extern const float UDP_TRANSFER_TIMEOUT;
    class TRopeDataPacket;
    struct IPeerQueueStats;

    struct TTransfer {
        TIntrusivePtr<IConnection> Connection;
        ui64 Id;

        TTransfer()
            : Id(0)
        {
        }
        TTransfer(const TIntrusivePtr<IConnection>& connection, const ui64 id)
            : Connection(connection)
            , Id(id)
        {
        }
    };
    size_t TTransferHash(const TTransfer& t);
    bool operator==(const TTransfer& lhv, const TTransfer& rhv);
    bool operator!=(const TTransfer& lhv, const TTransfer& rhv);

    struct TSendResult {
        enum EResult {
            FAILED = 0,
            OK = 1,
            CANCELED = 2
        };

        TTransfer Transfer;
        EResult Ok;

        TSendResult()
            : Transfer()
            , Ok(FAILED)
        {
        }
        TSendResult(const TTransfer& transfer, EResult ok)
            : Transfer(transfer)
            , Ok(ok)
        {
        }
    };

    struct IUdpHost: public TThrRefBase {
        virtual TUdpRequest* GetRequest() = 0;
        virtual TIntrusivePtr<IConnection> Connect(const TUdpAddress& address, const TConnectionSettings& connectionSettings) = 0;
        virtual TIntrusivePtr<IConnection> Connect(const TUdpAddress& address, const TUdpAddress& myAddress, const TConnectionSettings& connectionSettings) = 0;
        virtual TTransfer Send(const TIntrusivePtr<IConnection>& connection, TAutoPtr<TRopeDataPacket> data, EPacketPriority pp, const TTos& tos, ui8 netlibaColor) = 0;
        virtual bool GetSendResult(TSendResult* res) = 0;
        virtual void Cancel(const TTransfer& transfer) = 0;
        virtual void Step() = 0; //Does nothing
        virtual void Wait(float seconds) = 0;
        virtual void CancelWait() = 0; // thread safe
        //callback in GetAllPendingDataSize and GetDebugInfo is called from internal thread,
        //do not block it!!!
        virtual void GetAllPendingDataSize(TRequesterPendingDataAllStatsCb cb) = 0;
        virtual void GetDebugInfo(TDebugStringCb cb) = 0;
        virtual float GetFailRate() const = 0;
        virtual bool IsLocal(const TUdpAddress& address) const = 0;
    };

    TIntrusivePtr<IUdpHost> CreateUdpHost(const int port, float udpTransferTimeout = UDP_TRANSFER_TIMEOUT);
    TIntrusivePtr<IUdpHost> CreateUdpHost(const TIntrusivePtr<ISocket>& socket, float udpTransferTimeout = UDP_TRANSFER_TIMEOUT);

    void SetUdpMaxBandwidthPerIP(float f);
    void SetUdpSlowStart(bool enable);
    void DisableIBDetection();
    void EnableXsPing();
}

template <>
struct THash<NNetliba_v12::TTransfer> {
    inline size_t operator()(const NNetliba_v12::TTransfer& s) const {
        return NNetliba_v12::TTransferHash(s);
    }
};
