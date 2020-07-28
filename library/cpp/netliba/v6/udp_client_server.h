#pragma once

#include <util/generic/ptr.h>
#include <util/generic/guid.h>
#include <library/cpp/netliba/socket/socket.h>

#include "udp_address.h"
#include "net_request.h"

namespace NNetliba {
    class TRopeDataPacket;
    struct TRequesterPendingDataStats;
    struct IPeerQueueStats;

    struct TSendResult {
        int TransferId;
        bool Success;
        TSendResult()
            : TransferId(-1)
            , Success(false)
        {
        }
        TSendResult(int transferId, bool success)
            : TransferId(transferId)
            , Success(success)
        {
        }
    };

    enum EPacketPriority {
        PP_LOW,
        PP_NORMAL,
        PP_HIGH
    };

    // Step should be called from one and the same thread
    // thread safety is caller responsibility
    struct IUdpHost: public TThrRefBase {
        virtual TRequest* GetRequest() = 0;
        // returns trasferId
        // Send() needs correctly computed crc32
        // crc32 is expected to be computed outside of the thread talking to IUdpHost to avoid crc32 computation delays
        // packetGuid provides packet guid, if packetGuid is empty then guid is generated
        virtual int Send(const TUdpAddress& addr, TAutoPtr<TRopeDataPacket> data, int crc32, TGUID* packetGuid, EPacketPriority pp) = 0;
        virtual bool GetSendResult(TSendResult* res) = 0;
        virtual void Step() = 0;
        virtual void IBStep() = 0;
        virtual void Wait(float seconds) = 0; // does not use UdpHost
        virtual void CancelWait() = 0;        // thread safe
        virtual void GetPendingDataSize(TRequesterPendingDataStats* res) = 0;
        virtual TString GetDebugInfo() = 0;
        virtual void Kill(const TUdpAddress& addr) = 0;
        virtual TIntrusivePtr<IPeerQueueStats> GetQueueStats(const TUdpAddress& addr) = 0;
    };

    TIntrusivePtr<IUdpHost> CreateUdpHost(int port);
    TIntrusivePtr<IUdpHost> CreateUdpHost(const TIntrusivePtr<NNetlibaSocket::ISocket>& socket);

    void SetUdpMaxBandwidthPerIP(float f);
    void SetUdpSlowStart(bool enable);
    void DisableIBDetection();
}
