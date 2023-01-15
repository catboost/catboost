#pragma once

#include "udp_address.h"
#include "block_chain.h"
#include "net_request.h"
#include <util/generic/guid.h>
#include <util/system/hp_timer.h>

namespace NNetliba {
    struct TIBConnectInfo {
        TGUID SocketId;
        ui64 Subnet, Interface;
        int LID;
        int QPN;
    };

    struct TRCQueuePairHandshake {
        int QPN, PSN;
    };

    using TIBMsgHandle = i64;

    struct TIBSendResult {
        TIBMsgHandle Handle;
        bool Success;
        TIBSendResult()
            : Handle(0)
            , Success(false)
        {
        }
        TIBSendResult(TIBMsgHandle handle, bool success)
            : Handle(handle)
            , Success(success)
        {
        }
    };

    struct IIBPeer: public TThrRefBase {
        enum EState {
            CONNECTING,
            OK,
            FAILED,
        };
        virtual EState GetState() = 0;
    };

    struct IIBClientServer: public TThrRefBase {
        virtual TRequest* GetRequest() = 0;
        virtual TIBMsgHandle Send(TPtrArg<IIBPeer> peer, TRopeDataPacket* data, const TGUID& packetGuid) = 0;
        virtual bool GetSendResult(TIBSendResult* res) = 0;
        virtual bool Step(NHPTimer::STime tCurrent) = 0;
        virtual IIBPeer* ConnectPeer(const TIBConnectInfo& info, const TUdpAddress& peerAddr, const TUdpAddress& myAddr) = 0;
        virtual const TIBConnectInfo& GetConnectInfo() = 0;
    };

    IIBClientServer* CreateIBClientServer();
}
