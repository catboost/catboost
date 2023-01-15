#pragma once

#include <util/generic/guid.h>
#include <util/generic/ptr.h>
#include "udp_address.h"
#include "udp_debug.h"
#include "settings.h"
#include "block_chain.h"

#include <library/cpp/netliba/socket/allocator.h>

namespace NNetliba_v12 {
    struct IConnection: public TThrRefBase {
        virtual const TUdpAddress& GetAddress() const = 0;
        virtual const sockaddr_in6& GetWinsockAddress() const = 0;
        virtual const TGUID& GetGuid() const = 0;
        virtual TRequesterPendingDataStats GetPendingDataSize() const = 0;
        virtual bool IsAlive() const = 0;
        virtual TConnectionSettings GetSettings() const = 0;
        ~IConnection() override {
        }
    };

    // used by TUdpHost
    struct TUdpRequest: public TWithCustomAllocator {
        bool IsHighPriority;
        TIntrusivePtr<IConnection> Connection;
        TAutoPtr<TRopeDataPacket> Data;
    };

    // used by ib
    struct TIBRequest {
        TGUID ConnectionGuid;
        TAutoPtr<TRopeDataPacket> Data;
    };

}
