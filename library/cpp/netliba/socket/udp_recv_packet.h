#pragma once

#include <util/generic/noncopyable.h>
#include <util/system/defaults.h>

#include <memory>
#include "allocator.h"

namespace NNetlibaSocket {
    constexpr int UDP_MAX_PACKET_SIZE = 8900;

    class TUdpHostRecvBufAlloc;
    struct TUdpRecvPacket: public TWithCustomAllocator {
        friend class TUdpHostRecvBufAlloc;
        int DataStart = 0, DataSize = 0;
        std::shared_ptr<char> Data;

    private:
        int ArraySize_ = 0;
    };

    ///////////////////////////////////////////////////////////////////////////////

    class TUdpHostRecvBufAlloc: public TNonCopyable {
    private:
        mutable TUdpRecvPacket* RecvPktBuf;

        static TUdpRecvPacket* Alloc() {
            return new TUdpRecvPacket();
        }

    public:
        static TUdpRecvPacket* Create(const int dataSize) {
            TUdpRecvPacket* result = Alloc();
            result->Data.reset(TCustomAllocator<char>().allocate(dataSize), [=](char* p) { TCustomAllocator<char>().deallocate(p, dataSize); }, TCustomAllocator<char>());
            result->ArraySize_ = dataSize;
            return result;
        }
        void SetNewPacket() const {
            RecvPktBuf = CreateNewPacket();
        }

    public:
        static TUdpRecvPacket* CreateNewSmallPacket(int dataSize) {
            return Create(dataSize);
        }
        static TUdpRecvPacket* CreateNewPacket() {
            return Create(UDP_MAX_PACKET_SIZE);
        }
        static TUdpRecvPacket* Clone(const TUdpRecvPacket* pkt) {
            TUdpRecvPacket* result = Alloc();
            result->DataStart = pkt->DataStart;
            result->DataSize = pkt->DataSize;
            result->Data = pkt->Data;
            result->ArraySize_ = pkt->ArraySize_;
            return result;
        }

        TUdpHostRecvBufAlloc() {
            SetNewPacket();
        }
        ~TUdpHostRecvBufAlloc() {
            delete RecvPktBuf;
        }

        TUdpRecvPacket* ExtractPacket() {
            TUdpRecvPacket* res = RecvPktBuf;
            SetNewPacket();
            return res;
        }

        int GetBufSize() const {
            return RecvPktBuf->ArraySize_;
        }
        char* GetDataPtr() const {
            return RecvPktBuf->Data.get();
        }
    };
}
