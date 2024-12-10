#pragma once

#include <util/generic/guid.h>
#include <util/generic/maybe.h>
#include <util/system/defaults.h>
#include <util/system/yassert.h>
#include <limits>
#include "block_chain.h"
#include "net_acks.h"
#include "settings.h"
#include "udp_host_connection.h"
#include "udp_recv_packet.h"
#include "udp_socket.h"
#include "socket.h"

// THESE SMALL FUNCTIONS DEFINED IN HEADER FILE TO ALLOW INLINING THEM!

namespace NNetliba_v12 {
    using namespace NNetlibaSocket::NNetliba_v12;

    enum EMask {
        MASK_CLEAR = 31,
    };

    static_assert(CMD_END <= MASK_CLEAR + 1, "expect CMD_END <= MASK_CLEAR + 1"); // we use two most significant bits, see WriteDataPacketHeader and WriteTransferPacketHeader

    ///////////////////////////////////////////////////////////////////////////////

    inline bool IsValidCmd(const EUdpCmd cmd) {
        return CMD_BEGIN <= cmd && cmd < CMD_END;
    }

    inline bool IsDataCmd(const EUdpCmd cmd) {
        Y_ASSERT(IsValidCmd(cmd));
        switch (cmd) {
            case DATA:
            case DATA_SMALL:
                return true;
            default:
                return false;
        }
    }

    inline bool IsCancelTransferCmd(const EUdpCmd cmd) {
        return cmd == CANCEL_TRANSFER;
    }

    inline bool IsAckCmd(const EUdpCmd cmd) {
        Y_ASSERT(IsValidCmd(cmd));
        switch (cmd) {
            case ACK:
            case ACK_COMPLETE:
            case ACK_RESEND_NOSHMEM:
            case ACK_CANCELED:
                return true;
            default:
                return false;
        }
    }

    inline bool IsPingCmd(const EUdpCmd cmd) {
        Y_ASSERT(IsValidCmd(cmd));
        switch (cmd) {
            case PING:
            case PONG:
            case PONG_IB:
            case XS_PING:
            case XS_PONG:
                return true;
            default:
                return false;
        }
    }

    inline bool IsTransferCmd(const EUdpCmd cmd) {
        return IsDataCmd(cmd) || IsAckCmd(cmd) || IsCancelTransferCmd(cmd);
    }

    inline bool IsInConnectionCmd(const EUdpCmd cmd) {
        return IsTransferCmd(cmd) || IsPingCmd(cmd);
    }

    inline bool IsSystemCmd(const EUdpCmd cmd) {
        Y_ASSERT(IsValidCmd(cmd));
        switch (cmd) {
            case KILL:
                Y_ASSERT(!IsInConnectionCmd(cmd));
                return true;
            default:
                Y_ASSERT(IsInConnectionCmd(cmd));
                return false;
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////

    template <class T>
    inline T Read(const char** data) {
        T res = *(const T*)*data;
        *data += sizeof(res);
        return res;
    }

    template <class T>
    inline size_t Write(char** data, T res) {
        *(T*)*data = res;
        const size_t writen = sizeof(res);
        *data += writen;
        return writen;
    }

    inline void SetFlag(ui8* data, ui8 flag, bool enabled) {
        *data = enabled ? *data | flag : *data & ~flag;
    }

    //options per packet
    class TPacketOptions {
        static constexpr int PO_INFLATE_CONGESTION = 128;
        static constexpr int PO_USE_TOS_CONGESTION = 64;
        static constexpr int PO_NETLIBA_COLOR_IS_CUSTOM = 32;
        static constexpr int PO_ACK_TOS_IS_CUSTOM = 16;
        static constexpr int PO_TRANSFER_ID_IS_LONG = 8;
        static constexpr int PO_DO_NOT_USE_SHARED_MEMORY_FOR_LOCAL_CONNECTIONS = 4; //this is removed option
        static constexpr int PO_SMALL_MTU_USE_XS = 2;

        ui8 PacketFlags;
        TMaybe<ui8> AckTOS;
        TMaybe<ui8> Color;
        ui32 HigherTransferId;

    public:
        TPacketOptions()
            : PacketFlags(0)
            , HigherTransferId(0)
        {
        }

        bool Empty() const {
            return !PacketFlags;
        }

        bool Deserialize(const char** pktData, size_t* rest) {
            if (*rest == 0) {
                return false;
            }
            *rest -= sizeof(ui8);
            PacketFlags = Read<ui8>(pktData);
            if (PacketFlags & PO_NETLIBA_COLOR_IS_CUSTOM) {
                if (*rest < sizeof(ui8)) {
                    return false;
                }
                *rest -= sizeof(ui8);
                Color = Read<ui8>(pktData);
            }
            if (PacketFlags & PO_ACK_TOS_IS_CUSTOM) {
                if (*rest < sizeof(ui8)) {
                    return false;
                }
                *rest -= sizeof(ui8);
                AckTOS = Read<ui8>(pktData);
            }
            if (PacketFlags & PO_TRANSFER_ID_IS_LONG) {
                if (*rest < sizeof(ui32)) {
                    return false;
                }
                *rest -= sizeof(ui32);
                HigherTransferId = Read<ui32>(pktData);
            }
            return true;
        }

        size_t Serialize(char** buf) const {
            size_t writen = 0;
            writen += Write(buf, PacketFlags);
            if (PacketFlags & PO_NETLIBA_COLOR_IS_CUSTOM) {
                writen += Write(buf, Color.GetRef());
            }
            if (PacketFlags & PO_ACK_TOS_IS_CUSTOM) {
                writen += Write(buf, AckTOS.GetRef());
            }
            if (PacketFlags & PO_TRANSFER_ID_IS_LONG) {
                writen += Write(buf, HigherTransferId);
            }

            return writen;
        }

        TPacketOptions& SetInflateCongestion(bool enabled) {
            SetFlag(&PacketFlags, PO_INFLATE_CONGESTION, enabled);
            return *this;
        }

        TPacketOptions& SetTosCongestion(bool enabled) {
            SetFlag(&PacketFlags, PO_USE_TOS_CONGESTION, enabled);
            return *this;
        }

        TPacketOptions& SetDisableSharedMemory(bool enabled) {
            SetFlag(&PacketFlags, PO_DO_NOT_USE_SHARED_MEMORY_FOR_LOCAL_CONNECTIONS, enabled);
            return *this;
        }

        TPacketOptions& SetAckTos(ui8 tos) {
            PacketFlags |= PO_ACK_TOS_IS_CUSTOM;
            AckTOS = tos;
            return *this;
        }

        TPacketOptions& SetColor(ui8 color) {
            PacketFlags |= PO_NETLIBA_COLOR_IS_CUSTOM;
            Color = color;
            return *this;
        }

        TPacketOptions& SetSmallMtuUseXs(bool enabled) {
            SetFlag(&PacketFlags, PO_SMALL_MTU_USE_XS, enabled);
            return *this;
        }

        ui32 EncodeTransferId(ui64 transferId) {
            if (transferId > std::numeric_limits<ui32>::max()) {
                PacketFlags |= PO_TRANSFER_ID_IS_LONG;
                HigherTransferId = transferId >> 32;
            }
            return transferId & std::numeric_limits<ui32>::max();
        }

        bool IsInflateCongestion() const {
            return PacketFlags & PO_INFLATE_CONGESTION;
        }

        bool IsTosCongestion() const {
            return PacketFlags & PO_USE_TOS_CONGESTION;
        }

        bool IsSharedMemoryDisabled() const {
            return PacketFlags & PO_DO_NOT_USE_SHARED_MEMORY_FOR_LOCAL_CONNECTIONS;
        }

        const TMaybe<ui8>& GetAckTos() const {
            return AckTOS;
        }

        const TMaybe<ui8>& GetColor() const {
            return Color;
        }

        bool IsSmallMtuUseXs() const {
            return PacketFlags & PO_SMALL_MTU_USE_XS;
        }

        ui64 GetHigherTransferId() const {
            return HigherTransferId;
        }
    };

    //Transfer Options are valid only for "zero" packet in transfer
    class TTransferOptions {
        static constexpr int TO_HP_QUEUE = 128;
        static constexpr int TO_HAS_SHM = 64;

        ui8 TransferFlags;
        ui32 SharedMemorySize;
        TGUID SharedMemoryGuid;

    public:
        TTransferOptions()
            : TransferFlags(0)
            , SharedMemorySize(0)
        {
        }

        bool Empty() const {
            return !TransferFlags;
        }

        bool Deserialize(const char** pktData, size_t* rest) {
            if (*rest == 0) {
                return false;
            }
            *rest -= sizeof(ui8);
            TransferFlags = Read<ui8>(pktData);
            if (TransferFlags & TO_HAS_SHM) {
                const size_t shmDescSize = sizeof(TGUID) + sizeof(ui32);
                if (*rest < shmDescSize) {
                    return false;
                }
                *rest -= shmDescSize;
                SharedMemoryGuid = Read<TGUID>(pktData);
                SharedMemorySize = Read<ui32>(pktData);
            }
            return true;
        }

        size_t Serialize(char** buf) const {
            size_t writen = 0;
            writen += Write(buf, TransferFlags);
            if (TransferFlags & TO_HAS_SHM) {
                writen += Write(buf, SharedMemoryGuid);
                writen += Write(buf, SharedMemorySize);
            }
            return writen;
        }

        TTransferOptions& SetSharedMemory(ui32 sz, const TGUID& guid) {
            TransferFlags |= TO_HAS_SHM;
            SharedMemorySize = sz;
            SharedMemoryGuid = guid;
            return *this;
        }

        TTransferOptions& SetHighPriority(bool enabled) {
            SetFlag(&TransferFlags, TO_HP_QUEUE, enabled);
            return *this;
        }

        bool IsSharedMemoryRequired() const {
            return TransferFlags & TO_HAS_SHM;
        }

        const TGUID& GetSharedMemoryGuid() const {
            return SharedMemoryGuid;
        }

        size_t GetSharedMemorySize() const {
            return SharedMemorySize;
        }

        bool IsHighPriority() const {
            return TransferFlags & TO_HP_QUEUE;
        }

        ui8 GetFlags() const {
            return TransferFlags;
        }
    };

    //
    // How to use it?
    // If You just need bit flag use Transfer Or Packet options
    // If you need one more field:
    //   - Create class (lock look at TPacketOptions)
    //   - Add bit flag in the TOptionsVector
    //   - Modify Serialize/Deserialize function
    class TOptionsVector {
        static constexpr int OV_HAS_PACKET_OPTIONS = 128;
        static constexpr int OV_HAS_TRANSFER_OPTIONS = 64;

    public:
        TOptionsVector() {
        }
        bool Deserialize(const char** pktData, const size_t sizeLimit) {
            if (sizeLimit == 0) {
                return false;
            }
            const size_t sz = Read<ui8>(pktData); //Size of vector
            const char* vectorStart = *pktData;
            if (sizeLimit <= sz) {
                fprintf(stderr, "aux size (%d) greater than packet (%d)\n", (int)sz, (int)sizeLimit);
                return false;
            }

            const ui8 bitMap = Read<ui8>(pktData);
            //allowedToParseInit - is max size of data which we can read in parser
            //allowedToPArseCur - is rest of data which we can read in parser
            const size_t allowedToParseInit = sizeLimit - sizeof(bitMap) - sizeof(ui8);
            size_t allowedToParseCur = allowedToParseInit;
            if (bitMap & OV_HAS_PACKET_OPTIONS) {
                if (!PacketOpt.Deserialize(pktData, &allowedToParseCur)) {
                    fprintf(stderr, "can`t deserialize OV_HAS_PACKET_OPTIONS\n");
                    Y_ASSERT(false);
                    return false;
                }
            }
            if (bitMap & OV_HAS_TRANSFER_OPTIONS) {
                if (!TransferOpt.Deserialize(pktData, &allowedToParseCur)) {
                    fprintf(stderr, "can`t deserialize OV_HAS_TRANSFER_OPTIONS\n");
                    Y_ASSERT(false);
                    return false;
                }
            }
            //read - data which has been read from vector. bitMap is part of vector!
            const size_t read = allowedToParseInit - allowedToParseCur + sizeof(bitMap);
            //TODO: remove this assert and write ut
            Y_ASSERT((size_t)(*pktData - vectorStart) == read);
            if (sz > read) {
                const size_t rest = sz - read;
                //fprintf(stderr, "we have more than processed optionsMap, next: %d bytes will be ignored\n", (int)rest);
                *pktData += rest;
            }
            return true;
        }

        size_t Serialize(char** buf) {
            //store start position of vector to write size
            char* start = *buf;
            ui8 map = 0;
            //place for size (ui8) and map
            *buf += sizeof(ui8) + sizeof(map);
            //map also is element of vector
            size_t written = sizeof(map);
            if (!PacketOpt.Empty()) {
                map |= OV_HAS_PACKET_OPTIONS;
                written += PacketOpt.Serialize(buf);
            }
            if (!TransferOpt.Empty()) {
                map |= OV_HAS_TRANSFER_OPTIONS;
                written += TransferOpt.Serialize(buf);
            }
            Y_ABORT_UNLESS(written < 255, "TOptionsVector is too long\n");
            Write(&start, (ui8)written);
            Write(&start, map);
            return written;
        }

        TPacketOptions PacketOpt;
        TTransferOptions TransferOpt;
    };

    ///////////////////////////////////////////////////////////////////////////////

    inline void WriteBasicPacketHeader(char** buf, const ui8 cmd) {
        Y_ASSERT(IsValidCmd(EUdpCmd(cmd & MASK_CLEAR)));
        Write(buf, cmd);
    }

    inline bool ReadBasicPacketHeader(const char** pktData, const char* pktEnd, EUdpCmd* cmd, ui8* originalCmd) {
        *originalCmd = Read<ui8>(pktData);
        *cmd = EUdpCmd(*originalCmd & MASK_CLEAR);
        /*
        Cerr << "Read packet header: originalCmd=" << (*originalCmd & 255)
            << ", cmd=" << (*originalCmd & MASK_CLEAR)
            << ", pktData<=pktEnd:" << (*pktData <= pktEnd)
            << ", IsValidCmd(cmd)=" << IsValidCmd(*cmd)
            << Endl;
        */
        return *pktData <= pktEnd && IsValidCmd(*cmd);
    }

    ///////////////////////////////////////////////////////////////////////////////

    inline void WriteInConnectionPacketHeader(char** buf, const EUdpCmd cmd, const TConnection* connection, TOptionsVector* opt) {
        Y_ASSERT(IsInConnectionCmd(cmd));
        Y_ASSERT(!connection->GetGuid().IsEmpty());

        if (connection->GetSettings().GetInflateCongestion()) {
            opt->PacketOpt.SetInflateCongestion(true);
        }
        if (connection->GetSettings().GetUseTosCongestionAlgo()) {
            opt->PacketOpt.SetTosCongestion(true);
        }
        //this option was removed, but we have to set this flag for old versions
        opt->PacketOpt.SetDisableSharedMemory(true);
        // we always set this flag so that versions supporting XSMALL mtu value
        // would switch to it
        opt->PacketOpt.SetSmallMtuUseXs(true);

        // in "all default" case we don't need this extra byte flags
        WriteBasicPacketHeader(buf, (ui8)cmd);
        Write(buf, connection->GetGuid());
        Write(buf, connection->GetThisSideGuid());

        opt->Serialize(buf);
    }

    inline bool ReadInConnectionPacketHeaderTail(const char** pktData, const char* pktEnd, const ui8 originalCmd,
                                                 TGUID* connectionGuid, TGUID* thatSideGuid,
                                                 TConnectionSettings* settings, TOptionsVector* opt) {
        Y_ASSERT(IsInConnectionCmd(EUdpCmd(originalCmd & MASK_CLEAR)));

        *connectionGuid = Read<TGUID>(pktData);
        *thatSideGuid = Read<TGUID>(pktData);

        if (!opt->Deserialize(pktData, pktEnd - *pktData)) {
            return false;
        }

        *settings = TConnectionSettings();
        settings->SetInflateCongestion(opt->PacketOpt.IsInflateCongestion());
        settings->SetUseTosCongestionAlgo(opt->PacketOpt.IsTosCongestion());
        //settings->SetUseSharedMemoryForLocalConnections(!opt->PacketOpt.IsSharedMemoryDisabled());

        return *pktData <= pktEnd;
    }

    ///////////////////////////////////////////////////////////////////////////////

    inline void WriteTransferPacketHeader(char** buf, const EUdpCmd cmd, const TConnection* connection, ui64 transferId, TOptionsVector* opt) {
        Y_ASSERT(IsTransferCmd(cmd));
        Y_ASSERT(!connection->GetGuid().IsEmpty() && transferId);
        const ui32 transferLower = opt->PacketOpt.EncodeTransferId(transferId);

        WriteInConnectionPacketHeader(buf, cmd, connection, opt);
        Write(buf, transferLower);
    }

    ///////////////////////////////////////////////////////////////////////////////

    inline void WriteDataPacketHeader(char** buf, const EUdpCmd cmd, const TConnection* connection,
                                      ui64 transferId, int packetId, int ackTos, ui8 netlibaColor, TOptionsVector* optVector) {
        Y_ASSERT(IsDataCmd(cmd));

        const bool ackTosIsCustom = ackTos != TOS_DEFAULT;
        const bool netlibaColorIsCustom = netlibaColor != DEFAULT_NETLIBA_COLOR;

        if (ackTosIsCustom) {
            optVector->PacketOpt.SetAckTos(ackTos);
        }
        if (netlibaColorIsCustom) {
            optVector->PacketOpt.SetColor(netlibaColor);
        }
        WriteTransferPacketHeader(buf, cmd, connection, transferId, optVector);
        Write(buf, packetId);
    }

    inline bool ReadTransferHeader(const char** pktData, const char* pktEnd, ui64* transferId, const TOptionsVector& opt) {
        *transferId = Read<ui32>(pktData);
        *transferId |= (opt.PacketOpt.GetHigherTransferId() << 32);
        return *pktData <= pktEnd;
    }

    inline bool ReadPacketHeaderTail(const char** pktData, const char* pktEnd, int* packetId, int* ackTos, ui8* netlibaColor, const TOptionsVector& opt) {
        *packetId = Read<int>(pktData);
        *ackTos = opt.PacketOpt.GetAckTos().GetOrElse((int)TOS_DEFAULT);
        *netlibaColor = opt.PacketOpt.GetColor().GetOrElse((ui8)DEFAULT_NETLIBA_COLOR);
        return *pktData <= pktEnd;
    }

    ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////

    template <size_t TbufSize>
    inline void CheckedSendTo(TUdpSocket& s, const char (&buf)[TbufSize], const char* bufEnd, const sockaddr_in6& dst, const sockaddr_in6& src,
                              const ui8 tos, const EFragFlag frag) {
        const size_t len = bufEnd - buf;
        Y_ABORT_UNLESS(len <= TbufSize, "Increase buf size to at least %d bytes", (int)len);
        s.SendTo(buf, len, {dst, src}, tos, frag);
    }

    ///////////////////////////////////////////////////////////////////////////////

    inline void WriteAcksHeader(char** packetBuffer, EUdpCmd cmd, const TConnection* connection, ui64 transferId) {
        Y_ASSERT(IsAckCmd(cmd));
        TOptionsVector opt;
        WriteTransferPacketHeader(packetBuffer, cmd, connection, transferId, &opt);
    }

    inline bool ReadAckPacketHeader(const EUdpCmd cmd, const char** pktData, const char* pktEnd) {
        Y_ASSERT(IsAckCmd(cmd));
        return *pktData <= pktEnd;
    }

    ///////////////////////////////////////////////////////////////////////////////

    inline void SendAckCompleteOrCanceled(const EUdpCmd cmd, TUdpSocket& s, const TConnection* connection,
                                          ui64 transferId, int packetId, ui8 tos) {
        Y_ASSERT(cmd == ACK_COMPLETE || cmd == ACK_CANCELED);
        char buf[PACKET_HEADERS_SIZE], *pktData = buf + UDP_LOW_LEVEL_HEADER_SIZE;
        WriteAcksHeader(&pktData, cmd, connection, transferId);
        Write(&pktData, packetId); // we need packetId to update RTT
        CheckedSendTo(s, buf, pktData, connection->GetWinsockAddress(), connection->GetWinsockMyAddress(),
                      tos, FF_ALLOW_FRAG);
    }

    inline void SendAckComplete(TUdpSocket& s, const TConnection* connection, ui64 transferId, int packetId, ui8 tos) {
        return SendAckCompleteOrCanceled(ACK_COMPLETE, s, connection, transferId, packetId, tos);
    }

    inline void SendAckCanceled(TUdpSocket& s, const TConnection* connection, ui64 transferId, int packetId, ui8 tos) {
        return SendAckCompleteOrCanceled(ACK_CANCELED, s, connection, transferId, packetId, tos);
    }

    inline bool ReadAckCompleteAndCanceledTails(const char* pktData, const char* pktEnd, int* lastPacketId) {
        *lastPacketId = Read<int>(&pktData);
        return pktData == pktEnd;
    }

    ///////////////////////////////////////////////////////////////////////////////

    inline void SendAckResendNoShmem(TUdpSocket& s, TConnection* connection, ui64 transferId, ui8 tos) {
        char buf[PACKET_HEADERS_SIZE], *pktData = buf + UDP_LOW_LEVEL_HEADER_SIZE;
        WriteAcksHeader(&pktData, ACK_RESEND_NOSHMEM, connection, transferId);

        CheckedSendTo(s, buf, pktData, connection->GetWinsockAddress(), connection->GetWinsockMyAddress(),
                      tos, FF_ALLOW_FRAG);
    }

    ///////////////////////////////////////////////////////////////////////////////

    inline void SendCancelTransfer(TUdpSocket& s, const TConnection* connection, ui64 transferId, ui8 tos) {
        char buf[PACKET_HEADERS_SIZE], *pktData = buf + UDP_LOW_LEVEL_HEADER_SIZE;
        TOptionsVector opt;
        WriteTransferPacketHeader(&pktData, CANCEL_TRANSFER, connection, transferId, &opt);
        CheckedSendTo(s, buf, pktData, connection->GetWinsockAddress(), connection->GetWinsockMyAddress(),
                      tos, FF_ALLOW_FRAG);
    }

    ///////////////////////////////////////////////////////////////////////////////

    inline void SendXsPing(TUdpSocket& s, const TConnection* connection, const int selfNetworkOrderPort, ui8 tos) {
        char buf[UDP_XSMALL_PACKET_SIZE], *pktData = buf + UDP_LOW_LEVEL_HEADER_SIZE;
        // Cerr << GetAddressAsString(connection->GetAddress()) << " Sending xs ping" << Endl;

        TOptionsVector opt;
        WriteInConnectionPacketHeader(&pktData, XS_PING, connection, &opt);
        Write(&pktData, selfNetworkOrderPort);

        CheckedSendTo(s, buf, buf + Y_ARRAY_SIZE(buf), connection->GetWinsockAddress(), connection->GetWinsockMyAddress(),
                      tos, FF_DONT_FRAG);
    }

    inline void SendJumboPing(TUdpSocket& s, const TConnection* connection, const int selfNetworkOrderPort, ui8 tos) {
        //HACK: 50 is delta to decrease MTU
        char buf[UDP_MAX_PACKET_SIZE - 50], *pktData = buf + UDP_LOW_LEVEL_HEADER_SIZE;
        // Cerr << GetAddressAsString(connection->GetAddress()) << " Sending jumbo ping" << Endl;

        TOptionsVector opt;
        WriteInConnectionPacketHeader(&pktData, PING, connection, &opt);
        Write(&pktData, selfNetworkOrderPort);

        CheckedSendTo(s, buf, buf + Y_ARRAY_SIZE(buf), connection->GetWinsockAddress(), connection->GetWinsockMyAddress(),
                      tos, FF_DONT_FRAG);
    }

    // not MTU discovery, just figure out IB address of the peer
    inline void SendIBOnlyPing(TUdpSocket& s, const TConnection* connection, const int selfNetworkOrderPort) {
        char buf[PACKET_HEADERS_SIZE], *pktData = buf + UDP_LOW_LEVEL_HEADER_SIZE;

        TOptionsVector opt;
        WriteInConnectionPacketHeader(&pktData, PING, connection, &opt);
        Write(&pktData, selfNetworkOrderPort);

        const ui8 tos = 0;
        CheckedSendTo(s, buf, pktData, connection->GetWinsockAddress(), connection->GetWinsockMyAddress(),
                      tos, FF_ALLOW_FRAG);
    }

    template <class TTPort>
    inline bool ReadPing(const char* pktData, const char* pktEnd, TTPort* port) {
        *port = Read<TTPort>(&pktData);
        return pktData <= pktEnd; // ping packets may be jumbo-packets
    }

    ///////////////////////////////////////////////////////////////////////////////

    inline void SendPong(TUdpSocket& s, const TConnection* connection, const sockaddr_in6& toAddress, bool isXs) {
        char buf[PACKET_HEADERS_SIZE], *pktData = buf + UDP_LOW_LEVEL_HEADER_SIZE;
        TOptionsVector opt;
        WriteInConnectionPacketHeader(&pktData, isXs ? XS_PONG : PONG, connection, &opt);

        const ui8 tos = 0; // TODO: tos?
        CheckedSendTo(s, buf, pktData, toAddress, connection->GetWinsockMyAddress(),
                      tos, FF_ALLOW_FRAG);
    }

    inline bool ReadPong(const char* pktData, const char* pktEnd) {
        return pktData == pktEnd;
    }

    ///////////////////////////////////////////////////////////////////////////////

    inline void SendIBPong(TUdpSocket& s, const TConnection* connection, const TIBConnectInfo& info, const sockaddr_in6& toAddress) {
        char buf[PACKET_HEADERS_SIZE], *pktData = buf + UDP_LOW_LEVEL_HEADER_SIZE;

        TOptionsVector opt;
        WriteInConnectionPacketHeader(&pktData, PONG_IB, connection, &opt);
        Write(&pktData, info);
        Write(&pktData, toAddress);

        const ui8 tos = 0; // TODO: tos?
        CheckedSendTo(s, buf, pktData, toAddress, connection->GetWinsockMyAddress(),
                      tos, FF_ALLOW_FRAG);
    }

    inline bool ReadIBPong(const char* pktData, const char* pktEnd, TIBConnectInfo* info, sockaddr_in6* myAddress) {
        *info = Read<TIBConnectInfo>(&pktData);
        *myAddress = Read<sockaddr_in6>(&pktData);
        return pktData == pktEnd;
    }

    ///////////////////////////////////////////////////////////////////////////////

    extern const ui64 KILL_PASSPHRASE1;
    extern const ui64 KILL_PASSPHRASE2;

    inline bool ReadKill(const char* pktData, const char* pktEnd) {
        const ui64 p1 = Read<ui64>(&pktData);
        const ui64 p2 = Read<ui64>(&pktData);
        return pktData == pktEnd && p1 == KILL_PASSPHRASE1 && p2 == KILL_PASSPHRASE2;
    }

    ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////

    // grouped acks, first int - packet_id, second int - bit mask for 32 packets preceding packet_id
    extern const size_t SIZEOF_ACK = 8;
    inline size_t WriteAck(TUdpInTransfer* xfer, int* dst, const size_t maxAcks) {
        if (xfer->NewPacketsToAck.empty())
            return 0;
        if (xfer->NewPacketsToAck.size() > 1)
            Sort(xfer->NewPacketsToAck.begin(), xfer->NewPacketsToAck.end());

        size_t ackCount = 0;
        int firstNotAckedPkt = xfer->NewPacketsToAck.back();
        for (int idx = xfer->NewPacketsToAck.ysize() - 1; idx >= -1; --idx) {
            // mask all packets in index interval (idx, firstNotAckedIdx]
            int pkt = -1;
            if (idx == -1 || (pkt = xfer->NewPacketsToAck[idx]) < firstNotAckedPkt - 32) {
                *dst++ = firstNotAckedPkt;

                // ack not only packets from NewPacketsToAck but also all other received
                // that could fit in bitmask. Acks are never enough!
                int bitMask = 0;
                int backPackets = Min(firstNotAckedPkt, 32);
                for (int k = 0; k < backPackets; ++k) {
                    if (xfer->GetPacket(firstNotAckedPkt - 1 - k)) {
                        bitMask |= 1U << k;
                    }
                }
                *dst++ = bitMask;
                if (++ackCount >= maxAcks || idx == -1)
                    break;
                firstNotAckedPkt = pkt;
                //printf("sending ack %d (mask %x)\n", pkt, bitMask);
            }
        }
        xfer->NewPacketsToAck.clear();
        return ackCount;
    }

    inline bool AckPacket(const int pkt, const float deltaT, const bool updateRTT, TAckTracker* ackTracker) {
        if (0 <= pkt && pkt < ackTracker->GetPacketCount()) {
            ackTracker->Ack(pkt, deltaT, updateRTT);
            return true;
        }
        Y_ASSERT(false);
        return false;
    }

    inline bool ReadAcks(const int* acks, const size_t ackCount, const float deltaT, TAckTracker* ackTracker) {
        for (size_t i = 0; i < ackCount; ++i) {
            const int pkt = *acks++;
            const int bitMask = *acks++;
            const bool updateRTT = i == ackCount - 1; // update RTT using only last packet in the pack
            AckPacket(pkt, deltaT, updateRTT, ackTracker);
            for (int k = 0; k < 32; ++k) {
                if (bitMask & (1 << k)) {
                    if (!AckPacket(pkt - k - 1, deltaT, false, ackTracker)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////////

    // by passing xfer we reduce number of hash map finds.
    inline void AddAcksToPacketQueue(TUdpSocket& s, char* packetBuffer, const size_t packetBufferSize,
                                     TConnection* connection, const ui64 transferId, TUdpInTransfer* xfer) {
        char* pktData = packetBuffer;
        WriteAcksHeader(&pktData, ACK, connection, transferId);

        const size_t acks = WriteAck(xfer, (int*)pktData, (size_t)(packetBufferSize - (pktData - packetBuffer)) / SIZEOF_ACK);
        pktData += acks * SIZEOF_ACK;

        const size_t ackPacketSize = connection->GetSmallMtuUseXs() ? UDP_XSMALL_PACKET_SIZE : UDP_SMALL_PACKET_SIZE;

        s.AddPacketToQueue(pktData - packetBuffer, {connection->GetWinsockAddress(), connection->GetWinsockMyAddress()},
                           xfer->AckTos, ackPacketSize);
    }

    inline bool ReadAndSetAcks(const char* pktData, const char* pktEnd, const float deltaT, TAckTracker* ackTracker) {
        return ReadAcks((const int*)pktData, (pktEnd - pktData) / SIZEOF_ACK, deltaT, ackTracker);
    }

    ///////////////////////////////////////////////////////////////////////////////

    inline void AddDataToPacketQueue(TUdpSocket& s, char* packetBuffer,
                                     TConnection* connection, const ui64 transferId, const TUdpOutTransfer& xfer,
                                     const int packetId, const int dataSize) {
        Y_ASSERT(xfer.PacketSize == UDP_PACKET_SIZE || xfer.PacketSize == UDP_SMALL_PACKET_SIZE || xfer.PacketSize == UDP_XSMALL_PACKET_SIZE);
        Y_ASSERT(xfer.LastPacketSize < xfer.PacketSize);
        Y_ASSERT(dataSize == xfer.PacketSize || dataSize == xfer.LastPacketSize);

        EUdpCmd cmd = xfer.PacketSize == UDP_PACKET_SIZE ? DATA : DATA_SMALL;
        TPosixSharedMemory* shm = xfer.Data->GetSharedData();

        char* pktData = packetBuffer;
        TOptionsVector TransferOptions;
        if (packetId == 0) {
            if (xfer.PacketPriority == PP_HIGH || xfer.PacketPriority == PP_SYSTEM) {
                TransferOptions.TransferOpt.SetHighPriority(true);
            }
            if (shm) {
                //fprintf(stderr, "TransferOptions.TransferOpt.SetSharedMemory sz: %i\n", (int)shm->GetSize());
                //We assume size of one shared memory region can`t be more than size of one transfer which can`t be more than 1.8G
                Y_ASSERT(shm->GetSizeT() <= std::numeric_limits<ui32>::max());
                TransferOptions.TransferOpt.SetSharedMemory((ui32)shm->GetSizeT(), shm->GetId());
            }
        }
        /*
        Cerr << GetAddressAsString(connection->GetAddress()) << " Sending packet " << packetId << " of " << xfer.PacketCount
            << " with xfer.PacketSize=" << xfer.PacketSize
            << ", *conn=" << size_t(connection)
            << ", conn.SmallMtuUseXs=" << connection->GetSmallMtuUseXs()
            << ", dataSize=" << dataSize
            << ", cmde=" << int(cmd)
            << Endl;
        */
        WriteDataPacketHeader(&pktData, cmd, connection, transferId, packetId, xfer.AckTos, xfer.NetlibaColor, &TransferOptions);

        // TODO: we can avoid memcpy here, but xfer.Data is a list of data chunks and we need whole packet...
        // we may use iovec!
        TBlockChainIterator dataReader(xfer.Data->GetChain());
        dataReader.Seek((int)(packetId * xfer.PacketSize));
        dataReader.Read(pktData, (int)dataSize);
        pktData += dataSize;

        // AddPacketToQueue YASSERTs buffer overflow
        s.AddPacketToQueue(pktData - packetBuffer, {connection->GetWinsockAddress(), connection->GetWinsockMyAddress()},
                           xfer.DataTos, xfer.PacketSize);
    }

    inline bool ReadDataPacket(const EUdpCmd cmd, const char** pktData, const char* pktEnd, const int packetId,
                               TIntrusivePtr<TPosixSharedMemory>* shm, size_t* packetSize, const TOptionsVector& opt) {
        Y_ASSERT(IsDataCmd(cmd));
        if (packetId != 0) {
            if (!opt.TransferOpt.Empty()) {
                fprintf(stderr, "TransferOptions can be used only with zero packetId, but has flags: %i \n", (int)opt.TransferOpt.GetFlags());
                return false;
            }
        }
        if (packetId == 0) {
            // link to attached shared memory
            if (opt.TransferOpt.IsSharedMemoryRequired()) {
                const TGUID shmId = opt.TransferOpt.GetSharedMemoryGuid();
                const size_t shmSize = opt.TransferOpt.GetSharedMemorySize();

                if (*pktData > pktEnd) {
                    return false;
                }

                if (!shm->Get()) {
                    *shm = new TPosixSharedMemory;
                    if (!(*shm)->Open(shmId, shmSize)) {
                        fprintf(stderr, "shm->Open failed! shmId = %s, shmSize = %d, opt flags: %d\n", GetGuidAsString(shmId).c_str(), (int)shmSize, (int)opt.TransferOpt.GetFlags());
                        abort();
                    }
                }
            }
        }

        const size_t expectedPacketSize = (cmd == DATA) ? UDP_PACKET_SIZE : (opt.PacketOpt.IsSmallMtuUseXs() ? UDP_XSMALL_PACKET_SIZE : UDP_SMALL_PACKET_SIZE);
        if (!(*packetSize)) {
            *packetSize = expectedPacketSize;
        }
        return *packetSize == expectedPacketSize;
    }
}
