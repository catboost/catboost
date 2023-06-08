#pragma once

#include <library/cpp/binsaver/bin_saver.h>

namespace NNetliba {
    struct TCollectiveInit {
        int Size, Rank;

        SAVELOAD(Size, Rank);
    };

    struct TCollectiveLinkSet {
        struct TLinkInfo {
            int QPN, PSN;
        };
        TVector<int> Hosts;              // host LIDs
        TVector<TVector<int>> HostGroup; // HostGroup[0] - switchId, HostGroup[1] - hostId within the switch
        TVector<TLinkInfo> Links;

        SAVELOAD(Hosts, HostGroup, Links);
    };

    struct IAllDataSync: public TThrRefBase {
        virtual void* GetRawData() = 0;
        virtual size_t GetRawDataSize() = 0;
        virtual void Sync() = 0;
        virtual void Flush() = 0;

        template <class T>
        T* GetData() {
            return static_cast<T*>(GetRawData());
        }
        template <class T>
        size_t GetSize() {
            return GetRawDataSize() / sizeof(T);
        }
    };

    struct IAllReduce: public IAllDataSync {
        virtual bool Resize(size_t dataSize) = 0;
    };

    struct IAllGather: public IAllDataSync {
        virtual bool Resize(const TVector<size_t>& szPerRank) = 0;
    };

    struct IReduceOp: public TThrRefBase {
        virtual void Reduce(void* dst, const void* add, size_t dataSize) const = 0;
    };

    template <class T, class TElem = typename T::TElem>
    class TAllReduceOp: public IReduceOp {
        T Op;

    public:
        TAllReduceOp() {
        }
        TAllReduceOp(T op)
            : Op(op)
        {
        }
        void Reduce(void* dst, const void* add, size_t dataSize) const override {
            TElem* dstPtr = (TElem*)(dst);
            const TElem* addPtr = (const TElem*)(add);
            TElem* finPtr = (TElem*)(((char*)dst) + dataSize);
            while (dstPtr < finPtr) {
                Op(dstPtr, *addPtr);
                ++dstPtr;
                ++addPtr;
            }
        }
    };

    // table of active peers for micro send/recv
    class TIBMicroPeerTable {
        TVector<ui8> Table; // == 0 means accept mesages from this qpn
        int TableSize;
        bool ParsePending;

    public:
        TIBMicroPeerTable()
            : ParsePending(true)
        {
            Init(0);
        }
        void Init(int tableSizeLog) {
            TableSize = 1 << tableSizeLog;
            ParsePending = true;
            Table.resize(0);
            Table.resize(TableSize, 0);
        }
        bool NeedParsePending() const {
            return ParsePending;
        }
        void StopParsePending() {
            ParsePending = false;
        }
        void StopQPN(int qpn, ui8 mask) {
            Y_ASSERT((Table[qpn & (TableSize - 1)] & mask) == 0);
            Table[qpn & (TableSize - 1)] |= mask;
        }
        void StopQPN(int qpn) {
            Y_ASSERT(Table[qpn & (TableSize - 1)] == 0);
            Table[qpn & (TableSize - 1)] = 0xff;
        }
        bool NeedQPN(int qpn) const {
            return Table[qpn & (TableSize - 1)] != 0xff;
        }
    };

    struct IIBCollective;
    class TIBCollective;
    class TIBRecvMicro: public TNonCopyable {
        TIBCollective& IB;
        ui64 Id;
        int QPN;
        void* Data;

    public:
        TIBRecvMicro(IIBCollective* col, TIBMicroPeerTable* peerTable);
        ~TIBRecvMicro();
        void* GetRawData() const {
            return Data;
        }
        template <class T>
        T* GetData() {
            return static_cast<T*>(GetRawData());
        }
        int GetQPN() const {
            return QPN;
        }
    };

    struct IIBCollective: public TThrRefBase {
        struct TRdmaRequest {
            int DstRank;
            ui64 RemoteAddr, LocalAddr;
            ui32 RemoteKey, LocalKey;
            ui64 Size;
        };

        virtual int GetRank() = 0;
        virtual int GetSize() = 0;
        virtual int GetGroupTypeCount() = 0;
        virtual int GetQPN(int rank) = 0;
        virtual bool TryWaitCompletion() = 0;
        virtual void WaitCompletion() = 0;
        virtual void Start(const TCollectiveLinkSet& links) = 0;
        virtual IAllGather* CreateAllGather(const TVector<size_t>& szPerRank) = 0;
        virtual IAllGather* CreateAllGather(size_t szPerRank) = 0;
        virtual IAllReduce* CreateAllReduce(size_t dataSize, TPtrArg<IReduceOp> reduceOp) = 0;
        virtual void RunBWTest(int groupType, int delta, int* targetRank, float* res) = 0;
        virtual void Fence() = 0;
        virtual void InitPeerTable(TIBMicroPeerTable* res) = 0;
        virtual bool TrySendMicro(int dstRank, const void* data, int dataSize) = 0;
        virtual void RdmaWrite(const TVector<TRdmaRequest>& reqs) = 0;
    };

    IIBCollective* CreateCollective(const TCollectiveInit& params, TCollectiveLinkSet* resLinks);
}
