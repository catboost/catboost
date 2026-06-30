#pragma once

#include "ib_low.h"

namespace NNetliba_v12 {
    // buffer id 0 is special, it is used when data is sent inline and should not be returned
    const size_t SMALL_PKT_SIZE = 4096;

    const ui64 BP_AH_USED_FLAG = 0x1000000000000000ul;
    const ui64 BP_BUF_ID_MASK = 0x00000000fffffffful;

    // single thread version
    class TIBBufferPool: public TThrRefBase, TNonCopyable {
        static constexpr int BLOCK_SIZE_LN = 11;
        static constexpr int BLOCK_SIZE = 1 << BLOCK_SIZE_LN;
        static constexpr int BLOCK_COUNT = 1024;

        struct TSingleBlock {
            TIntrusivePtr<TMemoryRegion> Mem;
            TVector<ui8> BlkRefCounts;
            TVector<TIntrusivePtr<TAddressHandle>> AHHolder;

            void Alloc(TPtrArg<TIBContext> ctx) {
                size_t dataSize = SMALL_PKT_SIZE * BLOCK_SIZE;
                Mem = new TMemoryRegion(ctx, dataSize);
                BlkRefCounts.resize(BLOCK_SIZE, 0);
                AHHolder.resize(BLOCK_SIZE);
            }
            char* GetBufData(ui64 idArg) {
                char* data = Mem->GetData();
                return data + (idArg & (BLOCK_SIZE - 1)) * SMALL_PKT_SIZE;
            }
        };

        TIntrusivePtr<TIBContext> IBCtx;
        TVector<int> FreeList;
        TVector<TSingleBlock> Blocks;
        size_t FirstFreeBlock;
        int PostRecvDeficit;
        TIntrusivePtr<TSharedReceiveQueue> SRQ;

        void AddBlock() {
            if (FirstFreeBlock == Blocks.size()) {
                Y_ABORT_UNLESS(0, "run out of buffers");
            }
            Blocks[FirstFreeBlock].Alloc(IBCtx);
            size_t start = (FirstFreeBlock == 0) ? 1 : FirstFreeBlock * BLOCK_SIZE;
            size_t finish = FirstFreeBlock * BLOCK_SIZE + BLOCK_SIZE;
            for (size_t i = start; i < finish; ++i) {
                FreeList.push_back(i);
            }
            ++FirstFreeBlock;
        }

    public:
        TIBBufferPool(TPtrArg<TIBContext> ctx, int maxSRQWorkRequests)
            : IBCtx(ctx)
            , FirstFreeBlock(0)
            , PostRecvDeficit(maxSRQWorkRequests)
        {
            Blocks.resize(BLOCK_COUNT);
            AddBlock();
            SRQ = new TSharedReceiveQueue(ctx, maxSRQWorkRequests);

            PostRecv();
        }
        TSharedReceiveQueue* GetSRQ() const {
            return SRQ.Get();
        }
        int AllocBuf() {
            if (FreeList.empty()) {
                AddBlock();
            }
            int id = FreeList.back();
            FreeList.pop_back();
            Y_ASSERT(++Blocks[id >> BLOCK_SIZE_LN].BlkRefCounts[id & (BLOCK_SIZE - 1)] == 1);
            return id;
        }
        void FreeBuf(ui64 idArg) {
            ui64 id = idArg & BP_BUF_ID_MASK;
            if (id == 0) {
                return;
            }
            Y_ASSERT(id > 0 && id < (ui64)(FirstFreeBlock * BLOCK_SIZE));
            FreeList.push_back(id);
            Y_ASSERT(--Blocks[id >> BLOCK_SIZE_LN].BlkRefCounts[id & (BLOCK_SIZE - 1)] == 0);
            if (idArg & BP_AH_USED_FLAG) {
                Blocks[id >> BLOCK_SIZE_LN].AHHolder[id & (BLOCK_SIZE - 1)] = nullptr;
            }
        }
        char* GetBufData(ui64 idArg) {
            ui64 id = idArg & BP_BUF_ID_MASK;
            return Blocks[id >> BLOCK_SIZE_LN].GetBufData(id);
        }
        int PostSend(TPtrArg<TRCQueuePair> qp, const void* data, size_t len) {
            if (len > SMALL_PKT_SIZE) {
                Y_ABORT_UNLESS(0, "buffer overrun");
            }
            if (len <= MAX_INLINE_DATA_SIZE) {
                qp->PostSend(nullptr, 0, data, len);
                return 0;
            } else {
                int id = AllocBuf();
                TSingleBlock& blk = Blocks[id >> BLOCK_SIZE_LN];
                char* buf = blk.GetBufData(id);
                memcpy(buf, data, len);
                qp->PostSend(blk.Mem, id, buf, len);
                return id;
            }
        }
        void PostSend(TPtrArg<TUDQueuePair> qp, TPtrArg<TAddressHandle> ah, int remoteQPN, int remoteQKey,
                      const void* data, size_t len) {
            if (len > SMALL_PKT_SIZE - 40) {
                Y_ABORT_UNLESS(0, "buffer overrun");
            }
            ui64 id = AllocBuf();
            TSingleBlock& blk = Blocks[id >> BLOCK_SIZE_LN];
            int ptr = id & (BLOCK_SIZE - 1);
            blk.AHHolder[ptr] = ah.Get();
            id |= BP_AH_USED_FLAG;
            if (len <= MAX_INLINE_DATA_SIZE) {
                qp->PostSend(ah, remoteQPN, remoteQKey, nullptr, id, data, len);
            } else {
                char* buf = blk.GetBufData(id);
                memcpy(buf, data, len);
                qp->PostSend(ah, remoteQPN, remoteQKey, blk.Mem, id, buf, len);
            }
        }
        void RequestPostRecv() {
            ++PostRecvDeficit;
        }
        void PostRecv() {
            for (int i = 0; i < PostRecvDeficit; ++i) {
                int id = AllocBuf();
                TSingleBlock& blk = Blocks[id >> BLOCK_SIZE_LN];
                char* buf = blk.GetBufData(id);
                SRQ->PostReceive(blk.Mem, id, buf, SMALL_PKT_SIZE);
            }
            PostRecvDeficit = 0;
        }
    };

    class TIBRecvPacketProcess: public TNonCopyable {
        TIBBufferPool& BP;
        ui64 Id;
        char* Data;

    public:
        TIBRecvPacketProcess(TIBBufferPool& bp, const ibv_wc& wc)
            : BP(bp)
            , Id(wc.wr_id)
        {
            Y_ASSERT(wc.opcode & IBV_WC_RECV);
            BP.RequestPostRecv();
            Data = BP.GetBufData(Id);
        }
        // intended for postponed packet processing
        // with this call RequestPostRecv() should be called outside (and PostRecv() too in order to avoid rnr situation)
        TIBRecvPacketProcess(TIBBufferPool& bp, const ui64 wr_id)
            : BP(bp)
            , Id(wr_id)
        {
            Data = BP.GetBufData(Id);
        }
        ~TIBRecvPacketProcess() {
            BP.FreeBuf(Id);
            BP.PostRecv();
        }
        char* GetData() const {
            return Data;
        }
        char* GetUDData() const {
            return Data + 40;
        }
        ibv_grh* GetGRH() const {
            return (ibv_grh*)Data;
        }
    };

}
