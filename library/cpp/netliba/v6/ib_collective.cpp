#include "stdafx.h"
#include "ib_collective.h"
#include "ib_mem.h"
#include "ib_buffers.h"
#include "ib_low.h"
#include "udp_http.h"
#include "udp_address.h"
#include <util/generic/deque.h>
#include <util/system/hp_timer.h>

namespace NNetliba {
    const int COL_SERVICE_LEVEL = 2;
    const int COL_DATA_SERVICE_LEVEL = 2;       // base level
    const int COL_DATA_SERVICE_LEVEL_COUNT = 6; // level count
    const int MAX_REQS_PER_PEER = 32;
    const int MAX_TOTAL_RDMA = 20;
    const int SEND_COUNT_TABLE_SIZE = 1 << 12; // must be power of 2

    struct TMergeRecord {
        struct TTransfer {
            int DstRank;
            int SL;
            int RangeBeg, RangeFin;
            int Id;

            TTransfer()
                : DstRank(-1)
                , SL(0)
                , RangeBeg(0)
                , RangeFin(0)
                , Id(0)
            {
            }
            TTransfer(int dstRank, int sl, int rangeBeg, int rangeFin, int id)
                : DstRank(dstRank)
                , SL(sl)
                , RangeBeg(rangeBeg)
                , RangeFin(rangeFin)
                , Id(id)
            {
            }
        };
        struct TInTransfer {
            int SrcRank;
            int SL;

            TInTransfer()
                : SrcRank(-1)
                , SL(0)
            {
            }
            TInTransfer(int srcRank, int sl)
                : SrcRank(srcRank)
                , SL(sl)
            {
            }
        };

        TVector<TTransfer> OutList;
        TVector<TInTransfer> InList;
        ui64 RecvMask;

        TMergeRecord()
            : RecvMask(0)
        {
        }
    };

    struct TMergeIteration {
        TVector<TMergeRecord> Ops;

        void Init(int colSize) {
            Ops.resize(colSize);
        }
        void Transfer(int srcRank, int dstRank, int sl, int rangeBeg, int rangeFin, int id) {
            Y_ABORT_UNLESS(id < 64, "recv mask overflow");
            Ops[srcRank].OutList.push_back(TMergeRecord::TTransfer(dstRank, sl, rangeBeg, rangeFin, id));
            Ops[dstRank].InList.push_back(TMergeRecord::TInTransfer(srcRank, sl));
            Ops[dstRank].RecvMask |= ui64(1) << id;
        }
    };

    struct TMergePlan {
        TVector<TMergeIteration> Iterations;
        TVector<int> RankReceiveCount;
        int ColSize;
        int MaxRankReceiveCount;

        TMergePlan()
            : ColSize(0)
            , MaxRankReceiveCount(0)
        {
        }
        void Init(int colSize) {
            Iterations.resize(0);
            RankReceiveCount.resize(0);
            RankReceiveCount.resize(colSize, 0);
            ColSize = colSize;
        }
        void Transfer(int iter, int srcRank, int dstRank, int sl, int rangeBeg, int rangeFin) {
            while (iter >= Iterations.ysize()) {
                TMergeIteration& res = Iterations.emplace_back();
                res.Init(ColSize);
            }
            int id = RankReceiveCount[dstRank]++;
            MaxRankReceiveCount = Max(MaxRankReceiveCount, id + 1);
            Y_ASSERT(id < 64);
            Iterations[iter].Transfer(srcRank, dstRank, sl, rangeBeg, rangeFin, id);
        }
    };

    struct TSRTransfer {
        int SrcRank, DstRank;
        int RangeBeg, RangeFin;

        TSRTransfer() {
            Zero(*this);
        }
        TSRTransfer(int srcRank, int dstRank, int rangeBeg, int rangeFin)
            : SrcRank(srcRank)
            , DstRank(dstRank)
            , RangeBeg(rangeBeg)
            , RangeFin(rangeFin)
        {
        }
    };

    static int SplitRange(THashMap<int, TVector<TSRTransfer>>* res, int iter, int beg, int fin) {
        int mid = (beg + fin + 1) / 2;
        if (mid == fin) {
            return iter;
        }
        for (int i = 0; i < fin - mid; ++i) {
            (*res)[iter].push_back(TSRTransfer(beg + i, mid + i, beg, mid));
            (*res)[iter].push_back(TSRTransfer(mid + i, beg + i, mid, fin));
        }
        if (fin - mid < mid - beg) {
            // [mid - 1] did not receive [mid;fin)
            (*res)[iter].push_back(TSRTransfer(mid, mid - 1, mid, fin));
        }
        int rv1 = SplitRange(res, iter + 1, beg, mid);
        int rv2 = SplitRange(res, iter + 1, mid, fin);
        return Max(rv1, rv2);
    }

    static void CreatePow2Merge(TMergePlan* plan, int colSize) {
        // finally everybody has full range [0;ColSize)
        // construct plan recursively, on each iteration split some range
        plan->Init(colSize);

        THashMap<int, TVector<TSRTransfer>> allTransfers;
        int maxIter = SplitRange(&allTransfers, 0, 0, colSize);

        for (int iter = 0; iter < maxIter; ++iter) {
            const TVector<TSRTransfer>& arr = allTransfers[maxIter - iter - 1]; // reverse order
            for (int i = 0; i < arr.ysize(); ++i) {
                const TSRTransfer& sr = arr[i];
                plan->Transfer(iter, sr.SrcRank, sr.DstRank, 0, sr.RangeBeg, sr.RangeFin);
            }
        }
    }

    struct TCoverInterval {
        int Beg, Fin; // [Beg;Fin)

        TCoverInterval()
            : Beg(0)
            , Fin(0)
        {
        }
        TCoverInterval(int b, int f)
            : Beg(b)
            , Fin(f)
        {
        }
    };

    enum EAllToAllMode {
        AA_POW2,
        AA_CIRCLE,
        AA_STAR,
        AA_POW2_MERGE,
    };
    static int AllToAll(TMergePlan* plan, int iter, int sl, EAllToAllMode mode, const TVector<int>& myGroup, TVector<TCoverInterval>* cover) {
        TVector<TCoverInterval>& hostCoverage = *cover;
        int groupSize = myGroup.ysize();

        for (int k = 1; k < groupSize; ++k) {
            int h1 = myGroup[k - 1];
            int h2 = myGroup[k];
            Y_ABORT_UNLESS(hostCoverage[h1].Fin == hostCoverage[h2].Beg, "Invalid host order in CreateGroupMerge()");
        }

        switch (mode) {
            case AA_POW2: {
                for (int delta = 1; delta < groupSize; delta *= 2) {
                    int sz = Min(delta, groupSize - delta);
                    for (int offset = 0; offset < groupSize; ++offset) {
                        int srcRank = myGroup[offset];
                        int dstRank = myGroup[(offset + delta) % groupSize];

                        int start = offset + 1 - sz;
                        int finish = offset + 1;
                        if (start < 0) {
                            // [start; myGroup.size())
                            int dataBeg = hostCoverage[myGroup[start + groupSize]].Beg;
                            int dataFin = hostCoverage[myGroup.back()].Fin;
                            plan->Transfer(iter, srcRank, dstRank, sl, dataBeg, dataFin);
                            // [0; finish)
                            dataBeg = hostCoverage[myGroup[0]].Beg;
                            dataFin = hostCoverage[myGroup[finish - 1]].Fin;
                            plan->Transfer(iter, srcRank, dstRank, sl, dataBeg, dataFin);
                        } else {
                            // [start;finish)
                            int dataBeg = hostCoverage[myGroup[start]].Beg;
                            int dataFin = hostCoverage[myGroup[finish - 1]].Fin;
                            plan->Transfer(iter, srcRank, dstRank, sl, dataBeg, dataFin);
                        }
                    }
                    ++iter;
                }
            } break;
            case AA_CIRCLE: {
                for (int dataDelta = 1; dataDelta < groupSize; ++dataDelta) {
                    for (int offset = 0; offset < groupSize; ++offset) {
                        int srcRank = myGroup[offset];
                        int dstRank = myGroup[(offset + 1) % groupSize];

                        int dataRank = myGroup[(offset + 1 - dataDelta + groupSize) % groupSize];
                        int dataBeg = hostCoverage[dataRank].Beg;
                        int dataFin = hostCoverage[dataRank].Fin;

                        plan->Transfer(iter, srcRank, dstRank, sl, dataBeg, dataFin);
                    }
                    ++iter;
                }
            } break;
            case AA_STAR: {
                for (int offset = 0; offset < groupSize; ++offset) {
                    for (int delta = 1; delta < groupSize; ++delta) {
                        int srcRank = myGroup[offset];
                        int dstRank = myGroup[(offset + delta) % groupSize];

                        int dataRank = myGroup[offset];
                        int dataBeg = hostCoverage[dataRank].Beg;
                        int dataFin = hostCoverage[dataRank].Fin;

                        plan->Transfer(iter, srcRank, dstRank, sl, dataBeg, dataFin);
                    }
                }
                ++iter;
            } break;
            case AA_POW2_MERGE: {
                TMergePlan pp;
                CreatePow2Merge(&pp, groupSize);
                for (int z = 0; z < pp.Iterations.ysize(); ++z) {
                    const TMergeIteration& mm = pp.Iterations[z];
                    for (int src = 0; src < mm.Ops.ysize(); ++src) {
                        const TMergeRecord& mr = mm.Ops[src];
                        int srcRank = myGroup[src];
                        for (int i = 0; i < mr.OutList.ysize(); ++i) {
                            int dstRank = myGroup[mr.OutList[i].DstRank];
                            plan->Transfer(iter, srcRank, dstRank, sl, 0, 1);
                        }
                    }
                    ++iter;
                }
            } break;
            default:
                Y_ASSERT(0);
                break;
        }
        {
            TCoverInterval cc(hostCoverage[myGroup[0]].Beg, hostCoverage[myGroup.back()].Fin);
            for (int k = 0; k < groupSize; ++k) {
                hostCoverage[myGroup[k]] = cc;
            }
        }
        return iter;
    }

    // fully populated matrix
    static void CreateGroupMerge(TMergePlan* plan, EAllToAllMode mode, const TVector<TVector<int>>& hostGroup) {
        int hostCount = hostGroup[0].ysize();
        int groupTypeCount = hostGroup.ysize();

        plan->Init(hostCount);

        TVector<int> gcount;
        gcount.resize(groupTypeCount, 0);
        for (int hostId = 0; hostId < hostCount; ++hostId) {
            for (int groupType = 0; groupType < groupTypeCount; ++groupType) {
                int val = hostGroup[groupType][hostId];
                gcount[groupType] = Max(gcount[groupType], val + 1);
            }
        }
        for (int hostId = 1; hostId < hostCount; ++hostId) {
            bool isIncrement = true;
            for (int groupType = 0; groupType < groupTypeCount; ++groupType) {
                int prev = hostGroup[groupType][hostId - 1];
                int cur = hostGroup[groupType][hostId];
                if (isIncrement) {
                    if (cur == prev + 1) {
                        isIncrement = false;
                    } else {
                        Y_ABORT_UNLESS(cur == 0, "ib_hosts, wrapped to non-zero");
                        Y_ABORT_UNLESS(prev == gcount[groupType] - 1, "ib_hosts, structure is irregular");
                        isIncrement = true;
                    }
                } else {
                    Y_ABORT_UNLESS(prev == cur, "ib_hosts, structure is irregular");
                }
            }
        }

        TVector<TCoverInterval> hostCoverage;
        for (int i = 0; i < hostCount; ++i) {
            hostCoverage.push_back(TCoverInterval(i, i + 1));
        }

        int baseIter = 0;
        for (int groupType = hostGroup.ysize() - 1; groupType >= 0; --groupType) {
            Y_ASSERT(hostGroup[groupType].ysize() == hostCount);
            TVector<TVector<int>> hh;
            hh.resize(gcount[groupType]);
            for (int rank = 0; rank < hostGroup[groupType].ysize(); ++rank) {
                int groupId = hostGroup[groupType][rank];
                hh[groupId].push_back(rank);
            }
            int newIter = 0;
            for (int groupId = 0; groupId < hh.ysize(); ++groupId) {
                int nn = AllToAll(plan, baseIter, 0, mode, hh[groupId], &hostCoverage); // seems to be fastest
                if (newIter == 0) {
                    newIter = nn;
                } else {
                    Y_ABORT_UNLESS(newIter == nn, "groups should be of same size");
                }
            }
            baseIter = newIter;
        }
        //printf("%d iterations symmetrical plan\n", baseIter);
    }

    //////////////////////////////////////////////////////////////////////////
    struct TAllDataSync {
        enum {
            WR_COUNT = 64 * 2
        };

        int CurrentBuffer;
        TIntrusivePtr<TIBMemBlock> MemBlock[2];
        TIntrusivePtr<TComplectionQueue> CQ;
        TIntrusivePtr<TSharedReceiveQueue> SRQ;
        TIntrusivePtr<TIBMemBlock> FakeRecvMem;
        size_t DataSize, BufSize;
        size_t CurrentOffset, ReadyOffset;
        bool WasFlushed;
        int ActiveRDMACount;
        ui64 FutureRecvMask;
        TIntrusivePtr<IReduceOp> ReduceOp;

        struct TBlockInfo {
            ui64 Addr;
            ui32 Key;
        };
        struct TSend {
            TBlockInfo RemoteBlocks[2];
            TIntrusivePtr<TRCQueuePair> QP;
            size_t SrcOffset;
            size_t DstOffset;
            size_t Length;
            ui32 ImmData;
            int DstRank;
            union {
                struct {
                    int RangeBeg, RangeFin;
                } Gather;
                struct {
                    int SrcIndex, DstIndex;
                } Reduce;
            };
        };
        struct TRecv {
            TIntrusivePtr<TRCQueuePair> QP;
            int SrcRank;
        };
        struct TReduce {
            size_t DstOffset, SrcOffset;
            int DstIndex, SrcIndex;
        };
        struct TIteration {
            TVector<TSend> OutList;
            TVector<TRecv> InList;
            TVector<TReduce> ReduceList;
            ui64 RecvMask;
        };
        TVector<TIteration> Iterations;

    public:
        void* GetRawData() {
            char* myData = (char*)MemBlock[CurrentBuffer]->GetData();
            return myData + CurrentOffset;
        }
        size_t GetRawDataSize() {
            return DataSize;
        }
        void PostRecv() {
            SRQ->PostReceive(FakeRecvMem->GetMemRegion(), 0, FakeRecvMem->GetData(), FakeRecvMem->GetSize());
        }
        void Sync() {
            Y_ASSERT(WasFlushed && "Have to call Flush() before data fill & Sync()");
            char* myData = (char*)MemBlock[CurrentBuffer]->GetData();

            ui64 recvMask = FutureRecvMask;
            FutureRecvMask = 0;
            int recvDebt = 0;
            for (int z = 0; z < Iterations.ysize(); ++z) {
                const TIteration& iter = Iterations[z];
                for (int k = 0; k < iter.OutList.ysize(); ++k) {
                    const TSend& ss = iter.OutList[k];
                    const TBlockInfo& remoteBlk = ss.RemoteBlocks[CurrentBuffer];
                    ss.QP->PostRDMAWriteImm(remoteBlk.Addr + ss.DstOffset, remoteBlk.Key, ss.ImmData,
                                            MemBlock[CurrentBuffer]->GetMemRegion(), 0, myData + ss.SrcOffset, ss.Length);
                    ++ActiveRDMACount;
                    //printf("-> %d, imm %d (%" PRId64 " bytes)\n", ss.DstRank, ss.ImmData, ss.Length);
                    //printf("send %d\n", ss.SrcOffset);
                }
                ibv_wc wc;
                while ((recvMask & iter.RecvMask) != iter.RecvMask) {
                    int rv = CQ->Poll(&wc, 1);
                    if (rv > 0) {
                        Y_ABORT_UNLESS(wc.status == IBV_WC_SUCCESS, "AllGather::Sync fail, status %d", (int)wc.status);
                        if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                            //printf("Got %d\n", wc.imm_data);
                            ++recvDebt;
                            ui64 newBit = ui64(1) << wc.imm_data;
                            if (recvMask & newBit) {
                                Y_ABORT_UNLESS((FutureRecvMask & newBit) == 0, "data from 2 Sync() ahead is impossible");
                                FutureRecvMask |= newBit;
                            } else {
                                recvMask |= newBit;
                            }
                        } else if (wc.opcode == IBV_WC_RDMA_WRITE) {
                            --ActiveRDMACount;
                        } else {
                            Y_ASSERT(0);
                        }
                    } else {
                        if (recvDebt > 0) {
                            PostRecv();
                            --recvDebt;
                        }
                    }
                }
                for (int k = 0; k < iter.ReduceList.ysize(); ++k) {
                    const TReduce& rr = iter.ReduceList[k];
                    ReduceOp->Reduce(myData + rr.DstOffset, myData + rr.SrcOffset, DataSize);
                    //printf("Merge %d -> %d (%d bytes)\n", rr.SrcOffset, rr.DstOffset, DataSize);
                }
                //printf("Iteration %d done\n", z);
            }
            while (recvDebt > 0) {
                PostRecv();
                --recvDebt;
            }
            CurrentOffset = ReadyOffset;
            WasFlushed = false;
            //printf("new cur offset %g\n", (double)CurrentOffset);
            //printf("Sync complete\n");
        }
        void Flush() {
            Y_ASSERT(!WasFlushed);
            CurrentBuffer = 1 - CurrentBuffer;
            CurrentOffset = 0;
            WasFlushed = true;
        }

    public:
        TAllDataSync(size_t bufSize, TPtrArg<TIBMemPool> memPool, TPtrArg<IReduceOp> reduceOp)
            : CurrentBuffer(0)
            , DataSize(0)
            , BufSize(bufSize)
            , CurrentOffset(0)
            , ReadyOffset(0)
            , WasFlushed(false)
            , ActiveRDMACount(0)
            , FutureRecvMask(0)
            , ReduceOp(reduceOp)
        {
            if (memPool) {
                MemBlock[0] = memPool->Alloc(BufSize);
                MemBlock[1] = memPool->Alloc(BufSize);
                CQ = new TComplectionQueue(memPool->GetIBContext(), WR_COUNT);
                SRQ = new TSharedReceiveQueue(memPool->GetIBContext(), WR_COUNT);
                FakeRecvMem = memPool->Alloc(4096);
            } else {
                MemBlock[0] = new TIBMemBlock(BufSize);
                MemBlock[1] = new TIBMemBlock(BufSize);
                CQ = new TComplectionQueue(nullptr, WR_COUNT);
                SRQ = new TSharedReceiveQueue(nullptr, WR_COUNT);
                FakeRecvMem = new TIBMemBlock(4096);
            }
            for (int i = 0; i < WR_COUNT; ++i) {
                PostRecv();
            }
        }
        ~TAllDataSync() {
            while (ActiveRDMACount > 0) {
                ibv_wc wc;
                int rv = CQ->Poll(&wc, 1);
                if (rv > 0) {
                    if (wc.opcode == IBV_WC_RDMA_WRITE) {
                        --ActiveRDMACount;
                    } else {
                        Y_ASSERT(0);
                    }
                }
            }
        }
    };

    class TAllReduce: public IAllReduce {
        TAllDataSync DataSync;
        size_t BufSizeMult;
        size_t ReadyOffsetMult;

    public:
        TAllReduce(size_t bufSize, TPtrArg<TIBMemPool> memPool, TPtrArg<IReduceOp> reduceOp)
            : DataSync(bufSize, memPool, reduceOp)
            , BufSizeMult(0)
            , ReadyOffsetMult(0)
        {
        }
        TAllDataSync& GetDataSync() {
            return DataSync;
        }
        void* GetRawData() override {
            return DataSync.GetRawData();
        }
        size_t GetRawDataSize() override {
            return DataSync.GetRawDataSize();
        }
        void Sync() override {
            DataSync.Sync();
        }
        void Flush() override {
            DataSync.Flush();
        }

        bool Resize(size_t dataSize) override {
            size_t repSize = (dataSize + 63) & (~63ull);
            size_t bufSize = repSize * BufSizeMult;

            if (bufSize > DataSync.BufSize) {
                return false;
            }

            for (int z = 0; z < DataSync.Iterations.ysize(); ++z) {
                TAllDataSync::TIteration& iter = DataSync.Iterations[z];
                for (int i = 0; i < iter.OutList.ysize(); ++i) {
                    TAllDataSync::TSend& snd = iter.OutList[i];
                    snd.Length = dataSize;
                    snd.SrcOffset = snd.Reduce.SrcIndex * repSize;
                    snd.DstOffset = snd.Reduce.DstIndex * repSize;
                }

                for (int i = 0; i < iter.ReduceList.ysize(); ++i) {
                    TAllDataSync::TReduce& red = iter.ReduceList[i];
                    red.SrcOffset = red.SrcIndex * repSize;
                    red.DstOffset = red.DstIndex * repSize;
                }
            }
            DataSync.ReadyOffset = ReadyOffsetMult * repSize;
            DataSync.DataSize = dataSize;
            return true;
        }
        friend class TIBCollective;
    };

    class TAllGather: public IAllGather {
        TAllDataSync DataSync;
        int ColSize;

    public:
        TAllGather(int colSize, size_t bufSize, TPtrArg<TIBMemPool> memPool)
            : DataSync(bufSize, memPool, nullptr)
            , ColSize(colSize)
        {
        }
        TAllDataSync& GetDataSync() {
            return DataSync;
        }
        void* GetRawData() override {
            return DataSync.GetRawData();
        }
        size_t GetRawDataSize() override {
            return DataSync.GetRawDataSize();
        }
        void Sync() override {
            DataSync.Sync();
        }
        void Flush() override {
            DataSync.Flush();
        }

        bool Resize(const TVector<size_t>& szPerRank) override {
            Y_ABORT_UNLESS(szPerRank.ysize() == ColSize, "Invalid size array");

            TVector<size_t> offsets;
            offsets.push_back(0);
            for (int rank = 0; rank < ColSize; ++rank) {
                offsets.push_back(offsets.back() + szPerRank[rank]);
            }

            size_t dataSize = offsets.back();
            if (dataSize > DataSync.BufSize) {
                return false;
            }

            for (int z = 0; z < DataSync.Iterations.ysize(); ++z) {
                TAllDataSync::TIteration& iter = DataSync.Iterations[z];
                for (int i = 0; i < iter.OutList.ysize(); ++i) {
                    TAllDataSync::TSend& snd = iter.OutList[i];
                    int rangeBeg = snd.Gather.RangeBeg;
                    int rangeFin = snd.Gather.RangeFin;
                    snd.Length = offsets[rangeFin] - offsets[rangeBeg];
                    snd.SrcOffset = offsets[rangeBeg];
                    snd.DstOffset = snd.SrcOffset;
                }
            }
            DataSync.DataSize = dataSize;
            return true;
        }
    };

    struct TIBAddr {
        int LID, SL;

        TIBAddr()
            : LID(0)
            , SL(0)
        {
        }
        TIBAddr(int lid, int sl)
            : LID(lid)
            , SL(sl)
        {
        }
    };
    inline bool operator==(const TIBAddr& a, const TIBAddr& b) {
        return a.LID == b.LID && a.SL == b.SL;
    }
    inline bool operator<(const TIBAddr& a, const TIBAddr& b) {
        if (a.LID == b.LID) {
            return a.SL < b.SL;
        }
        return a.LID < b.LID;
    }

    struct TIBAddrHash {
        int operator()(const TIBAddr& a) const {
            return a.LID + a.SL * 4254515;
        }
    };

    class TIBCollective: public IIBCollective {
        struct TPendingMessage {
            int QPN;
            ui64 WorkId;

            TPendingMessage() {
                Zero(*this);
            }
            TPendingMessage(int qpn, ui64 wid)
                : QPN(qpn)
                , WorkId(wid)
            {
            }
        };

        struct TBlockInform {
            TAllDataSync::TBlockInfo RemoteBlocks[2];
            int PSN, QPN;
        };

        struct TPeerConnection {
            TAllDataSync::TBlockInfo RemoteBlocks[2];
            TIntrusivePtr<TRCQueuePair> QP;
        };

        struct TBWTest {
            ui64 Addr;
            ui32 RKey;
        };

        TIntrusivePtr<TIBPort> Port;
        TIntrusivePtr<TIBMemPool> MemPool;
        int ColSize, ColRank;
        TVector<int> Hosts; // host LIDs
        TVector<TVector<int>> HostGroup;
        TVector<TIntrusivePtr<TRCQueuePair>> Peers;
        TIntrusivePtr<TComplectionQueue> CQ;
        TIntrusivePtr<TIBBufferPool> BP;
        ui8 SendCountTable[SEND_COUNT_TABLE_SIZE];
        ui8 RDMACountTable[SEND_COUNT_TABLE_SIZE];
        TDeque<TPendingMessage> Pending;
        TMergePlan MergePlan, ReducePlan;
        int QPNTableSizeLog;

        void WriteCompleted(const ibv_wc& wc) {
            --SendCountTable[wc.qp_num & (SEND_COUNT_TABLE_SIZE - 1)];
            if (wc.opcode == IBV_WC_RDMA_WRITE) {
                --RDMACountTable[wc.qp_num & (SEND_COUNT_TABLE_SIZE - 1)];
            }
            BP->FreeBuf(wc.wr_id);
        }
        bool GetMsg(ui64* resWorkId, int* resQPN, TIBMicroPeerTable* tbl) {
            if (tbl->NeedParsePending()) {
                for (TDeque<TPendingMessage>::iterator z = Pending.begin(); z != Pending.end(); ++z) {
                    if (!tbl->NeedQPN(z->QPN)) {
                        continue;
                    }
                    *resWorkId = z->WorkId;
                    *resQPN = z->QPN;
                    Pending.erase(z);
                    return true;
                }
                //printf("Stop parse pending\n");
                tbl->StopParsePending();
            }
            for (;;) {
                ibv_wc wc;
                int rv = CQ->Poll(&wc, 1);
                if (rv > 0) {
                    Y_ABORT_UNLESS(wc.status == IBV_WC_SUCCESS, "WaitForMsg() fail, status %d", (int)wc.status);
                    if (wc.opcode & IBV_WC_RECV) {
                        BP->RequestPostRecv();
                        if (tbl->NeedQPN(wc.qp_num)) {
                            *resWorkId = wc.wr_id;
                            *resQPN = wc.qp_num;
                            return true;
                        } else {
                            Pending.push_back(TPendingMessage(wc.qp_num, wc.wr_id));
                            BP->PostRecv();
                        }
                    } else {
                        WriteCompleted(wc);
                    }
                } else {
                    return false;
                }
            }
        }

        bool ProcessSendCompletion(const ibv_wc& wc) {
            Y_ABORT_UNLESS(wc.status == IBV_WC_SUCCESS, "WaitForMsg() fail, status %d", (int)wc.status);
            if (wc.opcode & IBV_WC_RECV) {
                BP->RequestPostRecv();
                Pending.push_back(TPendingMessage(wc.qp_num, wc.wr_id));
                BP->PostRecv();
            } else {
                WriteCompleted(wc);
                return true;
            }
            return false;
        }

        void WaitCompletion(ibv_wc* res) {
            ibv_wc& wc = *res;
            for (;;) {
                int rv = CQ->Poll(&wc, 1);
                if (rv > 0 && ProcessSendCompletion(wc)) {
                   break;
                }
            }
        }

        bool TryWaitCompletion() override {
            ibv_wc wc;
            for (;;) {
                int rv = CQ->Poll(&wc, 1);
                if (rv > 0) {
                    if (ProcessSendCompletion(wc)) {
                        return true;
                    }
                } else {
                    return false;
                }
            }
        }

        void WaitCompletion() override {
            ibv_wc wc;
            WaitCompletion(&wc);
        }

        ui64 WaitForMsg(int qpn) {
            for (TDeque<TPendingMessage>::iterator z = Pending.begin(); z != Pending.end(); ++z) {
                if (z->QPN == qpn) {
                    ui64 workId = z->WorkId;
                    Pending.erase(z);
                    return workId;
                }
            }
            ibv_wc wc;
            for (;;) {
                int rv = CQ->Poll(&wc, 1);
                if (rv > 0) {
                    Y_ABORT_UNLESS(wc.status == IBV_WC_SUCCESS, "WaitForMsg() fail, status %d", (int)wc.status);
                    if (wc.opcode & IBV_WC_RECV) {
                        BP->RequestPostRecv();
                        if ((int)wc.qp_num == qpn) {
                            return wc.wr_id;
                        } else {
                            Pending.push_back(TPendingMessage(wc.qp_num, wc.wr_id));
                            BP->PostRecv();
                        }
                    } else {
                        WriteCompleted(wc);
                    }
                }
            }
        }

        bool AllocOperationSlot(TPtrArg<TRCQueuePair> qp) {
            int way = qp->GetQPN() & (SEND_COUNT_TABLE_SIZE - 1);
            if (SendCountTable[way] >= MAX_REQS_PER_PEER) {
                return false;
            }
            ++SendCountTable[way];
            return true;
        }
        bool AllocRDMAWriteSlot(TPtrArg<TRCQueuePair> qp) {
            int way = qp->GetQPN() & (SEND_COUNT_TABLE_SIZE - 1);
            if (SendCountTable[way] >= MAX_REQS_PER_PEER) {
                return false;
            }
            if (RDMACountTable[way] >= MAX_OUTSTANDING_RDMA) {
                return false;
            }
            ++SendCountTable[way];
            ++RDMACountTable[way];
            return true;
        }
        bool TryPostSend(TPtrArg<TRCQueuePair> qp, const void* data, size_t len) {
            if (AllocOperationSlot(qp)) {
                BP->PostSend(qp, data, len);
                return true;
            }
            return false;
        }
        void PostSend(TPtrArg<TRCQueuePair> qp, const void* data, size_t len) {
            while (!TryPostSend(qp, data, len)) {
                WaitCompletion();
            }
        }
        int GetRank() override {
            return ColRank;
        }
        int GetSize() override {
            return ColSize;
        }
        int GetGroupTypeCount() override {
            return HostGroup.ysize();
        }
        int GetQPN(int rank) override {
            if (rank == ColRank) {
                Y_ASSERT(0 && "there is no qpn connected to localhost");
                return 0;
            }
            return Peers[rank]->GetQPN();
        }

        void Start(const TCollectiveLinkSet& links) override {
            Hosts = links.Hosts;
            HostGroup = links.HostGroup;
            for (int k = 0; k < ColSize; ++k) {
                if (k == ColRank) {
                    continue;
                }
                const TCollectiveLinkSet::TLinkInfo& lnk = links.Links[k];
                ibv_ah_attr peerAddr;
                MakeAH(&peerAddr, Port, Hosts[k], COL_SERVICE_LEVEL);
                Peers[k]->Init(peerAddr, lnk.QPN, lnk.PSN);
            }

            //CreatePow2Merge(&MergePlan, ColSize);
            //CreatePow2Merge(&ReducePlan, ColSize);
            CreateGroupMerge(&MergePlan, AA_STAR, HostGroup);
            CreateGroupMerge(&ReducePlan, AA_POW2_MERGE, HostGroup);
        }

        void CreateDataSyncQPs(
            TPtrArg<TComplectionQueue> cq,
            TPtrArg<TSharedReceiveQueue> srq,
            TPtrArg<TIBMemBlock> memBlock0,
            TPtrArg<TIBMemBlock> memBlock1,
            const TMergePlan& plan,
            THashMap<TIBAddr, TPeerConnection, TIBAddrHash>* res) {
            THashMap<TIBAddr, TPeerConnection, TIBAddrHash>& connections = *res;

            TIBMemBlock* memBlock[2] = {memBlock0, memBlock1};

            // make full peer list
            TVector<TIBAddr> peerList;
            for (int z = 0; z < plan.Iterations.ysize(); ++z) {
                const TMergeRecord& rr = plan.Iterations[z].Ops[ColRank];
                for (int i = 0; i < rr.OutList.ysize(); ++i) {
                    const TMergeRecord::TTransfer& tr = rr.OutList[i];
                    peerList.push_back(TIBAddr(tr.DstRank, tr.SL));
                }
                for (int i = 0; i < rr.InList.ysize(); ++i) {
                    const TMergeRecord::TInTransfer& tr = rr.InList[i];
                    peerList.push_back(TIBAddr(tr.SrcRank, tr.SL));
                }
            }
            Sort(peerList.begin(), peerList.end());
            peerList.erase(Unique(peerList.begin(), peerList.end()), peerList.end());

            // establish QPs and exchange mem block handlers
            for (int z = 0; z < peerList.ysize(); ++z) {
                const TIBAddr& ibAddr = peerList[z];
                int dstRank = ibAddr.LID;
                TPeerConnection& dst = connections[ibAddr];

                dst.QP = new TRCQueuePair(Port->GetCtx(), cq, srq, TAllDataSync::WR_COUNT);

                TBlockInform myBlock;
                for (int k = 0; k < 2; ++k) {
                    myBlock.RemoteBlocks[k].Addr = memBlock[k]->GetAddr();
                    myBlock.RemoteBlocks[k].Key = memBlock[k]->GetMemRegion()->GetRKey();
                }
                myBlock.PSN = dst.QP->GetPSN();
                myBlock.QPN = dst.QP->GetQPN();
                PostSend(Peers[dstRank], &myBlock, sizeof(myBlock));
            }

            for (int z = 0; z < peerList.ysize(); ++z) {
                const TIBAddr& ibAddr = peerList[z];
                int dstRank = ibAddr.LID;
                int sl = COL_DATA_SERVICE_LEVEL + ClampVal(ibAddr.SL, 0, COL_DATA_SERVICE_LEVEL_COUNT);

                TPeerConnection& dst = connections[ibAddr];

                ui64 wr_id = WaitForMsg(Peers[dstRank]->GetQPN());
                TIBRecvPacketProcess pkt(*BP, wr_id);
                const TBlockInform& info = *(TBlockInform*)pkt.GetData();
                ibv_ah_attr peerAddr;
                MakeAH(&peerAddr, Port, Hosts[dstRank], COL_DATA_SERVICE_LEVEL + sl);
                dst.QP->Init(peerAddr, info.QPN, info.PSN);
                dst.RemoteBlocks[0] = info.RemoteBlocks[0];
                dst.RemoteBlocks[1] = info.RemoteBlocks[1];
            }
            Fence();
        }

        IAllGather* CreateAllGather(const TVector<size_t>& szPerRank) override {
            const TMergePlan& plan = MergePlan;

            Y_ABORT_UNLESS(szPerRank.ysize() == ColSize, "Invalid size array");

            size_t totalSize = 0;
            for (int i = 0; i < szPerRank.ysize(); ++i) {
                totalSize += szPerRank[i];
            }
            size_t bufSize = 4096;
            while (totalSize >= bufSize) {
                bufSize *= 2;
            }

            TAllGather* res = new TAllGather(ColSize, bufSize, MemPool);
            TAllDataSync& ds = res->GetDataSync();

            THashMap<TIBAddr, TPeerConnection, TIBAddrHash> connections;
            CreateDataSyncQPs(ds.CQ, ds.SRQ, ds.MemBlock[0], ds.MemBlock[1], plan, &connections);

            // build plan
            for (int z = 0; z < plan.Iterations.ysize(); ++z) {
                const TMergeRecord& rr = plan.Iterations[z].Ops[ColRank];
                if (rr.OutList.empty() && rr.InList.empty()) {
                    continue;
                }
                TAllDataSync::TIteration& iter = ds.Iterations.emplace_back();
                for (int i = 0; i < rr.OutList.ysize(); ++i) {
                    const TMergeRecord::TTransfer& tr = rr.OutList[i];
                    TAllDataSync::TSend& snd = iter.OutList.emplace_back();
                    TPeerConnection& pc = connections[TIBAddr(tr.DstRank, tr.SL)];

                    snd.ImmData = tr.Id;
                    snd.Gather.RangeBeg = tr.RangeBeg;
                    snd.Gather.RangeFin = tr.RangeFin;
                    snd.QP = pc.QP;
                    snd.RemoteBlocks[0] = pc.RemoteBlocks[0];
                    snd.RemoteBlocks[1] = pc.RemoteBlocks[1];
                    snd.DstRank = tr.DstRank;
                }
                for (int i = 0; i < rr.InList.ysize(); ++i) {
                    const TMergeRecord::TInTransfer& tr = rr.InList[i];
                    TAllDataSync::TRecv& rcv = iter.InList.emplace_back();
                    TPeerConnection& pc = connections[TIBAddr(tr.SrcRank, tr.SL)];
                    rcv.QP = pc.QP;
                    rcv.SrcRank = tr.SrcRank;
                }
                iter.RecvMask = rr.RecvMask;
            }
            bool rv = res->Resize(szPerRank);
            Y_ABORT_UNLESS(rv, "oops");

            return res;
        }
        IAllGather* CreateAllGather(size_t szPerRank) override {
            TVector<size_t> arr;
            arr.resize(ColSize, szPerRank);
            return CreateAllGather(arr);
        }

        IAllReduce* CreateAllReduce(size_t dataSize, TPtrArg<IReduceOp> reduceOp) override {
            const TMergePlan& plan = ReducePlan;

            size_t bufSizeMult = plan.MaxRankReceiveCount + 1;
            size_t bufSize = 4096;
            {
                size_t sz = (dataSize + 64) * bufSizeMult;
                while (sz > bufSize) {
                    bufSize *= 2;
                }
            }

            TAllReduce* res = new TAllReduce(bufSize, MemPool, reduceOp);
            TAllDataSync& ds = res->GetDataSync();

            THashMap<TIBAddr, TPeerConnection, TIBAddrHash> connections;
            CreateDataSyncQPs(ds.CQ, ds.SRQ, ds.MemBlock[0], ds.MemBlock[1], plan, &connections);

            // build plan
            int currentDataOffset = 0;
            for (int z = 0; z < plan.Iterations.ysize(); ++z) {
                const TMergeRecord& rr = plan.Iterations[z].Ops[ColRank];
                if (rr.OutList.empty() && rr.InList.empty()) {
                    continue;
                }
                TAllDataSync::TIteration& iter = ds.Iterations.emplace_back();
                for (int i = 0; i < rr.OutList.ysize(); ++i) {
                    const TMergeRecord::TTransfer& tr = rr.OutList[i];
                    TAllDataSync::TSend& snd = iter.OutList.emplace_back();
                    TPeerConnection& pc = connections[TIBAddr(tr.DstRank, tr.SL)];

                    snd.ImmData = tr.Id;
                    snd.Reduce.SrcIndex = currentDataOffset;
                    snd.Reduce.DstIndex = tr.Id + 1;
                    snd.QP = pc.QP;
                    snd.RemoteBlocks[0] = pc.RemoteBlocks[0];
                    snd.RemoteBlocks[1] = pc.RemoteBlocks[1];
                    snd.DstRank = tr.DstRank;
                }

                for (int i = 0; i < rr.InList.ysize(); ++i) {
                    const TMergeRecord::TInTransfer& tr = rr.InList[i];
                    TAllDataSync::TRecv& rcv = iter.InList.emplace_back();
                    TPeerConnection& pc = connections[TIBAddr(tr.SrcRank, tr.SL)];
                    rcv.QP = pc.QP;
                    rcv.SrcRank = tr.SrcRank;
                }
                iter.RecvMask = rr.RecvMask;

                TVector<int> inputOffset;
                inputOffset.push_back(currentDataOffset);
                int newDataOffset = currentDataOffset;
                for (int i = 0; i < 64; ++i) {
                    if (rr.RecvMask & (1ull << i)) {
                        int offset = i + 1;
                        inputOffset.push_back(offset);
                        newDataOffset = Max(offset, newDataOffset);
                    }
                }
                for (int i = 0; i < inputOffset.ysize(); ++i) {
                    if (inputOffset[i] == newDataOffset) {
                        continue;
                    }
                    TAllDataSync::TReduce& red = iter.ReduceList.emplace_back();
                    red.SrcIndex = inputOffset[i];
                    red.DstIndex = newDataOffset;
                }
                currentDataOffset = newDataOffset;
            }
            res->BufSizeMult = bufSizeMult;
            res->ReadyOffsetMult = currentDataOffset;

            bool rv = res->Resize(dataSize);
            Y_ABORT_UNLESS(rv, "oops");

            return res;
        }

        void Fence() override {
            const TMergePlan& plan = ReducePlan;

            for (int z = 0; z < plan.Iterations.ysize(); ++z) {
                const TMergeRecord& rr = plan.Iterations[z].Ops[ColRank];
                for (int i = 0; i < rr.OutList.ysize(); ++i) {
                    const TMergeRecord::TTransfer& tr = rr.OutList[i];
                    char c;
                    PostSend(Peers[tr.DstRank], &c, sizeof(c));
                }

                for (int i = 0; i < rr.InList.ysize(); ++i) {
                    const TMergeRecord::TInTransfer& tr = rr.InList[i];
                    ui64 wr_id = WaitForMsg(Peers[tr.SrcRank]->GetQPN());
                    TIBRecvPacketProcess pkt(*BP, wr_id);
                }
            }
        }
        void RunBWTest(int groupType, int delta, int* targetRank, float* res) override {
            const int BUF_SIZE = 8 * 1024 * 1024;
            TIntrusivePtr<TIBMemBlock> sendMem, recvMem;
            sendMem = MemPool->Alloc(BUF_SIZE);
            recvMem = MemPool->Alloc(BUF_SIZE);

            int myGroup = HostGroup[groupType][ColRank];
            int myGroupPos = 0;
            TVector<int> gg;
            Y_ASSERT(HostGroup[groupType].ysize() == ColSize);
            for (int rank = 0; rank < ColSize; ++rank) {
                if (HostGroup[groupType][rank] == myGroup) {
                    if (rank == ColRank) {
                        myGroupPos = gg.ysize();
                    }
                    gg.push_back(rank);
                }
            }
            if (delta >= gg.ysize()) {
                *targetRank = -1;
                *res = 0;
                return;
            }

            int sendRank = gg[(myGroupPos + delta) % gg.ysize()];
            int recvRank = gg[(myGroupPos + gg.ysize() - delta) % gg.ysize()];
            *targetRank = sendRank;
            TIntrusivePtr<TRCQueuePair> sendRC = Peers[sendRank];
            TIntrusivePtr<TRCQueuePair> recvRC = Peers[recvRank];
            {
                TBWTest bw;
                bw.Addr = recvMem->GetAddr();
                bw.RKey = recvMem->GetMemRegion()->GetRKey();
                PostSend(recvRC, &bw, sizeof(bw));
            }
            TBWTest dstMem;
            {
                ui64 wr_id = WaitForMsg(sendRC->GetQPN());
                TIBRecvPacketProcess pkt(*BP, wr_id);
                dstMem = *(TBWTest*)pkt.GetData();
            }
            // run
            TVector<double> score;
            for (int iter = 0; iter < 5; ++iter) {
                while (!AllocRDMAWriteSlot(sendRC)) {
                    WaitCompletion();
                    Y_ASSERT(0 && "measurements are imprecise");
                }
                NHPTimer::STime t;
                NHPTimer::GetTime(&t);
                sendRC->PostRDMAWrite(dstMem.Addr, dstMem.RKey, sendMem->GetMemRegion(), 0, sendMem->GetData(), BUF_SIZE);
                for (;;) {
                    ibv_wc wc;
                    WaitCompletion(&wc);
                    if (wc.opcode == IBV_WC_RDMA_WRITE) {
                        if (wc.qp_num != (ui32)sendRC->GetQPN()) {
                            abort();
                        }
                        break;
                    }
                }
                double tPassed = NHPTimer::GetTimePassed(&t);
                double speed = BUF_SIZE / tPassed / 1000000000.0; // G/sec
                score.push_back(speed);
            }
            Sort(score.begin(), score.end());
            // signal completion & wait for signal
            *res = score[score.size() / 2];
            {
                char bb;
                PostSend(sendRC, &bb, sizeof(bb));
                ui64 wr_id = WaitForMsg(recvRC->GetQPN());
                TIBRecvPacketProcess pkt(*BP, wr_id);
            }
        }
        bool TrySendMicro(int dstRank, const void* data, int dataSize) override {
            return TryPostSend(Peers[dstRank], data, dataSize);
        }
        void InitPeerTable(TIBMicroPeerTable* res) override {
            res->Init(QPNTableSizeLog);
        }
        void RdmaWrite(const TVector<TRdmaRequest>& reqs) override {
            TVector<TVector<int>> reqPerRank;
            reqPerRank.resize(ColSize);
            int reqCount = reqs.ysize();
            for (int i = 0; i < reqCount; ++i) {
                reqPerRank[reqs[i].DstRank].push_back(i);
            }
            int inFlight = 0; // IB congestion control sucks :/ so we limit number of simultaneous rdmas
            int startRank = ColRank;
            while (reqCount > 0) {
                if (inFlight < MAX_TOTAL_RDMA) {
                    for (int z = 0; z < ColSize; ++z) {
                        int dstRank = (startRank + 1 + z) % ColSize;
                        if (reqPerRank[dstRank].empty()) {
                            continue;
                        }
                        Y_ASSERT(dstRank != ColRank && "sending self is meaningless");
                        TRCQueuePair* qp = Peers[dstRank].Get();
                        if (AllocRDMAWriteSlot(qp)) {
                            const TRdmaRequest& rr = reqs[reqPerRank[dstRank].back()];
                            qp->PostRDMAWrite(rr.RemoteAddr, rr.RemoteKey, rr.LocalAddr, rr.LocalKey, 0, rr.Size);
                            reqPerRank[dstRank].pop_back();
                            if (++inFlight >= MAX_TOTAL_RDMA) {
                                startRank = dstRank;
                                break;
                            }
                        }
                    }
                }
                {
                    ibv_wc wc;
                    WaitCompletion(&wc);
                    if (wc.opcode == IBV_WC_RDMA_WRITE) {
                        --inFlight;
                        --reqCount;
                    }
                }
            }
        }

    public:
        TIBCollective(TPtrArg<TIBPort> port, TPtrArg<TIBMemPool> memPool,
                      const TCollectiveInit& params,
                      TCollectiveLinkSet* resLinks)
            : Port(port)
            , MemPool(memPool)
            , QPNTableSizeLog(0)
        {
            ColSize = params.Size;
            ColRank = params.Rank;

            int maxOutstandingQueries = MAX_REQS_PER_PEER * ColSize + 10;
            CQ = new TComplectionQueue(Port->GetCtx(), maxOutstandingQueries * 2);
            BP = new TIBBufferPool(Port->GetCtx(), maxOutstandingQueries);

            Peers.resize(ColSize);
            resLinks->Links.resize(ColSize);
            TVector<int> qpnArr;
            for (int k = 0; k < ColSize; ++k) {
                if (k == ColRank) {
                    continue;
                }
                TRCQueuePair* rc = new TRCQueuePair(Port->GetCtx(), CQ, BP->GetSRQ(), MAX_REQS_PER_PEER);
                Peers[k] = rc;
                TCollectiveLinkSet::TLinkInfo& lnk = resLinks->Links[k];
                lnk.PSN = rc->GetPSN();
                lnk.QPN = rc->GetQPN();

                qpnArr.push_back(lnk.QPN);
            }
            resLinks->Hosts.resize(ColSize);
            resLinks->Hosts[ColRank] = Port->GetLID();

            static_assert(MAX_REQS_PER_PEER < 256, "expect MAX_REQS_PER_PEER < 256"); // sent count will fit into SendCountTable[]
            Zero(SendCountTable);
            Zero(RDMACountTable);

            if (!qpnArr.empty()) {
                for (;;) {
                    TVector<ui8> qpnTable;
                    int qpnTableSize = 1 << QPNTableSizeLog;
                    qpnTable.resize(qpnTableSize, 0);
                    bool ok = true;
                    for (int i = 0; i < qpnArr.ysize(); ++i) {
                        int idx = qpnArr[i] & (qpnTableSize - 1);
                        if (++qpnTable[idx] == 2) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) {
                        break;
                    }
                    ++QPNTableSizeLog;
                }
                //printf("QPN table, size_log %d\n", QPNTableSizeLog);
            }
        }

        friend class TIBRecvMicro;
    };

    TIBRecvMicro::TIBRecvMicro(IIBCollective* col, TIBMicroPeerTable* peerTable)
        : IB(*(TIBCollective*)col)
    {
        Y_ASSERT(typeid(IB) == typeid(TIBCollective));
        if (IB.GetMsg(&Id, &QPN, peerTable)) {
            Data = IB.BP->GetBufData(Id);
        } else {
            Data = nullptr;
        }
    }

    TIBRecvMicro::~TIBRecvMicro() {
        if (Data) {
            IB.BP->FreeBuf(Id);
            IB.BP->PostRecv();
        }
    }

    IIBCollective* CreateCollective(const TCollectiveInit& params, TCollectiveLinkSet* resLinks) {
        return new TIBCollective(GetIBDevice(), GetIBMemPool(), params, resLinks);
    }
}
