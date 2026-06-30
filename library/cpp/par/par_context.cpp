#include "par_context.h"
#include "par.h"

#include <util/random/random.h>
#include <util/random/shuffle.h>
#include <library/cpp/deprecated/atomic/atomic_ops.h>
#include <util/system/types.h>
#include <util/system/yassert.h>
#include <util/system/yield.h>

const int MAX_SIMULTANEOUS_SENDS = 1;

namespace NPar {
    static int IsAllSet(const TVector<bool>& v) {
        int res = true;
        for (int i = 0; i < v.ysize(); ++i) {
            res &= (int)v[i];
        }
        return res;
    }

    static void FilterSelected(TVector<TVector<int>>* arr, const TVector<TVector<bool>>& ready, bool* hasNotReadyHost) {
        for (int hostId = 0; hostId < arr->ysize(); ++hostId) {
            TVector<int>& vec = (*arr)[hostId];
            int dst = 0;
            for (int i = 0; i < vec.ysize(); ++i) {
                if (IsAllSet(ready[vec[i]])) {
                    vec[dst++] = vec[i];
                } else {
                    *hasNotReadyHost = true;
                }
            }
            vec.resize(dst);
        }
    }

    struct TCtxTransferSrc {
        TVector<int> SrcComps;
    };
    struct TRequiredTransfer {
        int EnvId, HostId, Part;
        int SrcGroupId;
        int DstComp, SrcCount;
    };
    struct TRequiredTransferCmp {
        bool operator()(const TRequiredTransfer& a, const TRequiredTransfer& b) const {
            return a.SrcCount < b.SrcCount;
        }
    };
    void TContextDistributor::DoSend() {
        CHROMIUM_TRACE_FUNCTION();

        if (QueryProc.Get() == nullptr)
            return;

        TVector<TCtxTransferSrc> srcHolder;
        TVector<TRequiredTransfer> xferList;
        srcHolder.reserve(100);

        bool allComplete = true;
        for (THashMap<int, TFullCtxInfo>::iterator i = EnvId2Info.begin(); i != EnvId2Info.end(); ++i) {
            TFullCtxInfo& info = i->second;
            for (int hostId = 0; hostId < info.HostId2Computer.ysize(); ++hostId) {
                bool& isFullyDistributed = info.IsFullyDistributed[hostId];
                if (isFullyDistributed)
                    continue;

                isFullyDistributed = true;
                const TVector<int>& compList = info.HostId2Computer[hostId];
                int partCount = info.Data[hostId].GetPartCount();
                for (int part = 0; part < partCount; ++part) {
                    int srcCount = 0;
                    int srcGroupId = srcHolder.ysize();
                    TCtxTransferSrc& src = srcHolder.emplace_back();
                    TVector<int> target;
                    if (info.Data[hostId].Info.Get())
                        src.SrcComps.push_back(-1);
                    for (int z = 0; z < compList.ysize(); ++z) {
                        int compId = compList[z];
                        if (info.ReadyMask[compId][part]) {
                            ++srcCount;
                            if (ComputerSendCount[compId] < MAX_SIMULTANEOUS_SENDS) {
                                src.SrcComps.push_back(compId);
                            }
                        } else {
                            isFullyDistributed = false;
                            allComplete = false;
                            if (!info.CopyInitiated[compId][part]) {
                                target.push_back(compId);
                            }
                        }
                    }
                    if (!target.empty() && !src.SrcComps.empty()) {
                        Shuffle(src.SrcComps.begin(), src.SrcComps.end());
                        for (int z = 0; z < target.ysize(); ++z) {
                            TRequiredTransfer xfer;
                            xfer.EnvId = i->first;
                            xfer.HostId = hostId;
                            xfer.Part = part;
                            xfer.SrcGroupId = srcGroupId;
                            xfer.DstComp = target[z];
                            xfer.SrcCount = srcCount;
                            xferList.push_back(xfer);
                        }
                    }
                }
            }
        }
        AtomicSet(DistributionIsComplete, allComplete);

        Shuffle(xferList.begin(), xferList.end());
        std::sort(xferList.begin(), xferList.end(), TRequiredTransferCmp());

        for (int i = 0; i < xferList.ysize(); ++i) {
            const TRequiredTransfer& xfer = xferList[i];
            TFullCtxInfo& fullCtx = EnvId2Info[xfer.EnvId];
            const TCtxDataPart& data = fullCtx.Data[xfer.HostId];
            if (fullCtx.CopyInitiated[xfer.DstComp][xfer.Part])
                continue;
            const TVector<int>& srcComps = srcHolder[xfer.SrcGroupId].SrcComps;
            for (int z = 0; z < srcComps.ysize(); ++z) {
                int srcComp = srcComps[z];
                int& sendCount = ComputerSendCount[srcComp + 1];
                if (sendCount >= MAX_SIMULTANEOUS_SENDS)
                    continue;
                PAR_DEBUG_LOG << Sprintf("forward context %d (p%d v%d) from %d to %d\n", xfer.EnvId, xfer.HostId, data.Version, srcComp, xfer.DstComp);
                if (srcComp < 0) {
                    TVector<char> buf;
                    MakeDataPartCmd(xfer.EnvId, data.Version, xfer.Part, data.GetPartCount(), data.KeepRawData, data.BinData[xfer.Part], &buf);
                    PerformSend(srcComp, xfer.DstComp,
                                xfer.DstComp, "ctx",
                                fullCtx, xfer.EnvId, xfer.HostId, xfer.Part, data.Version, &buf);

                } else {
                    TVector<char> buf;
                    {
                        TContextForwardCmd fwdCmd(xfer.EnvId, xfer.DstComp, xfer.Part);
                        SerializeToMem(&buf, fwdCmd);
                    }
                    PerformSend(srcComp, xfer.DstComp,
                                srcComp, "ctx_fwd",
                                fullCtx, xfer.EnvId, xfer.HostId, xfer.Part, data.Version, &buf);
                }
                break; // no need to send multiple times
            }
        }
    }

    void TContextDistributor::PerformSend(int srcComp, int dstComp,
                                          int queryComp, const char* cmd,
                                          TFullCtxInfo& fullCtx, int envId, int hostId, int part, int dataVersion,
                                          TVector<char>* buf) {
        CHROMIUM_TRACE_FUNCTION();
        ++ComputerSendCount[srcComp + 1];
        ++QueryId;
        TransferInfos[QueryId] = TTransferInfo(envId, hostId, part, srcComp, dstComp, dataVersion);
        fullCtx.CopyInitiated[dstComp][part] = true;
        AtomicAdd(ActiveReqCount, 1);
        QueryProc->SendQuery(queryComp, cmd, buf, this, QueryId);
    }

    void TContextDistributor::GotResponse(int id, TVector<char>*) {
        CHROMIUM_TRACE_FUNCTION();
        TGuard<TMutex> g(Sync);
        THashMap<int, TTransferInfo>::iterator i = TransferInfos.find(id);
        Y_ASSERT(i != TransferInfos.end());
        const TTransferInfo& xfer = i->second;

        --ComputerSendCount[xfer.SenderComp + 1];

        TFullCtxInfo& info = EnvId2Info[xfer.EnvId];
        if (xfer.Version == info.MaxVersion) {
            // new comp is up to date
            Y_ASSERT(info.ReadyMask[xfer.DstComp][xfer.Part] == false);
            info.ReadyMask[xfer.DstComp][xfer.Part] = true;
            PAR_DEBUG_LOG << Sprintf("Comp %d confirmed env %d version %d part %d\n", xfer.DstComp, xfer.EnvId, xfer.Version, xfer.Part);
        }
        TransferInfos.erase(i);

        DoSend();
        AtomicAdd(ActiveReqCount, -1);
    }

    TContextDistributor::TContextDistributor(TRemoteQueryProcessor* queryProc, TLocalDataBuffer* writeBuffer)
        : QueryProc(queryProc)
        , DistributionIsComplete(true)
        , QueryId(0)
        , ActiveReqCount(0)
        , WriteBuffer(writeBuffer)
    {
        SlaveCount = QueryProc.Get() ? QueryProc->GetSlaveCount() : 1;
        ComputerSendCount.resize(SlaveCount + 1, 0);
    }

    TContextDistributor::~TContextDistributor() {
        while (AtomicGet(ActiveReqCount))
            ThreadYield();
    }

    void TContextDistributor::CreateNewContext(int envId, int parentEnvId, const TVector<int>& computer2HostId) {
        CHROMIUM_TRACE_FUNCTION();
        TGuard<TMutex> g(Sync);
        Y_ASSERT(QueryProc.Get() == nullptr || computer2HostId.ysize() == GetSlaveCount());
        Y_ASSERT(envId > 0);

        if (envId <= 0) {
            Y_ASSERT(0 && "envId should be positive");
            return;
        }

        bool existingEnv = false;
        if (EnvId2Info.find(envId) != EnvId2Info.end()) {
            existingEnv = true;
            return;
        }

        if (parentEnvId != 0 && EnvId2Info.find(parentEnvId) == EnvId2Info.end()) {
            Y_ASSERT(0 && "invalid parentEnvId");
            return;
        }

        TFullCtxInfo& dst = EnvId2Info[envId];
        dst.ParentEnvId = parentEnvId;

        if (existingEnv) {
            Y_ABORT_UNLESS(dst.Computer2HostId == computer2HostId);
        }
        dst.Computer2HostId = computer2HostId;

        for (int i = 0; i < computer2HostId.ysize(); ++i) {
            int host = computer2HostId[i];
            if (host >= dst.HostId2Computer.ysize())
                dst.HostId2Computer.resize(host + 1);
            dst.HostId2Computer[host].push_back(i);
        }
        int computerCount = computer2HostId.ysize();
        dst.ReadyMask.resize(computerCount);
        dst.CopyInitiated.resize(computerCount);
        int hostIdCount = dst.HostId2Computer.ysize();
        dst.Data.resize(hostIdCount);
        dst.IsFullyDistributed.assign(hostIdCount, false);

        for (int hostId = 0; hostId < hostIdCount; ++hostId) {
            TCtxDataPart& part = dst.Data[hostId];
            AssignData(&part, dst, nullptr);
            dst.ResetHostIdReady(hostId, part.GetPartCount());
        }

        DoSend();
    }

    void TContextDistributor::AssignData(TCtxDataPart* part, TFullCtxInfo& dst, const IObjectBase* data) {
        CHROMIUM_TRACE_FUNCTION();
        part->Info = new TContextDataHolder;
        part->Info->Computer2HostId = dst.Computer2HostId;
        Y_ASSERT(data == nullptr || part->Info->Data != data); // otherwise stale execution can get wrong context data
        part->Info->Data = const_cast<IObjectBase*>(data);
        SerializeToMem(&part->BinData, *part->Info);
    }

    void TContextDistributor::SetContextData(int envId, int hostId, const IObjectBase* data, EKeepDataFlags keepContextRawData) {
        CHROMIUM_TRACE_FUNCTION();
        bool keepRemoteRawData = keepContextRawData & (KEEP_CONTEXT_RAW_DATA & ~KEEP_CONTEXT_ON_MASTER);
        {
            TGuard<TMutex> g(Sync);
            Y_ASSERT(EnvId2Info.find(envId) != EnvId2Info.end());
            TFullCtxInfo& dst = EnvId2Info[envId];
            TCtxDataPart& part = dst.Data[hostId];
            if (part.Version >= dst.MaxVersion) {
                ++part.Version;
                dst.MaxVersion = part.Version;
            } else {
                part.Version = dst.MaxVersion;
            }
            AssignData(&part, dst, data);
            part.KeepRawData = keepRemoteRawData;
            dst.ResetHostIdReady(hostId, part.GetPartCount());

            DoSend();
        }
        if (!keepRemoteRawData) {
            DeleteContextRawData(envId, hostId, keepContextRawData & KEEP_CONTEXT_ON_MASTER);
        }
    }

    void TContextDistributor::SetContextData(int envId, const TVector<int>& compIds, const TVector<i64>& dataIds, EKeepDataFlags keepContextRawData) {
        CHROMIUM_TRACE_FUNCTION();
        Y_ASSERT(compIds.ysize() == dataIds.ysize());
        TGuard<TMutex> g(Sync);
        Y_ASSERT(EnvId2Info.find(envId) != EnvId2Info.end());
        TFullCtxInfo& fullCtx = EnvId2Info[envId];
        Y_ASSERT(fullCtx.Data.ysize() == compIds.ysize());
        ++fullCtx.MaxVersion;
        for (int hostId = 0; hostId < compIds.ysize(); ++hostId) {
            TCtxDataPart& part = fullCtx.Data[hostId];
            part.Version = fullCtx.MaxVersion;
            part.Info = nullptr;
            part.BinData.clear();
            fullCtx.IsFullyDistributed[hostId] = false;
            fullCtx.ResetHostIdReady(hostId, 1);
        }
        for (int hostId = 0; hostId < compIds.ysize(); ++hostId) {
            int srcComp = compIds[hostId];
            if (srcComp == -1) {
                TCtxDataPart& part = fullCtx.Data[hostId];
                TObj<IObjectBase> obj = WriteBuffer->GetObject(dataIds[hostId], TLocalDataBuffer::DO_EXTRACT);
                AssignData(&part, fullCtx, obj);
            } else {
                int srcHostId = fullCtx.Computer2HostId[srcComp];
                {
                    TContextSetData cmd;
                    cmd.EnvId = envId;
                    cmd.Version = fullCtx.MaxVersion;
                    cmd.Computer2HostId = fullCtx.Computer2HostId;
                    cmd.DataId = dataIds[hostId];
                    cmd.KeepRawData = keepContextRawData == KEEP_CONTEXT_RAW_DATA;
                    if (srcHostId == hostId) {
                        cmd.DstCompId = srcComp;
                        PAR_DEBUG_LOG << Sprintf("context %d (p%d v%d) set wb data on %d\n", envId, hostId, fullCtx.MaxVersion, srcComp);
                    } else {
                        const TVector<int>& dstList = fullCtx.HostId2Computer[hostId];
                        cmd.DstCompId = dstList[RandomNumber(dstList.size())];
                        PAR_DEBUG_LOG << Sprintf("context %d (p%d v%d) copy wb data from %d to %d\n", envId, hostId, fullCtx.MaxVersion, srcComp, cmd.DstCompId);
                    }

                    TVector<char> buf;
                    SerializeToMem(&buf, cmd);
                    PerformSend(srcComp, cmd.DstCompId,
                                srcComp, "ctx_wb",
                                fullCtx, envId, hostId, 0, fullCtx.MaxVersion, &buf);
                }
            }
        }
        DoSend();
    }

    class TFreeMemWait: public IRemoteQueryResponseNotify {
        int ReqCount;
        TSystemEvent Ready;

    public:
        TFreeMemWait(int reqCount)
            : ReqCount(reqCount)
        {
            if (ReqCount)
                Ready.Reset();
            else
                Ready.Signal();
        }
        void GotResponse(int, TVector<char>*) override {
            CHROMIUM_TRACE_FUNCTION();
            if ((--ReqCount) == 0)
                Ready.Signal();
        }
        void Wait() {
            Ready.Wait();
        }
    };

    void TContextDistributor::DeleteContextRawData(int envId, int hostId, bool keepContextOnMaster) {
        CHROMIUM_TRACE_FUNCTION();
        WaitAllDistributionActivity(); // can wait more fine grained, but delay should not be much higher

        Y_ASSERT(EnvId2Info.find(envId) != EnvId2Info.end());
        TFullCtxInfo& dst = EnvId2Info[envId];

        // free mem on slaves
        if (QueryProc.Get()) {
            TVector<char> cmd;
            {
                TContextFreeCmd freeCmd;
                freeCmd.EnvId = envId;
                SerializeToMem(&cmd, freeCmd);
            }

            const TVector<int>& compList = dst.HostId2Computer[hostId];
            TIntrusivePtr<TFreeMemWait> fmw = new TFreeMemWait(compList.ysize());
            for (int i = 0; i < compList.ysize(); ++i) {
                int compId = compList[i];
                TVector<char> tmp = cmd;
                QueryProc->SendQuery(compId, "ctx_free", &tmp, fmw.Get(), i);
            }
            fmw->Wait();
        }

        // free mem on master
        TCtxDataPart& part = dst.Data[hostId];
        TVector<TVector<char>>().swap(part.BinData);
        if (!keepContextOnMaster) {
            part.Info = nullptr;
        }
    }

    void TContextDistributor::GetVersions(int envId, int* hostIdCount, THashMap<int, int>* envId2version) {
        CHROMIUM_TRACE_FUNCTION();
        TGuard<TMutex> g(Sync);
        Y_ASSERT(EnvId2Info.find(envId) != EnvId2Info.end() && "unregistered envId");
        const TFullCtxInfo& dst = EnvId2Info[envId];
        *hostIdCount = dst.Data.ysize();
        Y_ASSERT(dst.Data.ysize() == dst.HostId2Computer.ysize());
        envId2version->clear();
        for (int curEnvId = envId; curEnvId;) {
            const TFullCtxInfo& parent = EnvId2Info[curEnvId];
            (*envId2version)[curEnvId] = parent.MaxVersion;
            curEnvId = parent.ParentEnvId;
        }
    }

    bool TContextDistributor::GetReadyMask(const THashMap<int, int>& envId2Version,
                                           TVector<TVector<int>>* hostId2Computer,
                                           bool* hasNotReadyHost) {
        CHROMIUM_TRACE_FUNCTION();
        TGuard<TMutex> g(Sync);
        *hasNotReadyHost = false;
        TVector<TVector<int>> tmpBuf;
        if (!hostId2Computer)
            hostId2Computer = &tmpBuf;
        hostId2Computer->resize(0);
        for (THashMap<int, int>::const_iterator i = envId2Version.begin(); i != envId2Version.end(); ++i) {
            int envId = i->first;
            int version = i->second;
            Y_ASSERT(EnvId2Info.find(envId) != EnvId2Info.end());
            const TFullCtxInfo& ctx = EnvId2Info[envId];
            if (ctx.MaxVersion != version) {
                hostId2Computer->clear();
                return false;
            }
            if (hostId2Computer->empty())
                *hostId2Computer = ctx.HostId2Computer;
            FilterSelected(hostId2Computer, ctx.ReadyMask, hasNotReadyHost);
        }
        return true;
    }

    bool TContextDistributor::GetContextState(int hostId, const THashMap<int, int>& envId2Version, THashMap<int, TIntrusivePtr<TContextDataHolder>>* res) {
        CHROMIUM_TRACE_FUNCTION();
        TGuard<TMutex> g(Sync);
        res->clear();
        for (THashMap<int, int>::const_iterator z = envId2Version.begin(); z != envId2Version.end(); ++z) {
            int envId = z->first;
            int version = z->second;
            Y_ASSERT(EnvId2Info.find(envId) != EnvId2Info.end());
            const TFullCtxInfo& ctx = EnvId2Info[envId];
            const TCtxDataPart& dataPart = ctx.Data[hostId];
            TContextDataHolder* info = dataPart.Info.Get();
            if (!info || dataPart.Version != version)
                return false;
            (*res)[envId] = info;
        }
        return true;
    }

    const TVector<int>& TContextDistributor::GetComputer2HostId(int envId) {
        Y_ASSERT(EnvId2Info.find(envId) != EnvId2Info.end());
        return EnvId2Info[envId].Computer2HostId;
    }

    int TContextDistributor::GetHostIdCount(int envId) {
        Y_ASSERT(EnvId2Info.find(envId) != EnvId2Info.end());
        return EnvId2Info[envId].HostId2Computer.ysize();
    }

    void TContextDistributor::WaitDistribution() {
        CHROMIUM_TRACE_FUNCTION();
        while (!AtomicGet(DistributionIsComplete))
            ThreadYield();
        PAR_DEBUG_LOG << "Distribution complete" << Endl;
    }

    void TContextDistributor::WaitAllDistributionActivity() {
        CHROMIUM_TRACE_FUNCTION();
        while (AtomicGet(ActiveReqCount) > 0)
            ThreadYield();
    }

    //////////////////////////////////////////////////////////////////////////
}
