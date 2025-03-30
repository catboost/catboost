#include "compression.h"
#include "par_remote.h"
#include "par_log.h"
#include "par_host_stats.h"

#include <library/cpp/binsaver/mem_io.h>
#include <library/cpp/binsaver/util_stream_io.h>
#include <library/cpp/chromium_trace/interface.h>

#include <util/random/random.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <library/cpp/deprecated/atomic/atomic_ops.h>
#include <util/system/event.h>
#include <util/system/fs.h>
#include <util/system/hostname.h>
#include <util/system/hp_timer.h>
#include <util/system/spinlock.h>

namespace NPar {
    const char* DELAY_MATRIX_NAME = "delay_matrix.bin";
    const float PING_ITERATION_TIME = 30;

    struct TDelayData {
        TArray2D<TVector<float>> DelayMatrixData;
        TVector<TNetworkAddress> BaseSearcherAddrs;

        SAVELOAD(DelayMatrixData, BaseSearcherAddrs);
    };

    struct THostInitData {
        int CompId;
        TNetworkAddress MasterAddress;
        TVector<TNetworkAddress> BaseSearcherAddrs;

        SAVELOAD(CompId, MasterAddress, BaseSearcherAddrs);

        THostInitData()
            : CompId(-1)
        {
        }
    };

    struct TPingResult {
        int CompId;
        float Delay;

        TPingResult()
            : CompId(-1)
            , Delay(0)
        {
        }
        TPingResult(int compId, float delay)
            : CompId(compId)
            , Delay(delay)
        {
        }
    };

    struct TAllPingResults {
        TVector<TPingResult> PingResults;

        SAVELOAD(PingResults);
    };

    class TMetaRequester: public IRemoteQueryResponseNotify {
        TIntrusivePtr<TRemoteQueryProcessor> Meta;
        TVector<TVector<char>> Results;
        TAtomic QueryCount = 0;
        TAtomic ResultCount = 0;
        TSystemEvent Ready;

    public:
        TMetaRequester(TRemoteQueryProcessor* meta)
            : Meta(meta)
        {
        }
        void AddQuery(int compId, const char* query, TVector<char>* data) {
            Meta->SendQuery(compId, query, data, this, AtomicGet(QueryCount));
            AtomicIncrement(QueryCount);
        }
        void GotResponse(int id, TVector<char>* response) override {
            if (id >= Results.ysize())
                Results.resize(id + 1);
            Results[id].swap(*response);
            AtomicIncrement(ResultCount);
            if (AtomicGet(ResultCount) == AtomicGet(QueryCount))
                Ready.Signal();
        }
        void GetResults(TVector<TVector<char>>* res) {
            Ready.Reset();
            if (AtomicGet(QueryCount) != AtomicGet(ResultCount))
                Ready.Wait();
            if (res)
                res->swap(Results);
            Results.resize(0);
            AtomicSet(QueryCount, 0);
            AtomicSet(ResultCount, 0);
        }
    };
    //////////////////////////////////////////////////////////////////////////
    TGUID TRemoteQueryProcessor::SendQuery(int compId, const char* query, TVector<char>* cmdData, IRemoteQueryResponseNotify* proc, int resId) {
        CHROMIUM_TRACE_FUNCTION();

        TGUID reqId;
        CreateGuid(&reqId);
        RequestsData.EmplaceValue(reqId, new TQueryResultDst(proc, query, compId, resId));
        Requester->SendRequest(reqId, GetCompAddress(compId), query, cmdData);

        return reqId;
    }

    void TRemoteQueryProcessor::CancelQuery(const TGUID& reqId) {
        CHROMIUM_TRACE_FUNCTION();
        PAR_DEBUG_LOG << "At " << Requester->GetHostAndPort() << " cancel query: " << GetGuidAsString(reqId) << Endl;
        Requester->CancelRequest(reqId);
    }

    void TRemoteQueryProcessor::RegisterCmdType(const char* sz, ICmdProcessor* p) {
        Y_ASSERT(Requester.Get() == nullptr && "function is not thread safe, should be called before RunThread()");
        CmdProcessors[sz] = p;
    }

    void TRemoteQueryProcessor::SendReply(const TGUID& reqId, TVector<char>* response) {
        CHROMIUM_TRACE_FUNCTION();
        static TTiming& queryExecutionTiming = TParHostStats::GetTiming(ETimingTag::QueryExecutionTime);
        TIntrusivePtr<TQueryResultDst> queryResultDst;
        if (IncomingRequestsData.ExtractValueIfPresent(reqId, queryResultDst)) {
            queryExecutionTiming += (TInstant::Now() - queryResultDst->QueryCreationTime).SecondsFloat();
            Requester->SendResponse(reqId, response);
        }
    }

    void TRemoteQueryProcessor::RegisterCallback(const TGUID& reqId, IRemoteQueryCancelNotify* notify) {
        CHROMIUM_TRACE_FUNCTION();

        Y_ABORT_UNLESS(!reqId.IsEmpty());
        PAR_DEBUG_LOG << "At " << Requester->GetHostAndPort() << " Register cancel callback for request: " << GetGuidAsString(reqId) << Endl;
        TIntrusivePtr<TQueryResultDst> queryResultDst;
        auto functor = [notify](TIntrusivePtr<TQueryResultDst>& theQueryResultDst) {
            theQueryResultDst->CallbackVector.push_back(notify);
        };
        if (!IncomingRequestsData.LockedValueModify(reqId, functor)) {
            PAR_DEBUG_LOG << "At " << Requester->GetHostAndPort() << " No such request in map, probably already sent reply" << Endl;
        }
    }

    void TRemoteQueryProcessor::RunMaster(const TVector<TNetworkAddress>& baseSearcherAddrs, unsigned short masterListenPort) {
        CHROMIUM_TRACE_FUNCTION();
        Y_ASSERT(Requester.Get() == nullptr);
        BaseSearcherAddrs = baseSearcherAddrs;
        LastCounts.resize(BaseSearcherAddrs.ysize(), TAtomicWrap(0));

        SetRequester(CreateRequester(
            masterListenPort,
            [this](const TGUID& canceledReq) { QueryCancelCallback(canceledReq); },
            [this](TAutoPtr<TNetworkRequest>& nlReq) { IncomingQueryCallback(nlReq); },
            [this](TAutoPtr<TNetworkResponse> response) { ReplyCallback(response); }));

        MasterAddress = TNetworkAddress(HostName(), Requester->GetListenPort());
        DEBUG_LOG << "Listening on port: " << Requester->GetListenPort() << Endl;
        // init base searchers
        DEBUG_LOG << "Init base searchers" << Endl;
        TIntrusivePtr<TMetaRequester> mr(new TMetaRequester(this));
        int searcherCount = GetSlaveCount();
        for (int i = 0; i < searcherCount; ++i) {
            THostInitData initData;
            initData.CompId = i;
            initData.MasterAddress = MasterAddress;
            initData.BaseSearcherAddrs = BaseSearcherAddrs;
            TVector<char> cmdData;
            SerializeToMem(&cmdData, initData);
            mr->AddQuery(i, "init", &cmdData);
        }
        mr->GetResults(nullptr);

        // run_ping
        TArray2D<TVector<float>> delayMatrixData;

        // try to reuse previous ping stats
        bool needRunPing = true;
        if (NFs::Exists(DELAY_MATRIX_NAME)) {
            TDelayData data;
            SerializeFromFile(DELAY_MATRIX_NAME, data);
            if (data.BaseSearcherAddrs == BaseSearcherAddrs) {
                DEBUG_LOG << "Reusing ping times from " << DELAY_MATRIX_NAME << Endl;
                needRunPing = false;
                delayMatrixData.Swap(data.DelayMatrixData);
            }
        }

        if (needRunPing) {
            DEBUG_LOG << "Run ping times collection" << Endl;
            int pingCount = 0, iterCount = 0;
            while (pingCount < searcherCount * searcherCount * 10 && iterCount < 10) {
                for (int i = 0; i < searcherCount; ++i) {
                    mr->AddQuery(i, "run_ping", nullptr);
                }
                TVector<TVector<char>> res;
                mr->GetResults(&res);

                delayMatrixData.SetSizes(searcherCount, searcherCount);
                Y_ASSERT(res.ysize() == searcherCount);
                for (int srcCompId = 0; srcCompId < searcherCount; ++srcCompId) {
                    TAllPingResults stats;
                    SerializeFromMem(&res[srcCompId], stats);
                    for (int z = 0; z < stats.PingResults.ysize(); ++z) {
                        ++pingCount;
                        const TPingResult& pingRes = stats.PingResults[z];
                        delayMatrixData[srcCompId][pingRes.CompId].push_back(pingRes.Delay);
                    }
                }
                ++iterCount;
                DEBUG_LOG << "Ping iteration " << iterCount << " finished" << Endl;
            }
            DEBUG_LOG << "~" << pingCount / ((float)searcherCount * searcherCount) << " pings collected" << Endl;

            // store ping stats
            TDelayData data;
            data.BaseSearcherAddrs = BaseSearcherAddrs;
            data.DelayMatrixData = delayMatrixData;
            SerializeToFile(DELAY_MATRIX_NAME, data);
        }

        TDistributionTreesData distrTrees;
        BuildDistributionTree(&distrTrees, delayMatrixData);
        UniversalExecPlan = distrTrees.TreeBranch6;

        DEBUG_LOG << "Sending exec plan" << Endl;
        for (int i = 0; i < searcherCount; ++i) {
            TVector<char> cmdData;
            SerializeToMem(&cmdData, UniversalExecPlan);
            mr->AddQuery(i, "exec_plan", &cmdData);
        }
        mr->GetResults(nullptr);

        DEBUG_LOG << "Meta init complete, " << searcherCount << " hosts" << Endl;
    }

    void TRemoteQueryProcessor::RunSlave(int port) {
        Y_ASSERT(Requester.Get() == nullptr);

        RegisterCmdType("init", InitCmd.Get());
        RegisterCmdType("ping", PingCmd.Get());
        RegisterCmdType("run_ping", RunPingCmd.Get());
        RegisterCmdType("exec_plan", SetExecPlanCmd.Get());
        RegisterCmdType("stop", StopSlaveCmd.Get());
        RegisterCmdType("gather_stats", GatherStatsCmd.Get());

        SetRequester(CreateRequester(
            port,
            [this](const TGUID& canceledReq) { QueryCancelCallback(canceledReq); },
            [this](TAutoPtr<TNetworkRequest>& nlReq) { IncomingQueryCallback(nlReq); },
            [this](TAutoPtr<TNetworkResponse> response) { ReplyCallback(response); }));
        Y_ABORT_UNLESS(Requester.Get());
        SlaveFinish.Reset();
        SlaveFinish.Wait();
    }

    void TRemoteQueryProcessor::StopSlaves() {
        TIntrusivePtr<TMetaRequester> mr(new TMetaRequester(this));
        DEBUG_LOG << "Stopping slaves" << Endl;
        fflush(nullptr);
        int searcherCount = GetSlaveCount();
        TVector<bool> chkStopRes;
        chkStopRes.resize(searcherCount, false);
        for (bool canStop = false; !canStop;) {
            for (int i = 0; i < searcherCount; ++i) {
                if (!chkStopRes[i]) {
                    mr->AddQuery(i, "check_stop", nullptr);
                }
            }
            TVector<TVector<char>> res;
            mr->GetResults(&res);
            canStop = true;
            for (int i = 0; i < res.ysize(); ++i) {
                if (chkStopRes[i]) {
                    continue;
                }
                bool resVal = (res[i][0] != 0);
                chkStopRes[i] = resVal;
                canStop &= resVal;
            }
        }
        DEBUG_LOG << "Gathering debug stats from slaves" << Endl;
        for (int i = 0; i < searcherCount; ++i) {
            mr->AddQuery(i, "gather_stats", nullptr);
        }
        auto& masterTimings = Singleton<TParHostStats>()->ParTimings;
        TVector<TVector<char>> res;
        mr->GetResults(&res);
        Y_ABORT_UNLESS(
            res.ysize() == searcherCount,
            "res.ysize()=%d, searcherCount=%d",
            res.ysize(), searcherCount);
        for (int i = 0; i < searcherCount; ++i) {
            TParHostStats tmpStats;
            SerializeFromMem(&res.at(i), tmpStats);
            for (size_t j = 0; j < static_cast<size_t>(ETimingTag::TimingsCount); ++j) {
                *masterTimings.Timings[j] += *tmpStats.ParTimings.Timings[j];
            }
        }
        DEBUG_LOG << "Cumulative timings: " << Endl;
        for (size_t i = 0; i < static_cast<size_t>(ETimingTag::TimingsCount); ++i) {
            Cout << static_cast<ETimingTag>(i) << ": " << static_cast<double>(*masterTimings.Timings[i]) << Endl;
        }
        for (int i = 0; i < searcherCount; ++i) {
            mr->AddQuery(i, "stop", nullptr);
        }
        mr->GetResults(nullptr);
    }

    void TRemoteQueryProcessor::IncLastCount(int compId) {
        Y_ASSERT(compId >= 0 && compId < LastCounts.ysize());
        if (compId >= 0 && compId < LastCounts.ysize())
            AtomicAdd(LastCounts[compId].Counter, 1);
    }

    void TRemoteQueryProcessor::TInitCmd::NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) {
        CHROMIUM_TRACE_FUNCTION();

        if (!p->BaseSearcherAddrs.empty()) {
            ERROR_LOG << "Init called twice" << Endl;
            abort();
        }
        THostInitData initData;
        SerializeFromMem(&req->Data, initData);
        p->CompId = initData.CompId;
        p->MasterAddress = initData.MasterAddress;
        p->BaseSearcherAddrs = initData.BaseSearcherAddrs;
        p->LastCounts.resize(p->BaseSearcherAddrs.ysize(), TAtomicWrap(0));

        p->SendReply(req->ReqId, nullptr);
        PAR_DEBUG_LOG << "CompId " << p->CompId << " initialized" << Endl;
    }

    void TRemoteQueryProcessor::TPingCmd::NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) {
        p->SendReply(req->ReqId, nullptr);
    }

    void TRemoteQueryProcessor::TRunPingCmd::NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) {
        Y_ASSERT(QueryProc.Get() == nullptr && "single run_ping is allowed in parallel");
        QueryProc = p;
        BaseSearcherAddrs = p->BaseSearcherAddrs;
        ReqId = req->ReqId;
        LocalExecutor().Exec(this, 0, 0);
    }

    void TRemoteQueryProcessor::TRunPingCmd::LocalExec(int) {
        CHROMIUM_TRACE_FUNCTION();
        auto* requester = QueryProc->Requester.Get();
        TAllPingResults stats;
        float totalTime = 0;
        for (int i = 0; i < BaseSearcherAddrs.ysize() * 10; ++i) {
            int compId = RandomNumber(BaseSearcherAddrs.size());
            TVector<char> pingPkt;
            pingPkt.resize(100 * 1000);
            const TNetworkAddress& addr = BaseSearcherAddrs[compId];
            NHPTimer::STime tStart;
            NHPTimer::GetTime(&tStart);
            TAutoPtr<TNetworkResponse> answer = requester->Request(addr, "ping", &pingPkt);
            float ping = (float)NHPTimer::GetTimePassed(&tStart);
            stats.PingResults.push_back(TPingResult(compId, ping));

            totalTime += ping;
            if (totalTime > PING_ITERATION_TIME)
                break;
        }
        TVector<char> responseData;
        SerializeToMem(&responseData, stats);
        QueryProc->SendReply(ReqId, &responseData);
        QueryProc = nullptr;
    }
    void NPar::TRemoteQueryProcessor::TGatherStatsCmd::NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) {
        TVector<char> tmp;
        SerializeToMem(&tmp, *Singleton<TParHostStats>());
        DEBUG_LOG << "Sending par stats" << Endl;
        p->SendReply(req->ReqId, &tmp);
    }

    void TRemoteQueryProcessor::TSetExecPlanCmd::NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) {
        CHROMIUM_TRACE_FUNCTION();
        SerializeFromMem(&req->Data, p->UniversalExecPlan);
        p->SendReply(req->ReqId, nullptr);
    }

    void TRemoteQueryProcessor::TStopSlaveCmd::NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) {
        CHROMIUM_TRACE_FUNCTION();
        p->SendReply(req->ReqId, nullptr);
        // TODO: refactor this ugly master crash/hang fix on stop
        // NEH uses tcp-channels to send queries and replies, and it doesn't track query status
        // so it can wait for reply from stopped process
        Sleep(TDuration::Seconds(1));
        p->SlaveFinish.Signal();
    }

    void TRemoteQueryProcessor::SetRequester(TIntrusivePtr<IRequester> requester) noexcept {
        Requester = std::move(requester);
        RequesterIsSet = true;
    }

    void TRemoteQueryProcessor::WaitUntilRequesterIsSet() noexcept {
        if (!RequesterIsSet) {
            TSpinWait sw;

            while (!RequesterIsSet) {
                sw.Sleep();
            }
        }
    }

    void TRemoteQueryProcessor::QueryCancelCallback(const TGUID& canceledReq) {
        WaitUntilRequesterIsSet();
        QueryCancelCallbackImpl(canceledReq);
    }

    void TRemoteQueryProcessor::QueryCancelCallbackImpl(const TGUID& canceledReq) {
        CHROMIUM_TRACE_FUNCTION();
        NetworkEventsQueue.Enqueue(TNetworkEvent(canceledReq));
        NetworkEvent.Signal();
    }

    void TRemoteQueryProcessor::IncomingQueryCallback(TAutoPtr<TNetworkRequest>& nlReq) {
        WaitUntilRequesterIsSet();
        IncomingQueryCallbackImpl(nlReq);
    }

    void TRemoteQueryProcessor::IncomingQueryCallbackImpl(TAutoPtr<TNetworkRequest>& nlReq) {
        CHROMIUM_TRACE_FUNCTION();

        PAR_DEBUG_LOG << "At " << Requester->GetHostAndPort() << " Got request " << nlReq->Url << " " << GetGuidAsString(nlReq->ReqId) << Endl;
        NetworkEventsQueue.Enqueue(TNetworkEvent(nlReq.Release()));
        NetworkEvent.Signal();
    }

    void TRemoteQueryProcessor::ReplyCallback(TAutoPtr<TNetworkResponse> response) {
        WaitUntilRequesterIsSet();
        ReplyCallbackImpl(response);
    }

    void TRemoteQueryProcessor::ReplyCallbackImpl(TAutoPtr<TNetworkResponse> response) {
        CHROMIUM_TRACE_FUNCTION();
        PAR_DEBUG_LOG << "At " << Requester->GetHostAndPort() << " Got reply for redId " << GetGuidAsString(response->ReqId) << Endl;
        NetworkEventsQueue.Enqueue(TNetworkEvent(response.Release()));
        NetworkEvent.Signal();
    }

    TRemoteQueryProcessor::TRemoteQueryProcessor()
        : InitCmd(new TInitCmd)
        , PingCmd(new TPingCmd)
        , RunPingCmd(new TRunPingCmd)
        , SetExecPlanCmd(new TSetExecPlanCmd)
        , StopSlaveCmd(new TStopSlaveCmd)
        , GatherStatsCmd(new TGatherStatsCmd)
    {
        MetaThread = SystemThreadFactory()->Run([this]() {
            MetaThreadFunction();
        });
    }

    TRemoteQueryProcessor::~TRemoteQueryProcessor() {
        if (DoRun) {
            Stop();
        }
    }

    void TRemoteQueryProcessor::Stop() {
        if (DoRun) {
            DoRun = false;
            MetaThread->Join();
        }
    }

    void TRemoteQueryProcessor::MetaThreadFunction() {
        while (DoRun) {
            TNetworkEvent netEvent;
            while (NetworkEventsQueue.Dequeue(&netEvent)) {
                if (netEvent.EventType == TNetworkEvent::EType::IcomingQueryCancel) {
                    TIntrusivePtr<TQueryResultDst> queryInfoPtr;
                    if (!IncomingRequestsData.ExtractValueIfPresent(netEvent.ReqId, queryInfoPtr)) {
                        continue;
                    }
                    for (auto& callback : queryInfoPtr->CallbackVector) {
                        callback->OnCancel();
                    }
                } else if (netEvent.EventType == TNetworkEvent::EType::IncomingQuery) {
                    if (!CmdProcessors.contains(netEvent.Request->Url)) {
                        if (netEvent.Request->Url == "check_stop") {
                            TVector<char> tmp;
                            tmp.push_back(RequestsData.Empty());
                            Requester->SendResponse(netEvent.Request->ReqId, &tmp);
                        } else {
                            Y_ASSERT(0);
                        }
                    } else {
                        IncomingRequestsData.EmplaceValue(netEvent.Request->ReqId, new TQueryResultDst);
                        CmdProcessors.at(netEvent.Request->Url)->NewRequest(this, netEvent.Request.Get());
                    }
                } else if (netEvent.EventType == TNetworkEvent::EType::ReplyReceived) {
                    TIntrusivePtr<TQueryResultDst> queryInfoPtr;
                    if (!RequestsData.ExtractValueIfPresent(netEvent.Response->ReqId, queryInfoPtr)) {
                        continue;
                    }

                    if (netEvent.Response->Status == TNetworkResponse::EStatus::Canceled) {
                        PAR_DEBUG_LOG << "At " << Requester->GetHostAndPort() << " Query " << GetGuidAsString(netEvent.Response->ReqId) << " cancelled" << Endl;
                    } else if (netEvent.Response->Status == TNetworkResponse::EStatus::Ok) {
                        static TTiming& queryFullTime = TParHostStats::GetTiming(ETimingTag::QueryFullTime);
                        queryFullTime += (TInstant::Now() - queryInfoPtr->QueryCreationTime).SecondsFloat();
                        queryInfoPtr->Proc->GotResponse(queryInfoPtr->Id, &netEvent.Response->Data);
                    } else {
                        Y_ABORT();
                    }
                }
            }
            if (!NetworkEventsQueue.IsEmpty()) {
                continue;
            }
            NetworkEvent.WaitT(TDuration::MilliSeconds(500));
        }
    }
}
