#include "par_util.h"
#include "par_jobreq.h"

#include <util/generic/guid.h>

#include <library/cpp/threading/local_executor/local_executor.h>

namespace NPar {
    TGUID TJobExecutor::TCallback::GetMasterQueryId() {
        return TGUID();
    }

    //////////////////////////////////////////////////////////////////////////
    enum ERROp {
        RR_MAP,
        RR_MAPREDUCE
    };

    class TExecRemoteReduce: public IMRCommandCompleteNotify {
        enum ECurrentCommand {
            FIRST_MAP,
            REMOTE_MAP
        };
        TIntrusivePtr<IDCResultNotify> DCNotify;
        int ReqId;
        TIntrusivePtr<IUserContext> Ctx;
        ERROp Op;
        TVector<char> FinalMapSerialized;
        ECurrentCommand State;

        void MRCommandComplete(bool isCanceled, TVector<TVector<char>>* res) override {
            CHROMIUM_TRACE_FUNCTION();
            if (isCanceled) {
                DCNotify->DistrCmdComplete(ReqId, nullptr);
                return;
            }
            if (State == FIRST_MAP) {
                TJobDescription fmJob;
                fmJob.SetCurrentOperation(FinalMapSerialized);
                for (int i = 0; i < res->ysize(); ++i)
                    fmJob.AddQuery(-1, &(*res)[i]);
                if (Op == RR_MAPREDUCE)
                    fmJob.MergeResults();

                State = REMOTE_MAP;
                Ctx->Run(&fmJob, this);

            } else if (State == REMOTE_MAP) {
                TVector<char> dcRes;
                if (Op == RR_MAP) {
                    SerializeToMem(&dcRes, *res);
                } else {
                    Y_ASSERT(res->ysize() == 1);
                    dcRes.swap((*res)[0]);
                }
                DCNotify->DistrCmdComplete(ReqId, &dcRes);
            } else
                Y_ASSERT(0);
        }
        TGUID GetMasterQueryId() override {
            return DCNotify->GetDistrCmdMasterQueryId();
        }
        bool MRNeedCheckCancel() const override {
            return true;
        }
        bool MRIsCmdNeeded() const override {
            return DCNotify->IsDistrCmdNeeded();
        }

    public:
        TExecRemoteReduce(IDCResultNotify* dcNotify, int reqId, IUserContext* ctx, ERROp op,
                          const TVector<char>& finalMapSerialized)
            : DCNotify(dcNotify)
            , ReqId(reqId)
            , Ctx(ctx)
            , Op(op)
            , FinalMapSerialized(finalMapSerialized)
            , State(FIRST_MAP)
        {
        }
    };

    class TRemoteReduce: public IDistrCmd {
        OBJECT_NOCOPY_METHODS(TRemoteReduce);
        TObj<IDistrCmd> FinalMap;
        ERROp Op;
        TVector<char> FinalMapSerialized;

        SAVELOAD_OVERRIDE(IDistrCmd, FinalMap, Op, FinalMapSerialized);
    public:
        TRemoteReduce() {
        }
        TRemoteReduce(IDistrCmd* finalMap, ERROp op)
            : FinalMap(finalMap)
            , Op(op)
        {
            CHROMIUM_TRACE_FUNCTION();
            SerializeToMem(&FinalMapSerialized, FinalMap);
        }
        void ExecAsync(IUserContext* ctx, int hostId, TVector<char>* params, IDCResultNotify* dcNotify, int reqId) const override {
            CHROMIUM_TRACE_FUNCTION();
            (void)hostId;
            TJobDescription mapJob;
            SerializeFromMem(params, mapJob);

            ctx->Run(&mapJob, new TExecRemoteReduce(dcNotify, reqId, ctx, Op, FinalMapSerialized));
        }
        void MergeAsync(TVector<TVector<char>>* src, IDCResultNotify* dcNotify, int reqId) const override {
            Y_ASSERT(Op == RR_MAPREDUCE);
            FinalMap->MergeAsync(src, dcNotify, reqId);
        }
    };

    const int N_MAX_PART_COUNT = 100;
    static void RemoteMapReduceImpl(TJobDescription* job, IDistrCmd* finalMap, ERROp op) {
        CHROMIUM_TRACE_FUNCTION();

        TObj<IDistrCmd> hold(finalMap);
        if (job->ExecList.empty())
            return;

        int jobCount = job->ExecList.ysize();
        int partCount = Min(jobCount, N_MAX_PART_COUNT);
        int jobPerPart = (jobCount + partCount - 1) / partCount;

        TVector<bool> hasData;
        hasData.resize(jobCount);

        TJobDescription newJob;
        {
            newJob.Cmds.resize(1);
            TObj<TRemoteReduce> rr = new TRemoteReduce(finalMap, op);
            SerializeToMem(&newJob.Cmds[0], rr);
        }
        newJob.ExecList.resize(partCount);

        for (int part = 0; part < partCount; ++part) {
            int startIdx = part * jobPerPart;
            int finishIdx = Min(startIdx + jobPerPart, jobCount);
            if (finishIdx <= startIdx) {
                newJob.ExecList.resize(part);
                break;
            }
            TJobDescription descr;
            TVector<int> resultMap;
            ProjectJob(&descr, startIdx, finishIdx - startIdx, &resultMap, &hasData, *job);
            int paramId = newJob.AddParam(&descr);
            TJobParams& jp = newJob.ExecList[part];
            jp = TJobParams(0, paramId, part, -1, TJobDescription::ANYWHERE_HOST_ID);
        }
        job->Swap(&newJob);
#ifdef _DEBUG
        for (int i = 0; i < hasData.ysize(); ++i)
            Y_ASSERT(hasData[i]);
#endif
    }

    void RemoteMap(TJobDescription* job, IDistrCmd* finalMap) {
        CHROMIUM_TRACE_FUNCTION();
        RemoteMapReduceImpl(job, finalMap, RR_MAP);
    }

    void RemoteMapReduce(TJobDescription* job, IDistrCmd* finalMap) {
        CHROMIUM_TRACE_FUNCTION();
        RemoteMapReduceImpl(job, finalMap, RR_MAPREDUCE);
        job->MergeResults();
    }

    //////////////////////////////////////////////////////////////////////////
    struct TExecRange {
        int Start, Finish;
    };
    class TRemoteRangeExecutor: public IDistrCmd {
        OBJECT_NOCOPY_METHODS(TRemoteRangeExecutor);

        class TSharedCmd: public TThrRefBase {
            ~TSharedCmd() override {
                IObjectBase::SetThreadCheckMode(false);
                FinalMap = nullptr;
                IObjectBase::SetThreadCheckMode(true);
            }

        public:
            TObj<IDistrCmd> FinalMap;

            TSharedCmd(IDistrCmd* cmd = nullptr)
                : FinalMap(cmd)
            {
            }
        };

        TIntrusivePtr<TSharedCmd> SharedCmd;

        int operator&(IBinSaver& f)override {
            if (f.IsReading()) {
                SharedCmd = new TSharedCmd();
                f.Add(2, &SharedCmd->FinalMap);
            } else {
                if (SharedCmd.Get())
                    f.Add(2, &SharedCmd->FinalMap);
                else {
                    TObj<IDistrCmd> finalMap;
                    f.Add(2, &finalMap);
                }
            }
            return 0;
        }

        class TExecutor: public ILocallyExecutable, public IDCResultNotify {
            TIntrusivePtr<IUserContext> Context;
            int HostId;
            TIntrusivePtr<IDCResultNotify> DCNotify;
            int ReqId;
            TIntrusivePtr<TSharedCmd> SharedCmd;
            TAtomic JobCount;
            TVector<TVector<char>> ResultsBuf;
            int Start, Finish;

            void DoneJob() {
                if (AtomicAdd(JobCount, -1) > 0)
                    return;
                Y_ASSERT(JobCount == 0);
                if (!DCNotify->IsDistrCmdNeeded())
                    return;
                if (ResultsBuf.ysize() > 1) {
                    SharedCmd->FinalMap->MergeAsync(&ResultsBuf, DCNotify.Get(), ReqId);
                } else {
                    if (ResultsBuf.empty())
                        DCNotify->DistrCmdComplete(ReqId, nullptr);
                    else
                        DCNotify->DistrCmdComplete(ReqId, &ResultsBuf[0]);
                }
            }
            void DistrCmdComplete(int reqId, TVector<char>* res) override {
                Y_ASSERT(reqId >= Start && reqId < Finish);
                if (res) {
                    Y_ASSERT(ResultsBuf[reqId - Start].empty());
                    res->swap(ResultsBuf[reqId - Start]);
                }
                DoneJob();
            }
            bool IsDistrCmdNeeded() override {
                return DCNotify->IsDistrCmdNeeded();
            }
            TGUID GetDistrCmdMasterQueryId() override {
                return DCNotify->GetDistrCmdMasterQueryId();
            }
            void LocalExec(int id) override {
                if (!DCNotify->IsDistrCmdNeeded()) {
                    DoneJob();
                    return;
                }
                TVector<char> buf;
                buf.resize(sizeof(id));
                *(int*)&buf[0] = id;
                SharedCmd->FinalMap->ExecAsync(Context.Get(), HostId, &buf, this, id);
            }
            ~TExecutor() override {
                Y_ASSERT(JobCount == 0);
            }

        public:
            TExecutor(IUserContext* context, int hostId,
                      TSharedCmd* sharedCmd, int start, int finish,
                      IDCResultNotify* dcNotify, int reqId)
                : Context(context)
                , HostId(hostId)
                , DCNotify(dcNotify)
                , ReqId(reqId)
                , SharedCmd(sharedCmd)
                , JobCount(finish - start + 1)
                , Start(start)
                , Finish(finish)
            {
                ResultsBuf.resize(finish - start);
            }
            void Launch() {
                LocalExecutor().ExecRange(this, Start, Finish, TLocalExecutor::WAIT_COMPLETE);
                DoneJob();
            }
        };

    public:
        TRemoteRangeExecutor() {
        }
        TRemoteRangeExecutor(IDistrCmd* finalMap) {
            SharedCmd = new TSharedCmd(finalMap);
        }
        void ExecAsync(IUserContext* ctx, int hostId, TVector<char>* params, IDCResultNotify* dcNotify, int reqId) const override {
            CHROMIUM_TRACE_FUNCTION();
            (void)hostId;
            TExecRange range;
            SerializeFromMem(params, range);

            TIntrusivePtr<TExecutor> exec;
            exec = new TExecutor(ctx, hostId, SharedCmd.Get(), range.Start, range.Finish, dcNotify, reqId);
            exec->Launch();
        }
        void MergeAsync(TVector<TVector<char>>* src, IDCResultNotify* dcNotify, int reqId) const override {
            CHROMIUM_TRACE_FUNCTION();
            SharedCmd->FinalMap->MergeAsync(src, dcNotify, reqId);
        }
    };

    void MakeRunOnRangeImpl(TJobDescription* job, int start, int finish, IDistrCmd* finalMap) {
        CHROMIUM_TRACE_FUNCTION();
        int jobCount = finish - start;
        if (jobCount < 2000) {
            // direct way is faster
            job->SetCurrentOperation(finalMap);
            for (int i = start; i < finish; ++i) {
                job->AddMap(i);
            }
        } else {
            int partCount = 1000;
            int jobPerPart = (jobCount + partCount - 1) / partCount;
            job->SetCurrentOperation(new TRemoteRangeExecutor(finalMap));
            for (int part = 0; part < partCount; ++part) {
                TExecRange range;
                range.Start = start + part * jobPerPart;
                range.Finish = start + Min((part + 1) * jobPerPart, jobCount);
                if (range.Finish > range.Start)
                    job->AddMap(range);
            }
        }
        job->MergeResults();
    }
}
using namespace NPar;
REGISTER_SAVELOAD_CLASS(0xD669C40, TRemoteReduce)
REGISTER_SAVELOAD_CLASS(0xD690C00, TRemoteRangeExecutor)
