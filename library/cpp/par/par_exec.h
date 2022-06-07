#pragma once

#include "par.h"
#include "par_jobreq.h"
#include "par_remote.h"
#include "par_log.h"

#include <library/cpp/binsaver/mem_io.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/is_in.h>
#include <util/generic/vector.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <library/cpp/deprecated/atomic/atomic_ops.h>
#include <util/system/yield.h>

namespace NPar {
    struct TDeserializedCmds {
        TIntrusivePtr<TJobRequest> JobRequest;
        TVector<TObj<IDistrCmd>> Cmds;

        void Check(int cmdId) {
            CHROMIUM_TRACE_FUNCTION();
            if (Cmds[cmdId] == nullptr) {
                TVector<char> cmdBuf = JobRequest->Descr.Cmds[cmdId];
                Y_ASSERT(!cmdBuf.empty());
                SerializeFromMem(&cmdBuf, Cmds[cmdId]);
            }
        }
        TDeserializedCmds(TJobRequest* jr)
            : JobRequest(jr)
        {
            Cmds.resize(jr->Descr.Cmds.ysize());
        }
        ~TDeserializedCmds() {
            IObjectBase::SetThreadCheckMode(false);
            Cmds.clear();
            IObjectBase::SetThreadCheckMode(true);
        }
    };

    class TReduceExec: public ILocallyExecutable, public IDCResultNotify {
        TIntrusivePtr<TJobRequest> JobRequest;
        TIntrusivePtr<IMRCommandCompleteNotify> CompleteNotify, CancelCheck;
        TDeserializedCmds Cmds;
        TVector<TVector<char>> ResultData;
        TVector<bool> ResultHasData;
        TVector<int> RemapTable;
        TAtomic ReduceReqCount;
        TAtomic CurrentState; // 0 = working, -1 = complete/canceled
        TGUID MasterGuid;

        int CountHasData(int* iPtr) {
            int& i = *iPtr;
            int firstId = i;
            int totalJobCount = JobRequest->Descr.ExecList.ysize();
            int reduceId = JobRequest->Descr.ExecList[firstId].ReduceId;
            Y_ASSERT(ResultHasData[firstId]);
            int hasDataCount = 1;
            for (++i; i < totalJobCount && JobRequest->Descr.ExecList[i].ReduceId == reduceId; ++i)
                hasDataCount += (int)ResultHasData[i];
            return hasDataCount;
        }
        void DoneReduceTask() {
            if (AtomicAdd(ReduceReqCount, -1) != 0)
                return;
            if (AtomicCas(&CurrentState, -1, 0)) {
                if (!RemapTable.empty()) {
                    int reduceResultCount = RemapTable.ysize();
                    for (int i = 0; i < reduceResultCount; ++i)
                        ResultData[i].swap(ResultData[RemapTable[i]]);
                    ResultData.resize(reduceResultCount);
                }

                PAR_DEBUG_LOG << "Done reduce tasks" << Endl;
                CompleteNotify->MRCommandComplete(false, &ResultData);
                CompleteNotify = nullptr;
            }
        }
        void DistrCmdComplete(int reqId, TVector<char>* res) override {
            if (res)
                res->swap(ResultData[reqId]);
            else
                Cancel();
            DoneReduceTask();
        }
        bool IsDistrCmdNeeded() override {
            return NeedResult();
        }
        TGUID GetDistrCmdMasterQueryId() override {
            return MasterGuid;
        }
        void Cancel() {
            if (AtomicCas(&CurrentState, -1, 0)) {
                if (CompleteNotify.Get()) {
                    CompleteNotify->MRCommandComplete(true, nullptr);
                    CompleteNotify = nullptr;
                } else
                    Y_ASSERT(0 && "Notify ptr is lost");
            }
        }
        bool NeedResult() {
            if (AtomicGet(CurrentState) != 0)
                return false;
            if (CancelCheck.Get()) {
                if (!CancelCheck->MRIsCmdNeeded()) {
                    Cancel();
                    return true;
                }
            }
            return true;
        }
        void LocalExec(int id) override {
            if (!NeedResult())
                return;
            TJobParams& firstOp = JobRequest->Descr.ExecList[id];
            int cmdId = firstOp.CmdId;
            int lastId = id;
            int hasDataCount = CountHasData(&lastId);
            Y_ASSERT(hasDataCount > 1);
            if (id == 0 && hasDataCount == JobRequest->Descr.ExecList.ysize()) {
                // merging everything
                Cmds.Cmds[cmdId]->MergeAsync(&ResultData, this, id);
            } else {
                TVector<TVector<char>> src;
                src.resize(hasDataCount);
                int dst = 0;
                for (int i = id; i < lastId; ++i) {
                    if (ResultHasData[i])
                        src[dst++].swap(ResultData[i]);
                }
                Y_ASSERT(dst == hasDataCount);

                Cmds.Cmds[cmdId]->MergeAsync(&src, this, id);
            }
        }
        void StartReduce() {
            ReduceReqCount = 1;
            int totalJobCount = JobRequest->Descr.ExecList.ysize();
            RemapTable.resize(totalJobCount);
            int reduceResultCount = 0;
            for (int i = 0; i < totalJobCount;) {
                int firstId = i;
                int hasDataCount = CountHasData(&i);

                if (hasDataCount > 1) {
                    PAR_DEBUG_LOG << "Launch reduce task " << firstId << Endl;
                    int cmdId = JobRequest->Descr.ExecList[firstId].CmdId;
                    Cmds.Check(cmdId);
                    AtomicAdd(ReduceReqCount, 1);
                    if (JobRequest->IsLowPriority)
                        LocalExecutor().Exec(this, firstId, TLocalExecutor::MED_PRIORITY);
                    else
                        LocalExecutor().Exec(this, firstId, 0);
                    RemapTable[reduceResultCount++] = firstId;
                } else
                    RemapTable[reduceResultCount++] = firstId;
            }
            if (reduceResultCount == totalJobCount)
                RemapTable.resize(0);
            else
                RemapTable.resize(reduceResultCount);
            DoneReduceTask();
        }
        TReduceExec(
            TJobRequest* jobRequest,
            IMRCommandCompleteNotify* completeNotify,
            TVector<TVector<char>>* resultData,
            TVector<bool>* resultHasData)
            : JobRequest(jobRequest)
            , CompleteNotify(completeNotify)
            , Cmds(jobRequest)
            , ReduceReqCount(0)
            , CurrentState(0)
        {
            resultData->swap(ResultData);
            resultHasData->swap(ResultHasData);
            if (completeNotify->MRNeedCheckCancel())
                CancelCheck = completeNotify;
            MasterGuid = completeNotify->GetMasterQueryId();
        }

    public:
        static void Launch(TJobRequest* jobRequest,
                           IMRCommandCompleteNotify* completeNotify,
                           TVector<TVector<char>>* results,
                           TVector<bool>* resultHasData) {
            TIntrusivePtr<TReduceExec> op = new TReduceExec(jobRequest, completeNotify, results, resultHasData);
            op->StartReduce();
        }
    };

    bool RescheduleJobRequest(TJobRequest* src, const TVector<ui16>& parentExecPlan, int localCompId, int ignoreCompId);

    class TMRCommandExec: public IRemoteQueryResponseNotify, public ILocallyExecutable, public IMRCommandCompleteNotify, public IDCResultNotify {
        struct TMapResultBuffer {
            TVector<TVector<char>> ResultData;
            TVector<bool> ResultHasData;
        };
        struct TRemoteMapInfo {
            TVector<int> ResultMap;
            TIntrusivePtr<TJobRequest> JobRequest;
            int DstHost;
        };

        TIntrusivePtr<TRemoteQueryProcessor> QueryProc;
        TIntrusivePtr<TJobRequest> JobRequest;
        TIntrusivePtr<IMRCommandCompleteNotify> CompleteNotify, CancelCheck;
        TDeserializedCmds Cmds;

        // map job result buffers
        TMapResultBuffer RemoteMapBuf, LocalMapBuf;
        TMapResultBuffer* volatile MapResult;

        TAtomic LocalMapReqCount, RemoteMapReqCount;
        TIntrusivePtr<IUserContext> UserContext, LocalUserContext;

        // remote result buffers
        TVector<TRemoteMapInfo> MapParts;
        TVector<bool> PartCompleted;
        TVector<int> MapJob2PartId;
        TAtomic RemoteJobCount;
        int LocalPartId;

        TLockFreeStack<TGUID> AllReqList;
        TQueryCancelCallback<TMRCommandExec> CancelCallback;
        TGUID MasterGuid;

        void LaunchOps(int localCompId) {
            CHROMIUM_TRACE_FUNCTION();

            int mapJobCount = JobRequest->Descr.ExecList.ysize();
            LocalMapBuf.ResultData.resize(mapJobCount);
            LocalMapBuf.ResultHasData.resize(mapJobCount, false);
            MapJob2PartId.resize(mapJobCount, -1);

            bool localOnly = JobRequest->ExecPlan.empty();
            if (JobRequest->ExecPlan.ysize() == 1 && JobRequest->ExecPlan[0] == localCompId)
                localOnly = true;

            if (localOnly) {
                // local host only, launch local ops
                LocalUserContext = UserContext;

                LocalMapReqCount = 1;
                for (int i = 0; i < mapJobCount; ++i) {
                    Y_ASSERT(JobRequest->Descr.ExecList[i].CompId == localCompId);
                    int cmdId = JobRequest->Descr.ExecList[i].CmdId;
                    Cmds.Check(cmdId);
                }
                AtomicAdd(LocalMapReqCount, mapJobCount);
                int execRangeFlags = JobRequest->IsLowPriority ? TLocalExecutor::MED_PRIORITY : 0;
                LocalExecutor().ExecRange(this, 0, mapJobCount, execRangeFlags);
                DoneLocalMapTask();
            } else {
                LocalUserContext = UserContext->CreateLocalOnlyContext();

                RemoteMapReqCount = 1;

                // get subtasks
                TVector<TVector<ui16>> subTasks;
                GenerateSubtasks(JobRequest->ExecPlan, &subTasks);
                int partCount = subTasks.ysize();
                Y_ASSERT(partCount > 0); // one remote part is possible

                RemoteMapBuf.ResultData.resize(mapJobCount);
                RemoteMapBuf.ResultHasData.resize(mapJobCount, false);
                PartCompleted.resize(partCount, false);

                // split job on subtasks
                MapParts.resize(partCount);
                for (int i = 0; i < partCount; ++i) {
                    TJobRequest* jr = new TJobRequest;
                    TRemoteMapInfo* part = &MapParts[i];
                    ProjectJob(&jr->Descr, i, &part->ResultMap,
                               &RemoteMapBuf.ResultHasData, &MapJob2PartId,
                               JobRequest->Descr,
                               subTasks[i]);
                    jr->ExecPlan = subTasks[i];
                    jr->HostId2Computer = JobRequest->HostId2Computer;
                    jr->EnvId2Version = JobRequest->EnvId2Version;
                    jr->IsLowPriority = JobRequest->IsLowPriority;

                    part->JobRequest = jr;
                    part->DstHost = SelectRandomHost(jr->ExecPlan);
                }

                const int FAKE_REMOTE_JOB_COUNT = 100; // needed to prevent rescheduling of not started job
                RemoteMapReqCount += partCount;
                AtomicSet(RemoteJobCount, partCount + FAKE_REMOTE_JOB_COUNT);

                // launch part ops
                LocalPartId = -1;
                for (int i = 0; i < partCount; ++i) {
                    TRemoteMapInfo* part = &MapParts[i];
                    TJobRequest* jr = part->JobRequest.Get();
                    if (IsIn(jr->ExecPlan, localCompId)) {
                        Y_ASSERT(LocalPartId == -1);
                        part->DstHost = localCompId;
                        LocalPartId = i;
                        TMRCommandExec::Launch(jr, QueryProc.Get(), localCompId, UserContext.Get(), this);
                        AtomicAdd(RemoteJobCount, -1); // this one is not remote
                    } else {
                        TVector<char> buf;
                        SerializeToMem(&buf, *jr);
                        const char* mrCmd = JobRequest->IsLowPriority ? "mr_low" : "mr";
                        TGUID req = QueryProc->SendQuery(part->DstHost, mrCmd, &buf, this, i);
                        RegisterRemoteQuery(req);
                    }
                }
                //Y_ASSERT(LocalPartId >= 0); // can happen
                AtomicAdd(RemoteJobCount, -FAKE_REMOTE_JOB_COUNT);
                DoneRemoteMapTask();
                if (LocalPartId == -1)
                    TryToExecAllMapsLocally();
            }
            CancelCallback.Attach(this, QueryProc.Get(), MasterGuid);
        }
        void CancelAllRemoteQueries() {
            TGUID req;
            while (AllReqList.Dequeue(&req)) {
                PAR_DEBUG_LOG << "Cancel req " << GetGuidAsString(req) << Endl;
                QueryProc->CancelQuery(req);
            }
        }
        void RegisterRemoteQuery(const TGUID& req) {
            AllReqList.Enqueue(req);
            if (!NeedResult())
                CancelAllRemoteQueries();
        }
        void TryToExecAllMapsLocally() {
            CHROMIUM_TRACE_FUNCTION();

            // check if we have hostId data
            THashMap<int, bool> hostIdSet;
            int mapJobCount = JobRequest->Descr.ExecList.ysize();
            for (int i = 0; i < mapJobCount; ++i) {
                int hostId = JobRequest->Descr.ExecList[i].HostId;
                if (hostId == TJobDescription::ANYWHERE_HOST_ID) {
                    int hostIdCount = LocalUserContext->GetHostIdCount();
                    for (int h = 0; h < hostIdCount; ++h)
                        hostIdSet[h];
                    break;
                } else {
                    hostIdSet[hostId];
                }
            }

            if (LocalUserContext->HasHostIds(hostIdSet) && AtomicGet(MapResult) == nullptr) {
                Y_ASSERT(LocalMapReqCount == 0);
                LocalMapReqCount = 1;
                for (int i = 0; i < mapJobCount; ++i) {
                    int cmdId = JobRequest->Descr.ExecList[i].CmdId;
                    Cmds.Check(cmdId);
                }
                AtomicAdd(LocalMapReqCount, mapJobCount);
                LocalExecutor().ExecRange(this, 0, mapJobCount, TLocalExecutor::LOW_PRIORITY);
                DoneLocalMapTask();
            }
        }
        void Cancel() {
            CHROMIUM_TRACE_FUNCTION();
            // can cancel op if reduce has not started yet
            if (AtomicCas(&MapResult, (TMapResultBuffer*)-1, (TMapResultBuffer*)nullptr)) {
                PAR_DEBUG_LOG << "MRExec canceled" << Endl;
                CancelAllRemoteQueries();
                if (CompleteNotify.Get()) {
                    CompleteNotify->MRCommandComplete(true, nullptr);
                    CompleteNotify = nullptr;
                } else
                    Y_ASSERT(0 && "Notify ptr is lost");
            }
        }
        void ReschedulePartRequest(int partId) {
            CHROMIUM_TRACE_FUNCTION();
            TRemoteMapInfo* part = &MapParts[partId];

            PAR_DEBUG_LOG << "Try to reschedule part " << partId << Endl;
            TJobRequest* src = part->JobRequest.Get();
            QueryProc->IncLastCount(part->DstHost);

            // construct brand new plan
            int ignoreCompId = part->DstHost;
            int localCompId = QueryProc->GetCompId();
            if (!RescheduleJobRequest(src, JobRequest->ExecPlan, localCompId, ignoreCompId))
                return;

            int execCompId = SelectRandomHost(src->ExecPlan);
            Y_ASSERT(execCompId != ignoreCompId && "ignoreCompId is supposed to be excluded from execution?");

            TVector<char> buf;
            SerializeToMem(&buf, *src);
            TGUID req = QueryProc->SendQuery(execCompId, "mr_low", &buf, this, partId);
            RegisterRemoteQuery(req);
            PAR_DEBUG_LOG << "Part " << partId << " reasked" << Endl;
        }
        void CopyRemoteTaskResults(int partId, TVector<TVector<char>>* result) {
            if (PartCompleted[partId])
                return;
            TRemoteMapInfo* part = &MapParts[partId];
            const TVector<int>& resMap = part->ResultMap;
            Y_ASSERT(resMap.ysize() == result->ysize());
            for (int k = 0; k < result->ysize(); ++k) {
                int dstIndex = resMap[k];
                RemoteMapBuf.ResultData[dstIndex].swap((*result)[k]);
            }
            PartCompleted[partId] = true;
        }
        void GotResponse(int id, TVector<char>* response) override {
            CHROMIUM_TRACE_FUNCTION();

            if (!NeedResult())
                return;
            if (PartCompleted[id]) {
                PAR_DEBUG_LOG << "Ignoring duplicate result for part " << id << Endl;
                return;
            }

            TJobRequestReply res;
            SerializeFromMem(response, res);
            if (res.IsCanceled) {
                Cancel();
                return;
            }
            CopyRemoteTaskResults(id, &res.Result);
            if (AtomicAdd(RemoteJobCount, -1) == 1) {
                // last job left, try to reschedule it, probably new copy will finish faster
                int partId = -1;
                for (int i = 0; i < MapParts.ysize(); ++i) {
                    if (i != LocalPartId && PartCompleted[i] == false) {
                        Y_ASSERT(partId == -1);
                        partId = i;
                    }
                }
                //Y_ASSERT(partId != -1); // possible since RemoteJobCount is modified from LaunchOps() in different thread
                if (partId >= 0 && AtomicGet(MapResult) == nullptr)
                    ReschedulePartRequest(partId);
            }
            DoneRemoteMapTask();
        }
        void MRCommandComplete(bool isCanceled, TVector<TVector<char>>* res) override {
            if (isCanceled) {
                Cancel();
                return;
            }
            if (!NeedResult())
                return;
            CopyRemoteTaskResults(LocalPartId, res);
            if (AtomicGet(RemoteJobCount) > 0) {
                // completed local part, lets try to execute all the rest map jobs locally
                // "local-remote balance"
                TryToExecAllMapsLocally();
            }
            DoneRemoteMapTask();
        }
        TGUID GetMasterQueryId() override {
            return MasterGuid;
        }
        void DistrCmdComplete(int id, TVector<char>* res) override {
            if (res) {
                TVector<char>& dataBuf = LocalMapBuf.ResultData[id];
                res->swap(dataBuf);
                LocalMapBuf.ResultHasData[id] = true;
                DoneLocalMapTask();
            } else {
                Cancel();
            }
        }
        bool IsDistrCmdNeeded() override {
            return NeedResult();
        }
        TGUID GetDistrCmdMasterQueryId() override {
            return MasterGuid;
        }
        void LocalExec(int id) override {
            // do map task locally
            const TJobParams& params = JobRequest->Descr.ExecList[id];
            if (NeedResult() && AtomicGet(MapResult) == nullptr) {
                int partId = MapJob2PartId[id];
                if (partId == -1 || !PartCompleted[partId]) {
                    TVector<char>& dataBuf = LocalMapBuf.ResultData[id];
                    JobRequest->Descr.GetParam(params.ParamId, &dataBuf);
                    Cmds.Cmds[params.CmdId]->ExecAsync(LocalUserContext.Get(), params.HostId, &dataBuf, this, id);
                } else
                    DoneLocalMapTask();
            }
        }
        void StartReduce() {
            auto* const mapResult = AtomicGet(MapResult);
            TReduceExec::Launch(JobRequest.Get(), CompleteNotify.Get(), &mapResult->ResultData, &mapResult->ResultHasData);
        }
        void DoneRemoteMapTask() {
            if (AtomicAdd(RemoteMapReqCount, -1) != 0)
                return;
            if (AtomicCas(&MapResult, &RemoteMapBuf, (TMapResultBuffer*)nullptr)) {
                CancelAllRemoteQueries();
                PAR_DEBUG_LOG << "Remote maps completed first" << Endl;
                AtomicAdd(RemoteMapWins, 1);
                StartReduce();
            }
        }
        Y_FORCE_INLINE void DoneLocalMapTask() {
            if (AtomicAdd(LocalMapReqCount, -1) != 0)
                return;
            DoneLocalMapTaskImpl();
        }
        void DoneLocalMapTaskImpl() {
            if (AtomicCas(&MapResult, &LocalMapBuf, (TMapResultBuffer*)nullptr)) {
                CancelAllRemoteQueries();

                // so local map exec finished first, copy complete remote results to local buf
                // we have to copy since we did not execute locally remotely complete map jobs

                // no synchronization required since MapJob2PartId & PartCompleted are not resized
                // during remote execution and CopyRemoteTaskResults() can not be called from different
                // threads for the same partId (there is only one local part and all remote parts are
                // completed by calling GotResponse() from QueryProc thread)
                TVector<bool> partCompletedSnapshot = PartCompleted;
                int mapJobCount = LocalMapBuf.ResultData.ysize();
                for (int i = 0; i < mapJobCount; ++i) {
                    int partId = MapJob2PartId[i];
                    if (partId == -1 || !partCompletedSnapshot[partId])
                        continue;
                    LocalMapBuf.ResultData[i].swap(RemoteMapBuf.ResultData[i]);
                    LocalMapBuf.ResultHasData[i] = RemoteMapBuf.ResultHasData[i];
                }
                if (!MapParts.empty()) {
                    PAR_DEBUG_LOG << "Local maps completed first" << Endl;
                    AtomicAdd(LocalMapWins, 1); // count only wins over competing remote execs
                } else {
                    PAR_DEBUG_LOG << "Local maps done" << Endl;
                }
                StartReduce();
            }
        }
        bool NeedResult() {
            if (AtomicGet(MapResult) != nullptr)
                return false;
            if (CancelCheck.Get()) {
                if (!CancelCheck->MRIsCmdNeeded()) {
                    Cancel();
                    return false;
                }
            }
            return true;
        }
        TMRCommandExec(
            TJobRequest* jobRequest,
            TRemoteQueryProcessor* queryProc,
            IUserContext* userContext,
            IMRCommandCompleteNotify* completeNotify)
            : QueryProc(queryProc)
            , JobRequest(jobRequest)
            , CompleteNotify(completeNotify)
            , Cmds(jobRequest)
            , MapResult(nullptr)
            , LocalMapReqCount(0)
            , RemoteMapReqCount(0)
            , UserContext(userContext)
            , RemoteJobCount(0)
            , LocalPartId(-1)
        {
            if (completeNotify->MRNeedCheckCancel())
                CancelCheck = completeNotify;
            MasterGuid = completeNotify->GetMasterQueryId();
        }
        ~TMRCommandExec() override {
            CancelCallback.Detach();
        }

    public:
        void OnQueryCancel() {
            Cancel();
        }

    public:
        static TAtomic LocalMapWins, RemoteMapWins;

        static void Launch(TJobRequest* jobRequest,
                           TRemoteQueryProcessor* queryProc,
                           int localCompId,
                           IUserContext* userContext,
                           IMRCommandCompleteNotify* completeNotify) {
            TIntrusivePtr<TMRCommandExec> op;
            op = new TMRCommandExec(jobRequest, queryProc, userContext, completeNotify);
            op->LaunchOps(localCompId);
        }
    };

    void LaunchLocalJobRequest(TJobRequest* jr, int localCompId, IUserContext* userContext, IMRCommandCompleteNotify* mrNotify);
    bool LaunchJobRequest(TJobRequest* jr, TRemoteQueryProcessor* queryProc, IUserContext* userContext, IMRCommandCompleteNotify* mrNotify);

    class TSplitMRExec: public TThrRefBase {
        struct TBlock {
            int StartIdx, BlockLen;

            TBlock()
                : StartIdx(0)
                , BlockLen(0)
            {
            }
            TBlock(int startIdx, int blockLen)
                : StartIdx(startIdx)
                , BlockLen(blockLen)
            {
            }
        };

        TIntrusivePtr<TRemoteQueryProcessor> QueryProc;
        TIntrusivePtr<TJobRequest> JobRequest;
        TIntrusivePtr<IMRCommandCompleteNotify> CompleteNotify;
        TIntrusivePtr<IUserContext> UserContext;
        TVector<TVector<char>> ResultData;
        TVector<bool> ResultHasData;
        TVector<TBlock> Blocks;
        TAtomic CurrentBlockId;
        void* IsCanceledFlag;
        TAtomic ActiveOpCount;
        TGUID MasterGuid;

        enum {
            FIRST_BLOCK_LEN = 2
        };

        class TBlockCallback: public IMRCommandCompleteNotify {
            TIntrusivePtr<TSplitMRExec> Parent;
            TVector<int> ResultMap;

            void MRCommandComplete(bool isCanceled, TVector<TVector<char>>* res) override {
                if (isCanceled)
                    Parent->Cancel();
                else {
                    // copy results
                    Y_ASSERT(ResultMap.ysize() == res->ysize());
                    int count = ResultMap.ysize();
                    for (int i = 0; i < count; ++i)
                        Parent->ResultData[ResultMap[i]].swap((*res)[i]);
                    AtomicAdd(Parent->ActiveOpCount, -1);

                    Parent->StartNextBlock();
                }
            }
            TGUID GetMasterQueryId() override {
                return Parent->MasterGuid;
            }

        public:
            TBlockCallback(TSplitMRExec* parent, TVector<int>* resultMap)
                : Parent(parent)
            {
                resultMap->swap(ResultMap);
            }
        };

        void Cancel() {
            if (AtomicCas(&IsCanceledFlag, (void*)this, (void*)nullptr)) {
                PAR_DEBUG_LOG << "SplitMRExec canceled" << Endl;
                if (CompleteNotify.Get())
                    CompleteNotify->MRCommandComplete(true, nullptr);
                CompleteNotify = nullptr;
            }
        }
        void StartNextBlock() {
            IUserContext::EDataDistrState ds = UserContext->UpdateDataDistrState(nullptr);
            if (ds == IUserContext::DATA_UNAVAILABLE) {
                Cancel();
                return;
            }
            int blockId, nextBlockId;
            AtomicAdd(ActiveOpCount, 1);
            if (ds == IUserContext::DATA_COMPLETE) {
                int blockCount = Blocks.ysize();
                blockId = AtomicAdd(CurrentBlockId, blockCount) - blockCount;
                nextBlockId = Blocks.ysize();
            } else {
                blockId = AtomicAdd(CurrentBlockId, 1) - 1;
                nextBlockId = blockId + 1;
            }
            if (blockId >= Blocks.ysize()) {
                // no new ops to launch
                // check if the work is complete
                AtomicAdd(ActiveOpCount, -1); // cancel this block
                if (AtomicAdd(ActiveOpCount, 0) == 0 && AtomicCas(&IsCanceledFlag, (void*)this, (void*)nullptr))
                    TReduceExec::Launch(JobRequest.Get(), CompleteNotify.Get(), &ResultData, &ResultHasData);
            } else {
                int startIdx = Blocks[blockId].StartIdx;
                int blockLen = 0;
                for (int i = blockId; i < nextBlockId; ++i)
                    blockLen += Blocks[i].BlockLen;
                LaunchBlockRequest(startIdx, blockLen);
            }
        }
        void LaunchBlockRequest(int startIdx, int blockLen) {
            TIntrusivePtr<TJobRequest> jr = new TJobRequest;
            TVector<int> resultMap;
            ProjectJob(&jr->Descr, startIdx, blockLen, &resultMap, &ResultHasData, JobRequest->Descr);
            jr->EnvId2Version = JobRequest->EnvId2Version;
            jr->IsLowPriority = JobRequest->IsLowPriority;

            // get new distribution state, start next block in new conditions
            TIntrusivePtr<TBlockCallback> bc(new TBlockCallback(this, &resultMap));
            for (;;) {
                IUserContext::EDataDistrState ds = UserContext->UpdateDataDistrState(&jr->HostId2Computer);
                if (ds == IUserContext::DATA_UNAVAILABLE) {
                    Cancel();
                    return;
                }

                if (IsCanceledFlag)
                    return;

                if (LaunchJobRequest(jr.Get(), QueryProc.Get(), UserContext.Get(), bc.Get()))
                    return;

                // can not execute have to wait somehow
                ThreadYield();
            }
        }
        TSplitMRExec(TJobRequest* jobRequest,
                     TRemoteQueryProcessor* queryProc, IMRCommandCompleteNotify* mrNotify,
                     IUserContext* userContext)
            : QueryProc(queryProc)
            , JobRequest(jobRequest)
            , CompleteNotify(mrNotify)
            , UserContext(userContext)
            , CurrentBlockId(0)
            , IsCanceledFlag(nullptr)
            , ActiveOpCount(0)
        {
            int mapJobCount = jobRequest->Descr.ExecList.ysize();
            ResultData.resize(mapJobCount);
            ResultHasData.resize(mapJobCount);
            MasterGuid = CompleteNotify->GetMasterQueryId();

            // fill Blocks[]
            for (int startIdx = 0, blockLen = FIRST_BLOCK_LEN; startIdx < mapJobCount; startIdx += blockLen) {
                int cmdsLeft = mapJobCount - startIdx;
                blockLen = Min(cmdsLeft, blockLen * 2);
                if (blockLen * 2 > cmdsLeft)
                    blockLen = cmdsLeft; // allow faster then 2x grow on last iteration to avoid small last block
                Blocks.push_back(TBlock(startIdx, blockLen));
            }
        }
        static void CancelLaunch(IMRCommandCompleteNotify* mrNotify, IUserContext* userContext) {
            // bad luck, looks like data is outdated
            PAR_DEBUG_LOG << "Failed to launch SplitMRExec" << Endl;
            TIntrusivePtr<IMRCommandCompleteNotify> holdNotify(mrNotify);
            TIntrusivePtr<IUserContext> holdContext(userContext);
            mrNotify->MRCommandComplete(true, nullptr);
        }

    public:
        static void Launch(TJobRequest* jobRequest,
                           TRemoteQueryProcessor* queryProc, IUserContext* userContext,
                           IMRCommandCompleteNotify* mrNotify) {
            for (;;) {
                IUserContext::EDataDistrState ds = userContext->UpdateDataDistrState(&jobRequest->HostId2Computer);
                if (ds == IUserContext::DATA_UNAVAILABLE) {
                    CancelLaunch(mrNotify, userContext);
                    return;
                }
                if (ds != IUserContext::DATA_COMPLETE && jobRequest->Descr.ExecList.ysize() > FIRST_BLOCK_LEN * 2) {
                    TIntrusivePtr<TSplitMRExec> op = new TSplitMRExec(jobRequest, queryProc, mrNotify, userContext);
                    op->StartNextBlock();
                    op->StartNextBlock();
                    return;
                } else {
                    bool chk = LaunchJobRequest(jobRequest, queryProc, userContext, mrNotify);
                    if (chk)
                        return;
                    // can not execute have to wait somehow
                    ThreadYield();
                }
            }
        }
    };
}
