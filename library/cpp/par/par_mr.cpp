#include "par_mr.h"
#include "par_exec.h"
#include "par_context.h"

#include <library/cpp/binsaver/mem_io.h>

namespace NPar {
    class TUserContext: public IUserContext {
        TIntrusivePtr<TRemoteQueryProcessor> QueryProc;
        TIntrusivePtr<TJobRequest> AllJobs;
        typedef THashMap<int, TIntrusivePtr<TContextDataHolder>> TContextHash;
        TContextHash ContextData;
        TIntrusivePtr<TLocalDataBuffer> WriteBuffer;

        void Run(TJobDescription* descr, IMRCommandCompleteNotify* completeNotify) override;

    public:
        TUserContext(TRemoteQueryProcessor* queryProc, TJobRequest* allJobs,
                     THashMap<int, TIntrusivePtr<TContextDataHolder>>* contextData,
                     TLocalDataBuffer* writeBuffer)
            : QueryProc(queryProc)
            , AllJobs(allJobs)
            , WriteBuffer(writeBuffer)
        {
            ContextData.swap(*contextData);
        }
        const THashMap<int, int>& GetEnvId2Version() const override {
            return AllJobs->EnvId2Version;
        }
        const IObjectBase* GetContextData(int envId, int hostId) override {
            TContextHash::const_iterator z = ContextData.find(envId);
            if (z == ContextData.end()) {
                Y_ASSERT(0 && "not available envId");
                return nullptr;
            }
            TContextDataHolder* ctx = z->second.Get();
            int thisCompId = QueryProc->GetCompId();
            int thisHostId = ctx->Computer2HostId[thisCompId];
            if (thisHostId != hostId) {
                Y_ASSERT(0 && "data with this hostId is not available on this computer");
                return nullptr;
            }
            return ctx->Data;
        }
        bool HasHostIds(const THashMap<int, bool>& hostIdSet) override {
            if (ContextData.empty() || hostIdSet.empty())
                return true;
            // all contexts should have same hostid set so we can pick any
            TContextDataHolder* ctx = ContextData.begin()->second.Get();
            for (TContextHash::const_iterator z = ContextData.begin(); z != ContextData.end(); ++z) {
                Y_ASSERT(z->second.Get()->Computer2HostId == ctx->Computer2HostId);
            }
            int thisCompId = QueryProc->GetCompId();
            int thisHostId = ctx->Computer2HostId[thisCompId];
            if (hostIdSet.size() == 1 && hostIdSet.find(thisHostId) != hostIdSet.end())
                return true;
            return false;
        }
        int GetHostIdCount() override {
            return AllJobs->HostId2Computer.ysize();
        }
        EDataDistrState UpdateDataDistrState(TVector<TVector<int>>* hostId2Computer) override {
            // this is remote context and it knows nothing about data distribution in progress
            if (hostId2Computer)
                *hostId2Computer = AllJobs->HostId2Computer;
            return DATA_COMPLETE;
        }
        IUserContext* CreateLocalOnlyContext() override;
        TDataLocation RegisterData(int tblId, ui64 versionId, TVector<char>* p) override {
            i64 id = WriteBuffer->SetData(tblId, versionId, p);
            return TDataLocation(id, QueryProc->GetCompId());
        }
        TDataLocation RegisterData(int tblId, ui64 versionId, TVector<TVector<char>>* p) override {
            i64 id = WriteBuffer->SetData(tblId, versionId, p);
            return TDataLocation(id, QueryProc->GetCompId());
        }
        TDataLocation RegisterData(int tblId, ui64 versionId, const IObjectBase* obj) override {
            i64 id = WriteBuffer->SetObject(tblId, versionId, obj);
            return TDataLocation(id, QueryProc->GetCompId());
        }
        void CollectData(const TVector<TDataLocation>& data, TVector<TVector<char>>* res) override {
            NPar::CollectData(data, res, WriteBuffer.Get(), QueryProc.Get());
        }
        TRemoteQueryProcessor* GetQueryProc() const {
            return QueryProc.Get();
        }
    };

    class TLocalUserContext: public IUserContext {
        TIntrusivePtr<TUserContext> Parent;

        void Run(TJobDescription* descr, IMRCommandCompleteNotify* completeNotify) override;

    public:
        TLocalUserContext(TUserContext* parent)
            : Parent(parent)
        {
        }
        const THashMap<int, int>& GetEnvId2Version() const override {
            return Parent->GetEnvId2Version();
        }
        const IObjectBase* GetContextData(int envId, int hostId) override {
            return Parent->GetContextData(envId, hostId);
        }
        bool HasHostIds(const THashMap<int, bool>& hostIdSet) override {
            return Parent->HasHostIds(hostIdSet);
        }
        int GetHostIdCount() override {
            return Parent->GetHostIdCount();
        }
        EDataDistrState UpdateDataDistrState(TVector<TVector<int>>* hostId2Computer) override {
            if (hostId2Computer) {
                hostId2Computer->resize(0);
                hostId2Computer->resize(Parent->GetHostIdCount());
            }
            return Parent->UpdateDataDistrState(nullptr);
        }
        IUserContext* CreateLocalOnlyContext() override {
            return this;
        }
        TDataLocation RegisterData(int tblId, ui64 versionId, TVector<char>* p) override {
            return Parent->RegisterData(tblId, versionId, p);
        }
        TDataLocation RegisterData(int tblId, ui64 versionId, TVector<TVector<char>>* p) override {
            return Parent->RegisterData(tblId, versionId, p);
        }
        TDataLocation RegisterData(int tblId, ui64 versionId, const IObjectBase* obj) override {
            return Parent->RegisterData(tblId, versionId, obj);
        }
        void CollectData(const TVector<TDataLocation>& data, TVector<TVector<char>>* res) override {
            Parent->CollectData(data, res);
        }
    };

    class TRemoteMRCommandExec: public IMRCommandCompleteNotify, public TNonCopyable {
        TIntrusivePtr<TRemoteQueryProcessor> QueryProc;
        TIntrusivePtr<TJobRequest> AllJobs;
        TGUID ReqId;
        THashMap<int, TIntrusivePtr<TContextDataHolder>> ContextData;
        TIntrusivePtr<TUserContext> UserContext;
        TQueryCancelCallback<TRemoteMRCommandExec> CancelCallback;
        bool IsQueryCanceled;

    public:
        TRemoteMRCommandExec(TNetworkRequest* req, TRemoteQueryProcessor* queryProc)
            : QueryProc(queryProc)
            , IsQueryCanceled(false)
        {
            CHROMIUM_TRACE_FUNCTION();
            ReqId = req->ReqId;
            AllJobs = new TJobRequest;
            SerializeFromMem(&req->Data, *AllJobs);
            AllJobs->IsLowPriority = req->Url == "mr_low";
            CancelCallback.Attach(this, queryProc, ReqId);
        }
        ~TRemoteMRCommandExec() override {
            CancelCallback.Detach();
        }
        void LaunchRequest(TContextReplica* context) {
            if (context->GetContextState(AllJobs->EnvId2Version, &ContextData)) {
                UserContext = new TUserContext(QueryProc.Get(), AllJobs.Get(), &ContextData, context->GetWriteBuffer());
                int localCompId = QueryProc->GetCompId();
                TMRCommandExec::Launch(AllJobs.Get(), QueryProc.Get(), localCompId, UserContext.Get(), this);
            } else {
                PAR_DEBUG_LOG << "Get context state failed" << Endl;
                MRCommandComplete(true, nullptr);
            }
        }
        void MRCommandComplete(bool isCanceled, TVector<TVector<char>>* res) override {
            CHROMIUM_TRACE_FUNCTION();
            TJobRequestReply reply;
            reply.IsCanceled = isCanceled;
            if (res)
                res->swap(reply.Result);
            Y_ASSERT(isCanceled || !reply.Result.empty());
            TVector<char> buf;
            SerializeToMem(&buf, reply);
            QueryProc->SendReply(ReqId, &buf);
        }
        TGUID GetMasterQueryId() override {
            return ReqId;
        }
        bool MRNeedCheckCancel() const override {
            return true;
        }
        bool MRIsCmdNeeded() const override {
            return !IsQueryCanceled;
        }
        void OnQueryCancel() {
            IsQueryCanceled = true;
        }
    };

    void TMRCmdsProcessor::NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) {
        CHROMIUM_TRACE_FUNCTION();
        TIntrusivePtr<TRemoteMRCommandExec> exec = new TRemoteMRCommandExec(req, p);
        exec->LaunchRequest(Context.Get());
    }

    //////////////////////////////////////////////////////////////////////////
    void TUserContext::Run(TJobDescription* descr, IMRCommandCompleteNotify* completeNotify) {
        AsyncStartGroupRequest(descr, QueryProc.Get(), this, completeNotify);
    }

    IUserContext* TUserContext::CreateLocalOnlyContext() {
        return new TLocalUserContext(this);
    }

    void TLocalUserContext::Run(TJobDescription* descr, IMRCommandCompleteNotify* completeNotify) {
        AsyncStartGroupRequest(descr, Parent->GetQueryProc(), this, completeNotify);
    }

    //////////////////////////////////////////////////////////////////////////
    void AsyncStartGroupRequest(TJobDescription* descr,
                                TRemoteQueryProcessor* queryProc, IUserContext* userContext,
                                IMRCommandCompleteNotify* mrNotify) {
        TIntrusivePtr<TJobRequest> jr = new TJobRequest;
        jr->Descr.Cmds.swap(descr->Cmds);
        jr->Descr.ParamsData.swap(descr->ParamsData);
        jr->Descr.ParamsPtr.swap(descr->ParamsPtr);
        jr->EnvId2Version = userContext->GetEnvId2Version();
        jr->IsLowPriority = false;

        int hostIdCount = userContext->GetHostIdCount();

        for (int i = 0; i < descr->ExecList.ysize(); ++i) {
            TJobParams jp = descr->ExecList[i];
            if (jp.HostId == TJobDescription::MAP_HOST_ID) {
                // its a map request, assign to several hosts
                for (int hostId = 0; hostId < hostIdCount; ++hostId) {
                    jp.HostId = hostId;
                    jr->Descr.ExecList.push_back(jp);
                }
            } else {
                // direct request
                Y_ABORT_UNLESS(
                    (jp.HostId >= 0 && jp.HostId < hostIdCount) || jp.HostId == TJobDescription::ANYWHERE_HOST_ID,
                    "jp.HostId=%d, hostIdCount=%d",
                    jp.HostId, hostIdCount);
                jr->Descr.ExecList.push_back(jp);
            }
        }

        if (queryProc)
            TSplitMRExec::Launch(jr.Get(), queryProc, userContext, mrNotify);
        else
            LaunchLocalJobRequest(jr.Get(), -1, userContext, mrNotify);
    }
}
