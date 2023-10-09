#include "par_master.h"
#include "par_mr.h"
#include "par.h"
#include "par_context.h"

#include <util/random/shuffle.h>

namespace NPar {
    const int ENVID_ANYWHERE = 1;
    const int ENVID_EVERYWHERE = 2;

    class TGlobalUserContext: public IUserContext {
        TIntrusivePtr<TRemoteQueryProcessor> QueryProc;
        TIntrusivePtr<TContextDistributor> ContextMaster;

        THashMap<int, int> EnvId2Version;
        typedef THashMap<int, TIntrusivePtr<TContextDataHolder>> TContextHash;
        TVector<TContextHash> ContextData;
        int HostIdCount;

    public:
        TGlobalUserContext(TRemoteQueryProcessor* queryProc, TContextDistributor* contextMaster, int envId)
            : QueryProc(queryProc)
            , ContextMaster(contextMaster)
            , HostIdCount(-1)
        {
            CHROMIUM_TRACE_FUNCTION();
            ContextMaster->GetVersions(envId, &HostIdCount, &EnvId2Version);

            // everywhere is for remote ops only, so master should not report that it has data in this case
            if (envId != ENVID_EVERYWHERE) {
                ContextData.resize(HostIdCount);
                for (int hostId = 0; hostId < HostIdCount; ++hostId) {
                    if (!ContextMaster->GetContextState(hostId, EnvId2Version, &ContextData[hostId])) {
                        ContextData.clear();
                        break;
                    }
                }
            }
        }
        void Run(TJobDescription* descr, IMRCommandCompleteNotify* completeNotify) override;
        const THashMap<int, int>& GetEnvId2Version() const override {
            return EnvId2Version;
        }
        const IObjectBase* GetContextData(int envId, int hostId) override {
            Y_ASSERT(!ContextData.empty());
            const TContextHash& ctx = ContextData[hostId];
            TContextHash::const_iterator z = ctx.find(envId);
            Y_ASSERT(z != ctx.end());
            return z->second->Data;
        }
        bool HasHostIds(const THashMap<int, bool>& hostIdSet) override {
            (void)hostIdSet;
            return !ContextData.empty();
        }
        int GetHostIdCount() override {
            return HostIdCount;
        }
        EDataDistrState UpdateDataDistrState(TVector<TVector<int>>* hostId2Computer) override {
            bool hasNotReadyHost;
            bool ok = ContextMaster->GetReadyMask(EnvId2Version, hostId2Computer, &hasNotReadyHost);
            if (!ok)
                return DATA_UNAVAILABLE;
            if (hasNotReadyHost)
                return DATA_COPYING;
            return DATA_COMPLETE;
        }
        IUserContext* CreateLocalOnlyContext() override;

        TDataLocation RegisterData(int tblId, ui64 versionId, TVector<char>* p) override {
            TLocalDataBuffer* wb = ContextMaster->GetWriteBuffer();
            i64 id = wb->SetData(tblId, versionId, p);
            return TDataLocation(id, -1);
        }

        TDataLocation RegisterData(int tblId, ui64 versionId, TVector<TVector<char>>* p) override {
            TLocalDataBuffer* wb = ContextMaster->GetWriteBuffer();
            i64 id = wb->SetData(tblId, versionId, p);
            return TDataLocation(id, -1);
        }
        TDataLocation RegisterData(int tblId, ui64 versionId, const IObjectBase* obj) override {
            TLocalDataBuffer* wb = ContextMaster->GetWriteBuffer();
            i64 id = wb->SetObject(tblId, versionId, obj);
            return TDataLocation(id, -1);
        }
        void CollectData(const TVector<TDataLocation>& data, TVector<TVector<char>>* res) override {
            NPar::CollectData(data, res, ContextMaster->GetWriteBuffer(), QueryProc.Get());
        }
        void RunLocalOnly(TJobDescription* descr, IMRCommandCompleteNotify* completeNotify);
    };

    //////////////////////////////////////////////////////////////////////////
    class TRestrictedGlobalUserContext: public IUserContext {
        TIntrusivePtr<TGlobalUserContext> Parent;

    public:
        explicit TRestrictedGlobalUserContext(TGlobalUserContext* parent)
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
            EDataDistrState ret = Parent->UpdateDataDistrState(nullptr);
            if (ret == DATA_UNAVAILABLE)
                return DATA_UNAVAILABLE;
            return DATA_COMPLETE;
        }
        void Run(TJobDescription* descr, IMRCommandCompleteNotify* completeNotify) override {
            return Parent->RunLocalOnly(descr, completeNotify);
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

    IUserContext* TGlobalUserContext::CreateLocalOnlyContext() {
        return new TRestrictedGlobalUserContext(this);
    }

    void TGlobalUserContext::Run(TJobDescription* descr, IMRCommandCompleteNotify* completeNotify) {
        // copy is needed to handle UpdateDistrState() correctly
        // UpdateDistrState() is not thread safe and different contexts can have different HostId2Computer
        TIntrusivePtr<TGlobalUserContext> ctx = new TGlobalUserContext(*this);
        AsyncStartGroupRequest(descr, QueryProc.Get(), ctx.Get(), completeNotify);
    }

    void TGlobalUserContext::RunLocalOnly(TJobDescription* descr, IMRCommandCompleteNotify* completeNotify) {
        // copy is needed to handle UpdateDistrState() correctly
        // UpdateDistrState() is not thread safe and different contexts can have different HostId2Computer
        TGlobalUserContext* ctx = new TGlobalUserContext(*this);
        TIntrusivePtr<TRestrictedGlobalUserContext> userContext = new TRestrictedGlobalUserContext(ctx);

        AsyncStartGroupRequest(descr, QueryProc.Get(), userContext.Get(), completeNotify);
    }

    //////////////////////////////////////////////////////////////////////////
    class TEnvironment: public IEnvironment {
        OBJECT_NOCOPY_METHODS(TEnvironment);
        TIntrusivePtr<TRemoteQueryProcessor> QueryProc;
        TIntrusivePtr<TContextDistributor> ContextMaster;
        int EnvId;

    public:
        TEnvironment()
            : QueryProc(nullptr)
            , ContextMaster(nullptr)
            , EnvId(0)
        {
        }
        TEnvironment(TRemoteQueryProcessor* queryProc, TContextDistributor* contextMaster,
                     int envId, int parentEnvId, const TVector<int>& computer2HostId)
            : QueryProc(queryProc)
            , ContextMaster(contextMaster)
            , EnvId(envId)
        {
            ContextMaster->CreateNewContext(EnvId, parentEnvId, computer2HostId);
        }
        int GetEnvId() override {
            return EnvId;
        }
        int GetHostIdCount() override {
            return ContextMaster->GetHostIdCount(EnvId);
        }
        void SetContextData(int hostId, const IObjectBase* data, EKeepDataFlags keepContextRawData) override {
            ContextMaster->SetContextData(EnvId, hostId, data, keepContextRawData);
        }
        void SetContextData(const TVector<TDataLocation>& data, EKeepDataFlags keepContextRawData) override {
            TVector<int> compIds;
            TVector<i64> dataIds;
            int count = data.ysize();
            compIds.resize(count);
            dataIds.resize(count);
            for (int i = 0; i < count; ++i) {
                compIds[i] = data[i].CompId;
                dataIds[i] = data[i].DataId;
            }
            ContextMaster->SetContextData(EnvId, compIds, dataIds, keepContextRawData);
        }
        void DeleteContextRawData(int hostId, bool keepContextOnMaster) override {
            ContextMaster->DeleteContextRawData(EnvId, hostId, keepContextOnMaster);
        }
        void Run(TJobDescription* descr, IMRCommandCompleteNotify* completeNotify) override {
            TIntrusivePtr<TGlobalUserContext> ctx = new TGlobalUserContext(QueryProc.Get(), ContextMaster.Get(), EnvId);
            AsyncStartGroupRequest(descr, QueryProc.Get(), ctx.Get(), completeNotify);
        }
        IEnvironment* CreateChildEnvironment(int envId) override {
            return new TEnvironment(QueryProc.Get(), ContextMaster.Get(), envId, EnvId, ContextMaster->GetComputer2HostId(EnvId));
        }
        void CollectData(const TVector<TDataLocation>& data, TVector<TVector<char>>* res) override {
            NPar::CollectData(data, res, ContextMaster->GetWriteBuffer(), QueryProc.Get());
        }
    };

    //////////////////////////////////////////////////////////////////////////
    TMaster::TMaster(TRemoteQueryProcessor* queryProc, TContextDistributor* contextMaster)
        : QueryProc(queryProc)
        , ContextMaster(contextMaster)
    {
        int hostCount = contextMaster->GetSlaveCount();
        TVector<int> hostIds;
        hostIds.resize(hostCount, 0);
        Anywhere = CreateEnvironment(ENVID_ANYWHERE, hostIds);

        for (int i = 0; i < hostCount; ++i)
            hostIds[i] = i;
        Everywhere = CreateEnvironment(ENVID_EVERYWHERE, hostIds);
    }

    IEnvironment* TMaster::CreateEnvironment(int envId, const TVector<int>& hostIds) {
        return new TEnvironment(QueryProc.Get(), ContextMaster.Get(), envId, 0, hostIds);
    }

    TVector<int> TMaster::MakeHostIdMapping(int groupCount) {
        int hostCount = GetSlaveCount();
        if (QueryProc.Get() == nullptr)
            hostCount = groupCount;
        Y_ABORT_UNLESS(groupCount <= hostCount, "enough hosts to represent all groups");
        TVector<int> res;
        res.resize(hostCount);
        for (int i = 0; i < hostCount; ++i)
            res[i] = i % groupCount;
        Shuffle(res.begin(), res.end());
        return res;
    }

    int TMaster::GetSlaveCount() {
        return ContextMaster->GetSlaveCount();
    }
}
