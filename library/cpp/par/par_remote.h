#pragma once

#include "distr_tree.h"
#include "par_locked_hash.h"
#include "par_network.h"

#include <library/cpp/threading/atomic/bool.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/event.h>
#include <util/system/spinlock.h>
#include <util/thread/lfqueue.h>
#include <util/thread/factory.h>

namespace NPar {
    class TRemoteQueryProcessor;
    struct TNetworkEvent {
        enum class EType {
            IncomingQuery,
            IcomingQueryCancel,
            ReplyReceived
        };
        EType EventType;
        TGUID ReqId;
        TAtomicSharedPtr<TNetworkRequest> Request;
        TAtomicSharedPtr<TNetworkResponse> Response;
        TNetworkEvent() = default;
        explicit TNetworkEvent(TNetworkRequest* request)
            : EventType(EType::IncomingQuery)
            , ReqId(request->ReqId)
            , Request(request)
        {
        }
        explicit TNetworkEvent(TNetworkResponse* response)
            : EventType(EType::ReplyReceived)
            , ReqId(response->ReqId)
            , Response(response)
        {
        }
        explicit TNetworkEvent(const TGUID& reqId)
            : EventType(EType::IcomingQueryCancel)
            , ReqId(reqId)
        {
        }
    };

    struct ICmdProcessor : virtual public TThrRefBase {
        virtual void NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) = 0;
    };

    struct IRemoteQueryResponseNotify : virtual public TThrRefBase {
        // calls of this function are guaranteed to come from single thread
        virtual void GotResponse(int id, TVector<char>* response) = 0;
    };

    struct IRemoteQueryCancelNotify: public TThrRefBase {
        virtual void OnCancel() = 0;
    };

    class TRemoteQueryProcessor: public TThrRefBase {
        struct TQueryResultDst: public TThrRefBase {
            TIntrusivePtr<IRemoteQueryResponseNotify> Proc = nullptr;
            TString ReqName;
            int DstCompId = 0;
            int Id = 0;
            TInstant QueryCreationTime;
            TQueryResultDst()
                : QueryCreationTime(TInstant::Now())
            {
            }
            TQueryResultDst(IRemoteQueryResponseNotify* proc, const char* reqName, int dstCompId, int id)
                : Proc(proc)
                , ReqName(reqName)
                , DstCompId(dstCompId)
                , Id(id)
                , QueryCreationTime(TInstant::Now())
            {
            }
            using TQueryCancelCallbackVector = TVector<TIntrusivePtr<IRemoteQueryCancelNotify>>;
            TQueryCancelCallbackVector CallbackVector;
        };

        class TInitCmd: public ICmdProcessor {
            void NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) override;
        };
        class TPingCmd: public ICmdProcessor {
            void NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) override;
        };
        class TGatherStatsCmd: public ICmdProcessor {
            void NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) override;
        };
        class TRunPingCmd: public ICmdProcessor, public ILocallyExecutable {
            TGUID ReqId;
            TVector<TNetworkAddress> BaseSearcherAddrs;
            TIntrusivePtr<TRemoteQueryProcessor> QueryProc; // to prevent cycle should be used only when run ping is executing
        public:
            void NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) override;
            void LocalExec(int id) override;
        };
        class TSetExecPlanCmd: public ICmdProcessor {
            void NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) override;
        };
        class TStopSlaveCmd: public ICmdProcessor {
            void NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) override;
        };

        struct TAtomicWrap {
            TAtomic Counter;

            TAtomicWrap()
                : Counter(0)
            {
            }
            TAtomicWrap(int x)
                : Counter(x)
            {
            }
        };

        int CompId = -1;
        TVector<TNetworkAddress> BaseSearcherAddrs;
        TNetworkAddress MasterAddress;
        TVector<ui16> UniversalExecPlan;
        THashMap<TString, TIntrusivePtr<ICmdProcessor>> CmdProcessors;
        TIntrusivePtr<TInitCmd> InitCmd;
        TIntrusivePtr<TPingCmd> PingCmd;
        TIntrusivePtr<TRunPingCmd> RunPingCmd;
        TIntrusivePtr<TSetExecPlanCmd> SetExecPlanCmd;
        TIntrusivePtr<TStopSlaveCmd> StopSlaveCmd;
        TIntrusivePtr<TGatherStatsCmd> GatherStatsCmd;

        TSystemEvent SlaveFinish;
        TVector<TAtomicWrap> LastCounts;

        using TRequestHash = TSpinLockedKeyValueStorage<TGUID, TIntrusivePtr<TQueryResultDst>, TGUIDHash>;
        TRequestHash RequestsData;
        TRequestHash IncomingRequestsData;
        TLockFreeQueue<TNetworkEvent> NetworkEventsQueue;
        THolder<IThreadFactory::IThread> MetaThread;
        NAtomic::TBool DoRun = true;
        TAutoEvent NetworkEvent;

        NAtomic::TBool RequesterIsSet = false;
        TIntrusivePtr<IRequester> Requester;

    private:
        const TNetworkAddress& GetCompAddress(int compId) {
            return compId < 0 ? MasterAddress : BaseSearcherAddrs[compId];
        }

        void MetaThreadFunction();

        void SetRequester(TIntrusivePtr<IRequester> requester) noexcept;
        void WaitUntilRequesterIsSet() noexcept;

        void QueryCancelCallback(const TGUID& canceledReq);
        void QueryCancelCallbackImpl(const TGUID& canceledReq);

        void IncomingQueryCallback(TAutoPtr<TNetworkRequest>& nlReq);
        void IncomingQueryCallbackImpl(TAutoPtr<TNetworkRequest>& nlReq);

        void ReplyCallback(TAutoPtr<TNetworkResponse> response);
        void ReplyCallbackImpl(TAutoPtr<TNetworkResponse> response);

    public:
        TRemoteQueryProcessor();
        ~TRemoteQueryProcessor() override;
        int GetSlaveCount() const {
            return BaseSearcherAddrs.ysize();
        }
        int GetCompId() {
            return CompId;
        }
        void IncLastCount(int compId);

        void GetExecPlan(TVector<ui16>* res) {
            *res = UniversalExecPlan;
        }
        TGUID SendQuery(int compId, const char* query, TVector<char>* cmd, IRemoteQueryResponseNotify* proc, int resId);
        void CancelQuery(const TGUID& req);
        void RegisterCmdType(const char* sz, ICmdProcessor* p);
        void SendReply(const TGUID& reqId, TVector<char>* response);
        void RegisterCallback(const TGUID& reqId, IRemoteQueryCancelNotify* notify);

        void RunMaster(const TVector<TNetworkAddress>& baseSearcherAddrs, unsigned short masterListenPort = 0);
        void RunSlave(int port);

        void StopSlaves();

        // needs an explicit Stop instead of just a destructor because some code in MetaThread can hold TInstrusivePtrs
        // to 'this' and then it is possible that the destructor will be called from MetaThread that will mean
        // self TThread::Join that is prohibited
        void Stop();
    };

    template <class T>
    class TQueryCancelCallback: public TNonCopyable {
        class TCallback: public IRemoteQueryCancelNotify {
            T* Obj;
            TAdaptiveLock IsExecutingCancel;

            void OnCancel() override {
                TGuard lock(IsExecutingCancel);
                if (Obj)
                    Obj->OnQueryCancel();
                Obj = nullptr;
            }

        public:
            TCallback(T* obj)
                : Obj(obj)
            {
            }
            void Detach() {
                TGuard lock(IsExecutingCancel); // wait complete
                Obj = nullptr;
            }
        };
        TIntrusivePtr<TCallback> Callback;

    public:
        ~TQueryCancelCallback() {
            Y_ASSERT(Callback.Get() == nullptr);
        }
        void Attach(T* obj, TRemoteQueryProcessor* queryProc, const TGUID& reqId) {
            if (reqId.IsEmpty() || queryProc == nullptr)
                return;
            Callback = new TCallback(obj);
            queryProc->RegisterCallback(reqId, Callback.Get());
        }
        void Detach() {
            if (Callback.Get() == nullptr)
                return;
            Callback->Detach();
            Callback = nullptr;
        }
    };
}
