#include "rpc.h"
#include "rq.h"
#include "multi.h"
#include "location.h"
#include "utils.h"

#include <util/generic/hash.h>
#include <util/thread/factory.h>
#include <util/system/yield.h>
#include <util/system/spinlock.h>

using namespace NNeh;

namespace {
    typedef std::pair<TString, IServiceRef> TServiceDescr;
    typedef TVector<TServiceDescr> TServicesBase;

    class TServices: public TServicesBase, public TThrRefBase, public IOnRequest {
        typedef THashMap<TStringBuf, IServiceRef> TSrvs;

        struct TFunc: public IThreadFactory::IThreadAble {
            inline TFunc(TServices* parent)
                : Parent(parent)
            {
            }

            void DoExecute() override {
                TSrvs srvs;
                i64 version = -1;

                while (true) {
                    IRequestRef req = Parent->RQ_->Next();

                    if (!req) {
                        break;
                    }

                    const TStringBuf name = req->Service();
                    TSrvs::const_iterator it = srvs.find(name);

                    if (Y_UNLIKELY(it == srvs.end())) {
                        if (Parent->UpdateServices(srvs, version)) {
                            it = srvs.find(name);
                        }
                    }

                    if (Y_UNLIKELY(it == srvs.end())) {
                        it = srvs.find(AsStringBuf("*"));
                    }

                    if (Y_UNLIKELY(it == srvs.end())) {
                        req->SendError(IRequest::NotExistService);
                    } else {
                        try {
                            it->second->ServeRequest(req);
                        } catch (...) {
                            Cdbg << CurrentExceptionMessage() << Endl;
                        }
                    }
                }

                Parent->RQ_->Schedule(nullptr);
            }

            TServices* Parent;
        };

    public:
        inline TServices()
            : RQ_(CreateRequestQueue())
        {
        }

        inline TServices(TCheck check)
            : RQ_(CreateRequestQueue())
            , C_(check)
        {
        }

        inline ~TServices() override {
            LF_.Destroy();
        }

        inline void Add(const TString& service, IServiceRef srv) {
            TGuard<TSpinLock> guard(L_);

            push_back(std::make_pair(service, srv));
            AtomicIncrement(SelfVersion_);
        }

        inline void Loop(size_t threads) {
            TIntrusivePtr<TServices> self(this);
            IRequesterRef rr = MultiRequester(ListenAddrs(), this);
            TFunc func(this);

            typedef TAutoPtr<IThreadFactory::IThread> IThreadRef;
            TVector<IThreadRef> thrs;

            for (size_t i = 1; i < threads; ++i) {
                thrs.push_back(SystemThreadFactory()->Run(&func));
            }

            func.Execute();

            for (size_t i = 0; i < thrs.size(); ++i) {
                thrs[i]->Join();
            }
            RQ_->Clear();
        }

        inline void ForkLoop(size_t threads) {
            //here we can have trouble with binding port(s), so expect exceptions
            IRequesterRef rr = MultiRequester(ListenAddrs(), this);
            LF_.Reset(new TLoopFunc(this, threads, rr));
        }

        inline void Stop() {
            RQ_->Schedule(nullptr);
        }

        inline void SyncStopFork() {
            Stop();
            LF_.Destroy();
            RQ_->Clear();
        }

        void OnRequest(IRequestRef req) override {
            if (C_) {
                if (auto error = C_(req)) {
                    req->SendError(*error);
                    return;
                }
            }
            RQ_->Schedule(req);
        }

    private:
        class TLoopFunc: public TFunc {
        public:
            TLoopFunc(TServices* parent, size_t threads, IRequesterRef& rr)
                : TFunc(parent)
                , RR_(rr)
            {
                T_.reserve(threads);

                try {
                    for (size_t i = 0; i < threads; ++i) {
                        T_.push_back(SystemThreadFactory()->Run(this));
                    }
                } catch (...) {
                    //paranoid mode on
                    SyncStop();
                    throw;
                }
            }

            ~TLoopFunc() override {
                try {
                    SyncStop();
                } catch (...) {
                    Cdbg << AsStringBuf("neh rpc ~loop_func: ") << CurrentExceptionMessage() << Endl;
                }
            }

        private:
            void SyncStop() {
                Parent->Stop();

                for (size_t i = 0; i < T_.size(); ++i) {
                    T_[i]->Join();
                }
            }

            typedef TAutoPtr<IThreadFactory::IThread> IThreadRef;
            TVector<IThreadRef> T_;
            IRequesterRef RR_;
        };

        inline bool UpdateServices(TSrvs& srvs, i64& version) const {
            if (AtomicGet(SelfVersion_) == version) {
                return false;
            }

            srvs.clear();

            TGuard<TSpinLock> guard(L_);

            for (const auto& it : *this) {
                srvs[TParsedLocation(it.first).Service] = it.second;
            }
            version = AtomicGet(SelfVersion_);

            return true;
        }

        inline TListenAddrs ListenAddrs() const {
            TListenAddrs addrs;

            {
                TGuard<TSpinLock> guard(L_);

                for (const auto& it : *this) {
                    addrs.push_back(it.first);
                }
            }

            return addrs;
        }

        TSpinLock L_;
        IRequestQueueRef RQ_;
        THolder<TLoopFunc> LF_;
        TAtomic SelfVersion_ = 0;
        TCheck C_;
    };

    class TServicesFace: public IServices {
    public:
        inline TServicesFace()
            : S_(new TServices())
        {
        }

        inline TServicesFace(TCheck check)
            : S_(new TServices(check))
        {
        }

        void DoAdd(const TString& service, IServiceRef srv) override {
            S_->Add(service, srv);
        }

        void Loop(size_t threads) override {
            S_->Loop(threads);
        }

        void ForkLoop(size_t threads) override {
            S_->ForkLoop(threads);
        }

        void SyncStopFork() override {
            S_->SyncStopFork();
        }

        void Stop() override {
            S_->Stop();
        }

    private:
        TIntrusivePtr<TServices> S_;
    };
}

IServiceRef NNeh::Wrap(const TServiceFunction& func) {
    struct TWrapper: public IService {
        inline TWrapper(const TServiceFunction& f)
            : F(f)
        {
        }

        void ServeRequest(const IRequestRef& request) override {
            F(request);
        }

        TServiceFunction F;
    };

    return new TWrapper(func);
}

IServicesRef NNeh::CreateLoop() {
    return new TServicesFace();
}

IServicesRef NNeh::CreateLoop(TCheck check) {
    return new TServicesFace(check);
}
