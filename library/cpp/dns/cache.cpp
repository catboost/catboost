#include "cache.h"

#include "thread.h"

#include <util/system/tls.h>
#include <util/system/info.h>
#include <util/system/rwlock.h>
#include <util/thread/singleton.h>
#include <util/generic/singleton.h>
#include <util/generic/hash.h>

using namespace NDns;

namespace {
    struct TResolveTask {
        enum EMethod {
            Normal,
            Threaded
        };

        inline TResolveTask(const TResolveInfo& info, EMethod method)
            : Info(info)
            , Method(method)
        {
        }

        const TResolveInfo& Info;
        const EMethod Method;
    };

    class IDns {
    public:
        virtual ~IDns() = default;
        virtual const TResolvedHost* Resolve(const TResolveTask&) = 0;
    };

    typedef TAtomicSharedPtr<TResolvedHost> TResolvedHostPtr;

    struct THashResolveInfo {
        inline size_t operator()(const TResolveInfo& ri) const {
            return ComputeHash(ri.Host) ^ ri.Port;
        }
    };

    struct TCompareResolveInfo {
        inline bool operator()(const NDns::TResolveInfo& x, const NDns::TResolveInfo& y) const {
            return x.Host == y.Host && x.Port == y.Port;
        }
    };

    class TGlobalCachedDns: public IDns, public TNonCopyable {
    public:
        const TResolvedHost* Resolve(const TResolveTask& rt) override {
            //2. search host in cache
            {
                TReadGuard guard(L_);

                TCache::const_iterator it = C_.find(rt.Info);

                if (it != C_.end()) {
                    return it->second.Get();
                }
            }

            TResolvedHostPtr res = ResolveA(rt);

            //update cache
            {
                TWriteGuard guard(L_);

                std::pair<TCache::iterator, bool> updateResult = C_.insert(std::make_pair(TResolveInfo(res->Host, rt.Info.Port), res));
                TResolvedHost* rh = updateResult.first->second.Get();

                if (updateResult.second) {
                    //fresh resolved host, set cache record id for it
                    rh->Id = C_.size() - 1;
                }

                return rh;
            }
        }

        void AddAlias(const TString& host, const TString& alias) noexcept {
            TWriteGuard guard(LA_);

            A_[host] = alias;
        }

        static inline TGlobalCachedDns* Instance() {
            return SingletonWithPriority<TGlobalCachedDns, 65530>();
        }

    private:
        inline TResolvedHostPtr ResolveA(const TResolveTask& rt) {
            TString originalHost(rt.Info.Host);
            TString host(originalHost);

            //3. replace host to alias, if exist
            if (A_.size()) {
                TReadGuard guard(LA_);
                TStringBuf names[] = {"*", host};

                for (const auto& name : names) {
                    TAliases::const_iterator it = A_.find(name);

                    if (it != A_.end()) {
                        host = it->second;
                    }
                }
            }

            if (host.length() > 2 && host[0] == '[') {
                TString unbracedIpV6(host.data() + 1, host.size() - 2);
                host.swap(unbracedIpV6);
            }

            TAutoPtr<TNetworkAddress> na;

            //4. getaddrinfo (direct or in separate thread)
            if (rt.Method == TResolveTask::Normal) {
                na.Reset(new TNetworkAddress(host, rt.Info.Port));
            } else if (rt.Method == TResolveTask::Threaded) {
                na = ThreadedResolve(host, rt.Info.Port);
            } else {
                Y_ASSERT(0);
                throw yexception() << TStringBuf("invalid resolve method");
            }

            return new TResolvedHost(originalHost, *na);
        }

        typedef THashMap<TResolveInfo, TResolvedHostPtr, THashResolveInfo, TCompareResolveInfo> TCache;
        TCache C_;
        TRWMutex L_;
        typedef THashMap<TString, TString> TAliases;
        TAliases A_;
        TRWMutex LA_;
    };

    class TCachedDns: public IDns {
    public:
        inline TCachedDns(IDns* slave)
            : S_(slave)
        {
        }

        const TResolvedHost* Resolve(const TResolveTask& rt) override {
            //1. search in local thread cache
            {
                TCache::const_iterator it = C_.find(rt.Info);

                if (it != C_.end()) {
                    return it->second;
                }
            }

            const TResolvedHost* res = S_->Resolve(rt);

            C_[TResolveInfo(res->Host, rt.Info.Port)] = res;

            return res;
        }

    private:
        typedef THashMap<TResolveInfo, const TResolvedHost*, THashResolveInfo, TCompareResolveInfo> TCache;
        TCache C_;
        IDns* S_;
    };

    struct TThreadedDns: public TCachedDns {
        inline TThreadedDns()
            : TCachedDns(TGlobalCachedDns::Instance())
        {
        }
    };

    inline IDns* ThrDns() {
        return FastTlsSingleton<TThreadedDns>();
    }
}

namespace NDns {
    const TResolvedHost* CachedResolve(const TResolveInfo& ri) {
        TResolveTask rt(ri, TResolveTask::Normal);

        return ThrDns()->Resolve(rt);
    }

    const TResolvedHost* CachedThrResolve(const TResolveInfo& ri) {
        TResolveTask rt(ri, TResolveTask::Threaded);

        return ThrDns()->Resolve(rt);
    }

    void AddHostAlias(const TString& host, const TString& alias) {
        TGlobalCachedDns::Instance()->AddAlias(host, alias);
    }
}
