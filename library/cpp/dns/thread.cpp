#include "thread.h"

#include "magic.h"

#include <util/network/socket.h>
#include <util/thread/factory.h>
#include <util/thread/lfqueue.h>
#include <util/system/event.h>
#include <util/generic/vector.h>
#include <util/generic/singleton.h>

using namespace NDns;

namespace {
    class TThreadedResolver: public IThreadFactory::IThreadAble, public TNonCopyable {
        struct TResolveRequest {
            inline TResolveRequest(const TString& host, ui16 port)
                : Host(host)
                , Port(port)
            {
            }

            inline TNetworkAddressPtr Wait() {
                E.Wait();

                if (!Error) {
                    if (!Result) {
                        ythrow TNetworkResolutionError(EAI_AGAIN) << TStringBuf(": resolver down");
                    }

                    return Result;
                }

                Error->Raise();

                ythrow TNetworkResolutionError(EAI_FAIL) << TStringBuf(": shit happen");
            }

            inline void Resolve() noexcept {
                try {
                    Result = new TNetworkAddress(Host, Port);
                } catch (...) {
                    Error = SaveError();
                }

                Wake();
            }

            inline void Wake() noexcept {
                E.Signal();
            }

            TString Host;
            ui16 Port;
            TManualEvent E;
            TNetworkAddressPtr Result;
            IErrorRef Error;
        };

    public:
        inline TThreadedResolver()
            : E_(TSystemEvent::rAuto)
        {
            T_.push_back(SystemThreadFactory()->Run(this));
        }

        inline ~TThreadedResolver() override {
            Schedule(nullptr);

            for (size_t i = 0; i < T_.size(); ++i) {
                T_[i]->Join();
            }

            {
                TResolveRequest* rr = nullptr;

                while (Q_.Dequeue(&rr)) {
                    if (rr) {
                        rr->Wake();
                    }
                }
            }
        }

        static inline TThreadedResolver* Instance() {
            return Singleton<TThreadedResolver>();
        }

        inline TNetworkAddressPtr Resolve(const TString& host, ui16 port) {
            TResolveRequest rr(host, port);

            Schedule(&rr);

            return rr.Wait();
        }

    private:
        inline void Schedule(TResolveRequest* rr) {
            Q_.Enqueue(rr);
            E_.Signal();
        }

        void DoExecute() override {
            while (true) {
                TResolveRequest* rr = nullptr;

                while (!Q_.Dequeue(&rr)) {
                    E_.Wait();
                }

                if (rr) {
                    rr->Resolve();
                } else {
                    break;
                }
            }

            Schedule(nullptr);
        }

    private:
        TLockFreeQueue<TResolveRequest*> Q_;
        TSystemEvent E_;
        typedef TAutoPtr<IThreadFactory::IThread> IThreadRef;
        TVector<IThreadRef> T_;
    };
}

namespace NDns {
    TNetworkAddressPtr ThreadedResolve(const TString& host, ui16 port) {
        return TThreadedResolver::Instance()->Resolve(host, port);
    }
}
