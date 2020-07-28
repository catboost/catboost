#include "neh.h"
#include "rpc.h"

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/testing/unittest/tests_data.h>

#include <util/generic/scope.h>
#include <util/network/ip.h>
#include <util/system/condvar.h>
#include <util/system/guard.h>
#include <util/system/mutex.h>

namespace {
    class TServicer {
    public:
        TServicer()
            : Port_(PortManager_.GetPort())
            , Services_(NNeh::CreateLoop())
        {
            Services_->Add(Endpoint(), WaitingService_);
            Services_->ForkLoop(2);
        }

        TString Endpoint() const {
            return "http://localhost:" + ToString(Port_) + "/wait";
        }

        void Signal() {
            WaitingService_.Signal();
        }

        void SyncStopFork() {
            Services_->SyncStopFork();
        }

    private:
        class TWaitingService : public NNeh::IService {
        public:
            void ServeRequest(const NNeh::IRequestRef& r) override {
                with_lock (Lock_) {
                    CondVar_.WaitI(Lock_, [&] { return StopWaiting_; });
                }

                TVector<char> v;
                r->SendReply(v);
            }

            void Signal() noexcept {
                with_lock (Lock_) {
                    StopWaiting_ = true;
                }

                CondVar_.Signal();
            }

        public:
            TMutex Lock_;
            TCondVar CondVar_;
            bool StopWaiting_ = false;
        };

    private:
        TPortManager PortManager_;
        TIpPort Port_ = -1;
        TWaitingService WaitingService_;
        NNeh::IServicesRef Services_;
    };

    class TCallback : public NNeh::IOnRecv {
    public:
        TCallback(TMutex* lock, TCondVar* cv, bool* handleDestroyed, NNeh::THandleRef* h)
            : Lock_(lock)
            , Cv_(cv)
            , HandleDestroyed_(handleDestroyed)
            , H_(h)
        {
        }

        void OnNotify(NNeh::THandle&) override {
            (*H_) = nullptr;
            // after `*H_` being set to `nullptr` only `neh` will have ownership of handle
        }

        void OnRecv(NNeh::THandle&) override {
            with_lock (*Lock_) {
                *HandleDestroyed_ = true;
            }

            Cv_->Signal();
        }

    private:
        TMutex* Lock_ = nullptr;
        TCondVar* Cv_ = nullptr;
        bool* HandleDestroyed_ = nullptr;
        NNeh::THandleRef* H_ = nullptr;
    };
}

Y_UNIT_TEST_SUITE(CancelHeapUseAfterFreeTests) {
    Y_UNIT_TEST(TestHttp) {
        TServicer servicer;
        Y_DEFER { servicer.SyncStopFork(); };

        NNeh::TMessage m(servicer.Endpoint(), {});

        TMutex lock;
        TCondVar cv;
        bool handleDestroyed = false;

        NNeh::THandleRef h;
        TCallback c(&lock, &cv, &handleDestroyed, &h);;
        Y_DEFER { cv.Wait(lock, [&] { return handleDestroyed; }); };

        h = NNeh::Request(m, &c);
        h->Cancel();
        servicer.Signal();
    }
}
