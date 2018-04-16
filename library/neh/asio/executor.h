#pragma once

#include "asio.h"

#include <util/thread/pool.h>

namespace NAsio {
    class TIOServiceExecutor: public IThreadPool::IThreadAble {
    public:
        TIOServiceExecutor()
            : Work_(new TIOService::TWork(Srv_))
        {
            T_ = SystemThreadPool()->Run(this);
        }

        ~TIOServiceExecutor() override {
            SyncShutdown();
        }

        void DoExecute() override {
            Srv_.Run();
        }

        inline TIOService& GetIOService() noexcept {
            return Srv_;
        }

        void SyncShutdown() {
            if (Work_) {
                Work_.Destroy();
                Srv_.Abort(); //cancel all async operations, break Run() execution
                T_->Join();
            }
        }

    private:
        TIOService Srv_;
        TAutoPtr<TIOService::TWork> Work_;
        typedef TAutoPtr<IThreadPool::IThread> IThreadRef;
        IThreadRef T_;
    };

    class TExecutorsPool {
    public:
        TExecutorsPool(size_t executors)
            : C_(0)
        {
            for (size_t i = 0; i < executors; ++i) {
                E_.push_back(new TIOServiceExecutor());
            }
        }

        inline size_t Size() const noexcept {
            return +E_;
        }

        inline TIOServiceExecutor& GetExecutor() noexcept {
            TAtomicBase next = AtomicIncrement(C_);
            return *E_[next % +E_];
        }

        void SyncShutdown() {
            for (size_t i = 0; i < +E_; ++i) {
                E_[i]->SyncShutdown();
            }
        }

    private:
        TAtomic C_;
        TVector<TAutoPtr<TIOServiceExecutor>> E_;
    };
}
