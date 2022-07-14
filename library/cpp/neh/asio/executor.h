#pragma once

#include "asio.h"

#include <library/cpp/deprecated/atomic/atomic.h>

#include <util/thread/factory.h>
#include <util/system/thread.h>

namespace NAsio {
    class TIOServiceExecutor: public IThreadFactory::IThreadAble {
    public:
        TIOServiceExecutor()
            : Work_(new TIOService::TWork(Srv_))
        {
            T_ = SystemThreadFactory()->Run(this);
        }

        ~TIOServiceExecutor() override {
            SyncShutdown();
        }

        void DoExecute() override {
            TThread::SetCurrentThreadName("NehAsioExecutor");
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
        typedef TAutoPtr<IThreadFactory::IThread> IThreadRef;
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
            return E_.size();
        }

        inline TIOServiceExecutor& GetExecutor() noexcept {
            TAtomicBase next = AtomicIncrement(C_);
            return *E_[next % E_.size()];
        }

        void SyncShutdown() {
            for (size_t i = 0; i < E_.size(); ++i) {
                E_[i]->SyncShutdown();
            }
        }

    private:
        TAtomic C_;
        TVector<TAutoPtr<TIOServiceExecutor>> E_;
    };
}
