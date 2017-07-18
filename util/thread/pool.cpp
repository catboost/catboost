#include "pool.h"

#include <util/system/thread.h>
#include <util/generic/singleton.h>

using IThread = IThreadPool::IThread;

namespace {
    class TSystemThreadPool: public IThreadPool {
    public:
        class TPoolThread: public IThread {
        public:
            ~TPoolThread() override {
                if (Thr_) {
                    Thr_->Detach();
                }
            }

            void DoRun(IThreadAble* func) override {
                Thr_.Reset(new TThread(ThreadProc, func));

                Thr_->Start();
            }

            void DoJoin() noexcept override {
                if (!Thr_) {
                    return;
                }

                Thr_->Join();
                Thr_.Destroy();
            }

        private:
            static void* ThreadProc(void* func) {
                ((IThreadAble*)(func))->Execute();

                return nullptr;
            }

        private:
            THolder<TThread> Thr_;
        };

        inline TSystemThreadPool() noexcept {
        }

        IThread* DoCreate() override {
            return new TPoolThread;
        }
    };

    class TThrPoolFuncObj: public IThreadPool::IThreadAble {
    public:
        TThrPoolFuncObj(const std::function<void()>& func)
            : Func(func)
        {
        }
        void DoExecute() override {
            THolder<TThrPoolFuncObj> self(this);
            Func();
        }

    private:
        std::function<void()> Func;
    };
}

TAutoPtr<IThread> IThreadPool::Run(std::function<void()> func) {
    TAutoPtr<IThread> ret(DoCreate());

    ret->Run(new ::TThrPoolFuncObj(func));

    return ret;
}

static IThreadPool* SystemThreadPoolImpl() {
    return Singleton<TSystemThreadPool>();
}

static IThreadPool* systemPool = nullptr;

IThreadPool* SystemThreadPool() {
    if (systemPool) {
        return systemPool;
    }

    return SystemThreadPoolImpl();
}

void SetSystemThreadPool(IThreadPool* pool) {
    systemPool = pool;
}
