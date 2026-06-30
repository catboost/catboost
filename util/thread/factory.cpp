#include "factory.h"

#include <util/system/thread.h>
#include <util/generic/singleton.h>

using IThread = IThreadFactory::IThread;

namespace {
    class TSystemThreadFactory: public IThreadFactory {
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

        inline TSystemThreadFactory() noexcept {
        }

        IThread* DoCreate() override {
            return new TPoolThread;
        }
    };

    class TThreadFactoryFuncObj: public IThreadFactory::IThreadAble {
    public:
        TThreadFactoryFuncObj(const std::function<void()>& func)
            : Func(func)
        {
        }
        void DoExecute() override {
            THolder<TThreadFactoryFuncObj> self(this);
            Func();
        }

    private:
        std::function<void()> Func;
    };
} // namespace

THolder<IThread> IThreadFactory::Run(const std::function<void()>& func) {
    THolder<IThread> ret(DoCreate());

    ret->Run(new ::TThreadFactoryFuncObj(func));

    return ret;
}

static IThreadFactory* SystemThreadPoolImpl() {
    return Singleton<TSystemThreadFactory>();
}

static IThreadFactory* systemPool = nullptr;

IThreadFactory* SystemThreadFactory() {
    if (systemPool) {
        return systemPool;
    }

    return SystemThreadPoolImpl();
}

void SetSystemThreadFactory(IThreadFactory* pool) {
    systemPool = pool;
}
