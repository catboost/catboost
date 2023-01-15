#include "utils.h"
#include "lfqueue.h"
#include "jobqueue.h"
#include "pipequeue.h"

#include <util/thread/factory.h>
#include <util/generic/singleton.h>
#include <util/system/thread.h>

using namespace NNeh;

namespace {
    class TExecThread: public IThreadFactory::IThreadAble, public IJob {
    public:
        TExecThread()
            : T_(SystemThreadFactory()->Run(this))
        {
        }

        ~TExecThread() override {
            Enqueue(this);
            T_->Join();
        }

        inline void Enqueue(IJob* job) {
            Q_.Enqueue(job);
        }

    private:
        void DoRun(TCont* c) override {
            c->Executor()->Abort();
        }

        void DoExecute() override {
            SetHighestThreadPriority();

            TContExecutor e(RealStackSize(20000));

            e.Execute<TExecThread, &TExecThread::Dispatcher>(this);
        }

        inline void Dispatcher(TCont* c) {
            IJob* job;

            while ((job = Q_.Dequeue(c))) {
                try {
                    c->Executor()->Create(*job, "job");
                } catch (...) {
                    (*job)(c);
                }
            }
        }

        typedef TAutoPtr<IThreadFactory::IThread> IThreadRef;
        TOneConsumerPipeQueue<IJob> Q_;
        IThreadRef T_;
    };

    class TJobScatter: public IJobQueue {
    public:
        inline TJobScatter() {
            for (size_t i = 0; i < 2; ++i) {
                E_.push_back(new TExecThread());
            }
        }

        void ScheduleImpl(IJob* job) override {
            E_[TThread::CurrentThreadId() % E_.size()]->Enqueue(job);
        }

    private:
        typedef TAutoPtr<TExecThread> TExecThreadRef;
        TVector<TExecThreadRef> E_;
    };
}

IJobQueue* NNeh::JobQueue() {
    return Singleton<TJobScatter>();
}
