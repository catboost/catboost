#include "local_executor.h"

#include <util/system/thread.h>
#include <util/system/yield.h>
#include <util/generic/utility.h>

#ifdef _freebsd_
#include <sys/syscall.h>
#endif

#ifdef _win_
static void RegularYield() {
}
#else
// unix actually has cooperative multitasking! :)
// without this function program runs slower and system lags for some magic reason
static void RegularYield() {
    SchedYield();
}
#endif

namespace NPar {

    class TLocalRangeExecutor: public ILocallyExecutable {
        TIntrusivePtr<ILocallyExecutable> Exec;
        TAtomic Counter, WorkerCount;
        int LastId;

        void LocalExec(int) override {
            AtomicAdd(WorkerCount, 1);
            for (;;) {
                if (!DoSingleOp())
                    break;
            }
            AtomicAdd(WorkerCount, -1);
        }

    public:
        TLocalRangeExecutor(ILocallyExecutable* exec, int firstId, int lastId)
            : Exec(exec)
            , Counter(firstId)
            , WorkerCount(0)
            , LastId(lastId)
        {
        }
        bool DoSingleOp() {
            TAtomic id = AtomicAdd(Counter, 1) - 1;
            if (id >= LastId)
                return false;
            Exec->LocalExec(id);
            RegularYield();
            return true;
        }
        void WaitComplete() {
            while (WorkerCount > 0)
                RegularYield();
        }
        int GetRangeSize() const {
            return Max<int>(LastId - Counter, 0);
        }
    };

    struct TFunctionWrapper : ILocallyExecutable {
        TLocallyExecutableFunction Exec;
        TFunctionWrapper(TLocallyExecutableFunction exec)
            : Exec(exec)
        {
        }
        void LocalExec(int id) override {
            Exec(id);
        }
    };

    //////////////////////////////////////////////////////////////////////////
    const int FAST_ITERATIONS = 200;
    void* TLocalExecutor::HostWorkerThread(void* p) {
        TLocalExecutor* ctx = (TLocalExecutor*)p;
        TThread::CurrentThreadSetName("ParLocalExecutor");
        ctx->WorkerThreadId = AtomicAdd(ctx->ThreadId, 1);
        for (bool cont = true; cont;) {
            TSingleJob job;
            bool gotJob = false;
            for (int iter = 0; iter < FAST_ITERATIONS; ++iter) {
                if (ctx->GetJob(&job)) {
                    gotJob = true;
                    break;
                }
            }
            if (!gotJob) {
                ctx->HasJob.Reset();
                if (!ctx->GetJob(&job)) {
                    ctx->HasJob.Wait();
                    continue;
                }
            }
            if (job.Exec.Get()) {
                job.Exec->LocalExec(job.Id);
                RegularYield();
            } else {
                AtomicAdd(ctx->QueueSize, 1);
                ctx->JobQueue.Enqueue(job);
                ctx->HasJob.Signal();
                cont = false;
            }
        }
        AtomicAdd(ctx->ThreadCount, -1);
        return nullptr;
    }

    bool TLocalExecutor::GetJob(TSingleJob* job) {
        if (JobQueue.Dequeue(job)) {
            CurrentTaskPriority = TLocalExecutor::HIGH_PRIORITY;
            AtomicAdd(QueueSize, -1);
            return true;
        } else if (MedJobQueue.Dequeue(job)) {
            CurrentTaskPriority = TLocalExecutor::MED_PRIORITY;
            AtomicAdd(MPQueueSize, -1);
            return true;
        } else if (LowJobQueue.Dequeue(job)) {
            CurrentTaskPriority = TLocalExecutor::LOW_PRIORITY;
            AtomicAdd(LPQueueSize, -1);
            return true;
        }
        return false;
    }

    TLocalExecutor::~TLocalExecutor() {
        AtomicAdd(QueueSize, 1);
        JobQueue.Enqueue(TSingleJob(nullptr, 0));
        HasJob.Signal();
        while (ThreadCount) {
            ThreadYield();
        }
    }

    void TLocalExecutor::RunAdditionalThreads(int threadCount) {
        for (int i = 0; i < threadCount; i++)
            RunNewThread();
    }

    void TLocalExecutor::RunNewThread() {
        AtomicAdd(ThreadCount, 1);
        TThread thr(HostWorkerThread, this);
        thr.Start();
        thr.Detach();
    }

    void TLocalExecutor::Exec(ILocallyExecutable* exec, int id, int flags) {
        Y_ASSERT((flags & WAIT_COMPLETE) == 0); // unsupported
        int prior = Max<int>(CurrentTaskPriority, flags & PRIORITY_MASK);
        switch (prior) {
            case HIGH_PRIORITY:
                AtomicAdd(QueueSize, 1);
                JobQueue.Enqueue(TSingleJob(exec, id));
                break;
            case MED_PRIORITY:
                AtomicAdd(MPQueueSize, 1);
                MedJobQueue.Enqueue(TSingleJob(exec, id));
                break;
            case LOW_PRIORITY:
                AtomicAdd(LPQueueSize, 1);
                LowJobQueue.Enqueue(TSingleJob(exec, id));
                break;
            default:
                Y_ASSERT(0);
                break;
        }
        HasJob.Signal();
    }

    void TLocalExecutor::Exec(TLocallyExecutableFunction exec, int id, int flags) {
        Exec(new TFunctionWrapper(exec), id, flags);
    }

    void TLocalExecutor::LaunchRange(TLocalRangeExecutor* rangeExec, int queueSizeLimit,
                                     TAtomic* queueSize, TLockFreeQueue<TSingleJob>* jobQueue) {
        int count = Min<int>(ThreadCount + 1, rangeExec->GetRangeSize());
        if (queueSizeLimit >= 0 && *queueSize >= queueSizeLimit) {
            return;
        }
        AtomicAdd(*queueSize, count);
        for (int i = 0; i < count; ++i) {
            jobQueue->Enqueue(TSingleJob(rangeExec, 0));
        }
        HasJob.Signal();
    }

    void TLocalExecutor::ExecRange(ILocallyExecutable* exec, int firstId, int lastId, int flags) {
        Y_ASSERT(lastId >= firstId);
        if (firstId >= lastId) {
            TIntrusivePtr<ILocallyExecutable> tmp(exec); // ref and unref for possible deletion if unowned object passed
            return;
        }
        if ((flags & WAIT_COMPLETE) && (lastId - firstId) == 1) {
            TIntrusivePtr<ILocallyExecutable> execHolder(exec);
            execHolder->LocalExec(firstId);
            return;
        }
        TIntrusivePtr<TLocalRangeExecutor> rangeExec = new TLocalRangeExecutor(exec, firstId, lastId);
        int queueSizeLimit = (flags & WAIT_COMPLETE) ? 10000 : -1;
        int prior = Max<int>(CurrentTaskPriority, flags & PRIORITY_MASK);
        switch (prior) {
            case HIGH_PRIORITY:
                LaunchRange(rangeExec.Get(), queueSizeLimit, &QueueSize, &JobQueue);
                break;
            case MED_PRIORITY:
                LaunchRange(rangeExec.Get(), queueSizeLimit, &MPQueueSize, &MedJobQueue);
                break;
            case LOW_PRIORITY:
                LaunchRange(rangeExec.Get(), queueSizeLimit, &LPQueueSize, &LowJobQueue);
                break;
            default:
                Y_ASSERT(0);
                break;
        }
        if (flags & WAIT_COMPLETE) {
            int keepPrior = CurrentTaskPriority;
            CurrentTaskPriority = prior;
            while (rangeExec->DoSingleOp())
                ;
            CurrentTaskPriority = keepPrior;
            rangeExec->WaitComplete();
        }
    }

    void TLocalExecutor::ExecRange(TLocallyExecutableFunction exec, int firstId, int lastId, int flags) {
        ExecRange(new TFunctionWrapper(exec), firstId, lastId, flags);
    }

    void TLocalExecutor::ClearLPQueue() {
        for (bool cont = true; cont;) {
            cont = false;
            TSingleJob job;
            while (LowJobQueue.Dequeue(&job)) {
                AtomicAdd(LPQueueSize, -1);
                cont = true;
            }
            while (MedJobQueue.Dequeue(&job)) {
                AtomicAdd(MPQueueSize, -1);
                cont = true;
            }
        }
    }

//////////////////////////////////////////////////////////////////////////
#ifdef _freebsd_
    Y_POD_THREAD(long)
    ThreadSelfId;

    TFastFreeBsdEvent::TFastFreeBsdEvent()
        : ThreadCount(0)
        , Signaled(0)
    {
        Zero(ThreadId);
        for (int i = 0; i < MAX_THREAD_COUNT; ++i)
            ThreadState[i] = 0;
    }

    int TFastFreeBsdEvent::GetCurrentThreadIdx() {
        if (ThreadSelfId == 0) {
            syscall(SYS_thr_self, &ThreadSelfId);
        }

        int tc = Min((int)ThreadCount, (int)MAX_THREAD_COUNT);
        for (int i = 0; i < tc; ++i) {
            if (ThreadId[i] == ThreadSelfId)
                return i;
        }
        if (ThreadCount >= MAX_THREAD_COUNT)
            return -1;
        int arrIdx = AtomicAdd(ThreadCount, 1) - 1;
        if (arrIdx >= MAX_THREAD_COUNT)
            return -1;
        ThreadId[arrIdx] = ThreadSelfId;
        return arrIdx;
    }

    void TFastFreeBsdEvent::Signal() {
        if (AtomicCas(&Signaled, (void*)-1, (void*)0)) {
            int tc = Min((int)ThreadCount, (int)MAX_THREAD_COUNT);
            for (int i = 0; i < tc; ++i) {
                if (AtomicCas(&ThreadState[i], (void*)-1, (void*)-2))
                    syscall(SYS_thr_wake, ThreadId[i]);
            }
        }
    }

    void TFastFreeBsdEvent::Reset() {
        Signaled = (void*)0;
    }

    void TFastFreeBsdEvent::Wait() {
        if (Signaled)
            return;

        int arrIdx = GetCurrentThreadIdx();
        if (arrIdx < 0)
            return;

        timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 10;

        if (AtomicCas(&ThreadState[arrIdx], (void*)-2, (void*)0)) {
            while (!Signaled && ThreadState[arrIdx] == (void*)-2) {
                syscall(SYS_thr_suspend, &ts);
                ts.tv_nsec = Min(500 * 1000000, (int)ts.tv_nsec * 2);
            }
        }
        ThreadState[arrIdx] = (void*)0;
    }
#endif
}
