#include "local_executor.h"

#include <library/threading/future/future.h>

#include <util/generic/utility.h>
#include <util/system/event.h>
#include <util/system/thread.h>
#include <util/system/tls.h>
#include <util/system/yield.h>
#include <util/thread/lfqueue.h>

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

namespace {
    struct TFunctionWrapper : NPar::ILocallyExecutable {
        NPar::TLocallyExecutableFunction Exec;
        TFunctionWrapper(NPar::TLocallyExecutableFunction exec)
            : Exec(exec)
        {
        }
        void LocalExec(int id) override {
            Exec(id);
        }
    };

    class TFunctionWrapperWithPromise: public NPar::ILocallyExecutable {
    private:
        NPar::TLocallyExecutableFunction Exec;
        int FirstId, LastId;
        TVector<NThreading::TPromise<void>> Promises;

    public:
        TFunctionWrapperWithPromise(NPar::TLocallyExecutableFunction exec, int firstId, int lastId)
            : Exec(exec)
            , FirstId(firstId)
            , LastId(lastId)
        {
            Y_ASSERT(FirstId <= LastId);
            const int rangeSize = LastId - FirstId;
            Promises.resize(rangeSize, NThreading::NewPromise());
            for (auto& promise : Promises) {
                promise = NThreading::NewPromise();
            }
        }

        void LocalExec(int id) override {
            Y_ASSERT(FirstId <= id && id < LastId);
            NThreading::NImpl::SetValue(Promises[id - FirstId], [=] { Exec(id); });
        }

        TVector<NThreading::TFuture<void>> GetFutures() const {
            TVector<NThreading::TFuture<void>> out;
            out.reserve(Promises.ysize());
            for (auto& promise : Promises) {
                out.push_back(promise.GetFuture());
            }
            return out;
        }
    };

    struct TSingleJob {
        TIntrusivePtr<NPar::ILocallyExecutable> Exec;
        int Id{0};

        TSingleJob() = default;
        TSingleJob(NPar::ILocallyExecutable* exec, int id)
            : Exec(exec)
            , Id(id)
        {
        }
    };

    class TLocalRangeExecutor: public NPar::ILocallyExecutable {
        TIntrusivePtr<NPar::ILocallyExecutable> Exec;
        TAtomic Counter;
        TAtomic WorkerCount;
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
            while (AtomicGet(WorkerCount) > 0)
                RegularYield();
        }
        int GetRangeSize() const {
            return Max<int>(LastId - Counter, 0);
        }
    };

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

//////////////////////////////////////////////////////////////////////////
class NPar::TLocalExecutor::TImpl {
public:
    TLockFreeQueue<TSingleJob> JobQueue;
    TLockFreeQueue<TSingleJob> MedJobQueue;
    TLockFreeQueue<TSingleJob> LowJobQueue;
#ifdef _freebsd_
    TFastFreeBsdEvent HasJob;
#else
    Event HasJob;
#endif

    TAtomic ThreadCount{0};
    TAtomic QueueSize{0};
    TAtomic MPQueueSize{0};
    TAtomic LPQueueSize{0};
    TAtomic ThreadId{0};

    Y_THREAD(int)
    CurrentTaskPriority;
    Y_THREAD(int)
    WorkerThreadId;

    static void* HostWorkerThread(void* p);
    bool GetJob(TSingleJob* job);
    void RunNewThread();
    void LaunchRange(TLocalRangeExecutor* execRange, int queueSizeLimit,
                        TAtomic* queueSize, TLockFreeQueue<TSingleJob>* jobQueue);

    TImpl() = default;
    ~TImpl();
};

NPar::TLocalExecutor::TImpl::~TImpl() {
    AtomicAdd(QueueSize, 1);
    JobQueue.Enqueue(TSingleJob(nullptr, 0));
    HasJob.Signal();
    while (AtomicGet(ThreadCount)) {
        ThreadYield();
    }
}

void* NPar::TLocalExecutor::TImpl::HostWorkerThread(void* p) {
    static const int FAST_ITERATIONS = 200;

    auto* const ctx = (TImpl*)p;
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

bool NPar::TLocalExecutor::TImpl::GetJob(TSingleJob* job) {
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

void NPar::TLocalExecutor::TImpl::RunNewThread() {
    AtomicAdd(ThreadCount, 1);
    TThread thr(HostWorkerThread, this);
    thr.Start();
    thr.Detach();
}

void NPar::TLocalExecutor::TImpl::LaunchRange(TLocalRangeExecutor* rangeExec, int queueSizeLimit,
                                    TAtomic* queueSize, TLockFreeQueue<TSingleJob>* jobQueue) {
    int count = Min<int>(ThreadCount + 1, rangeExec->GetRangeSize());
    if (queueSizeLimit >= 0 && AtomicGet(*queueSize) >= queueSizeLimit) {
        return;
    }
    AtomicAdd(*queueSize, count);
    for (int i = 0; i < count; ++i) {
        jobQueue->Enqueue(TSingleJob(rangeExec, 0));
    }
    HasJob.Signal();
}

NPar::TLocalExecutor::TLocalExecutor()
    : Impl_{MakeHolder<TImpl>()} {
}

NPar::TLocalExecutor::~TLocalExecutor() = default;

void NPar::TLocalExecutor::RunAdditionalThreads(int threadCount) {
    for (int i = 0; i < threadCount; i++)
        Impl_->RunNewThread();
}

void NPar::TLocalExecutor::Exec(ILocallyExecutable* exec, int id, int flags) {
    Y_ASSERT((flags & WAIT_COMPLETE) == 0); // unsupported
    int prior = Max<int>(Impl_->CurrentTaskPriority, flags & PRIORITY_MASK);
    switch (prior) {
        case HIGH_PRIORITY:
            AtomicAdd(Impl_->QueueSize, 1);
            Impl_->JobQueue.Enqueue(TSingleJob(exec, id));
            break;
        case MED_PRIORITY:
            AtomicAdd(Impl_->MPQueueSize, 1);
            Impl_->MedJobQueue.Enqueue(TSingleJob(exec, id));
            break;
        case LOW_PRIORITY:
            AtomicAdd(Impl_->LPQueueSize, 1);
            Impl_->LowJobQueue.Enqueue(TSingleJob(exec, id));
            break;
        default:
            Y_ASSERT(0);
            break;
    }
    Impl_->HasJob.Signal();
}

void NPar::TLocalExecutor::Exec(TLocallyExecutableFunction exec, int id, int flags) {
    Exec(new TFunctionWrapper(exec), id, flags);
}

void NPar::TLocalExecutor::ExecRange(ILocallyExecutable* exec, int firstId, int lastId, int flags) {
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
    int prior = Max<int>(Impl_->CurrentTaskPriority, flags & PRIORITY_MASK);
    switch (prior) {
        case HIGH_PRIORITY:
            Impl_->LaunchRange(rangeExec.Get(), queueSizeLimit, &Impl_->QueueSize, &Impl_->JobQueue);
            break;
        case MED_PRIORITY:
            Impl_->LaunchRange(rangeExec.Get(), queueSizeLimit, &Impl_->MPQueueSize, &Impl_->MedJobQueue);
            break;
        case LOW_PRIORITY:
            Impl_->LaunchRange(rangeExec.Get(), queueSizeLimit, &Impl_->LPQueueSize, &Impl_->LowJobQueue);
            break;
        default:
            Y_ASSERT(0);
            break;
    }
    if (flags & WAIT_COMPLETE) {
        int keepPrior = Impl_->CurrentTaskPriority;
        Impl_->CurrentTaskPriority = prior;
        while (rangeExec->DoSingleOp()) {
        }
        Impl_->CurrentTaskPriority = keepPrior;
        rangeExec->WaitComplete();
    }
}

void NPar::TLocalExecutor::ExecRange(TLocallyExecutableFunction exec, int firstId, int lastId, int flags) {
    ExecRange(new TFunctionWrapper(exec), firstId, lastId, flags);
}

void NPar::TLocalExecutor::ExecRangeWithThrow(TLocallyExecutableFunction exec, int firstId, int lastId, int flags) {
    Y_VERIFY((flags & WAIT_COMPLETE) != 0, "ExecRangeWithThrow() requires WAIT_COMPLETE to wait if exceptions arise.");
    TVector<NThreading::TFuture<void>> currentRun = ExecRangeWithFutures(exec, firstId, lastId, flags);
    for (auto& result : currentRun) {
        result.GetValueSync(); // Exception will be rethrown if exists. If several exception - only the one with minimal id is rethrown.
    }
}

TVector<NThreading::TFuture<void>>
NPar::TLocalExecutor::ExecRangeWithFutures(TLocallyExecutableFunction exec, int firstId, int lastId, int flags) {
    TFunctionWrapperWithPromise* execWrapper = new TFunctionWrapperWithPromise(exec, firstId, lastId);
    TVector<NThreading::TFuture<void>> out = execWrapper->GetFutures();
    ExecRange(execWrapper, firstId, lastId, flags);
    return out;
}

void NPar::TLocalExecutor::ClearLPQueue() {
    for (bool cont = true; cont;) {
        cont = false;
        TSingleJob job;
        while (Impl_->LowJobQueue.Dequeue(&job)) {
            AtomicAdd(Impl_->LPQueueSize, -1);
            cont = true;
        }
        while (Impl_->MedJobQueue.Dequeue(&job)) {
            AtomicAdd(Impl_->MPQueueSize, -1);
            cont = true;
        }
    }
}

int NPar::TLocalExecutor::GetQueueSize() const noexcept {
    return Impl_->QueueSize;
}

int NPar::TLocalExecutor::GetMPQueueSize() const noexcept {
    return Impl_->MPQueueSize;
}

int NPar::TLocalExecutor::GetLPQueueSize() const noexcept {
    return Impl_->LPQueueSize;
}

int NPar::TLocalExecutor::GetWorkerThreadId() noexcept {
    return Impl_->WorkerThreadId;
}

int NPar::TLocalExecutor::GetThreadCount() const noexcept {
    return Impl_->ThreadCount;
}

//////////////////////////////////////////////////////////////////////////
