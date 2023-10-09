#include "local_executor.h"

#include <library/cpp/threading/future/future.h>

#include <util/generic/utility.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/event.h>
#include <util/system/thread.h>
#include <util/system/tls.h>
#include <util/system/yield.h>
#include <util/thread/lfqueue.h>

#include <utility>

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
            : Exec(std::move(exec))
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
            : Exec(std::move(exec))
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
        TSingleJob(TIntrusivePtr<NPar::ILocallyExecutable> exec, int id)
            : Exec(std::move(exec))
            , Id(id)
        {
        }
    };

    class TLocalRangeExecutor: public NPar::ILocallyExecutable {
        TIntrusivePtr<NPar::ILocallyExecutable> Exec;
        alignas(64) TAtomic Counter;
        alignas(64) TAtomic WorkerCount;
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
        TLocalRangeExecutor(TIntrusivePtr<ILocallyExecutable> exec, int firstId, int lastId)
            : Exec(std::move(exec))
            , Counter(firstId)
            , WorkerCount(0)
            , LastId(lastId)
        {
        }
        bool DoSingleOp() {
            const int id = AtomicAdd(Counter, 1) - 1;
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

}

//////////////////////////////////////////////////////////////////////////
class NPar::TLocalExecutor::TImpl {
public:
    TLockFreeQueue<TSingleJob> JobQueue;
    TLockFreeQueue<TSingleJob> MedJobQueue;
    TLockFreeQueue<TSingleJob> LowJobQueue;
    alignas(64) TSystemEvent HasJob;

    TAtomic ThreadCount{0};
    alignas(64) TAtomic QueueSize{0};
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
    void LaunchRange(TIntrusivePtr<TLocalRangeExecutor> execRange, int queueSizeLimit,
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
    TThread::SetCurrentThreadName("ParLocalExecutor");
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

void NPar::TLocalExecutor::TImpl::LaunchRange(TIntrusivePtr<TLocalRangeExecutor> rangeExec,
                                              int queueSizeLimit,
                                              TAtomic* queueSize,
                                              TLockFreeQueue<TSingleJob>* jobQueue) {
    int count = Min<int>(ThreadCount + 1, rangeExec->GetRangeSize());
    if (queueSizeLimit >= 0 && AtomicGet(*queueSize) >= queueSizeLimit) {
        return;
    }
    AtomicAdd(*queueSize, count);
    jobQueue->EnqueueAll(TVector<TSingleJob>{size_t(count), TSingleJob(rangeExec, 0)});
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

void NPar::TLocalExecutor::Exec(TIntrusivePtr<ILocallyExecutable> exec, int id, int flags) {
    Y_ASSERT((flags & WAIT_COMPLETE) == 0); // unsupported
    int prior = Max<int>(Impl_->CurrentTaskPriority, flags & PRIORITY_MASK);
    switch (prior) {
        case HIGH_PRIORITY:
            AtomicAdd(Impl_->QueueSize, 1);
            Impl_->JobQueue.Enqueue(TSingleJob(std::move(exec), id));
            break;
        case MED_PRIORITY:
            AtomicAdd(Impl_->MPQueueSize, 1);
            Impl_->MedJobQueue.Enqueue(TSingleJob(std::move(exec), id));
            break;
        case LOW_PRIORITY:
            AtomicAdd(Impl_->LPQueueSize, 1);
            Impl_->LowJobQueue.Enqueue(TSingleJob(std::move(exec), id));
            break;
        default:
            Y_ASSERT(0);
            break;
    }
    Impl_->HasJob.Signal();
}

void NPar::ILocalExecutor::Exec(TLocallyExecutableFunction exec, int id, int flags) {
    Exec(new TFunctionWrapper(std::move(exec)), id, flags);
}

void NPar::TLocalExecutor::ExecRange(TIntrusivePtr<ILocallyExecutable> exec, int firstId, int lastId, int flags) {
    Y_ASSERT(lastId >= firstId);
    if (TryExecRangeSequentially([=] (int id) { exec->LocalExec(id); }, firstId, lastId, flags)) {
        return;
    }
    auto rangeExec = MakeIntrusive<TLocalRangeExecutor>(std::move(exec), firstId, lastId);
    int queueSizeLimit = (flags & WAIT_COMPLETE) ? 10000 : -1;
    int prior = Max<int>(Impl_->CurrentTaskPriority, flags & PRIORITY_MASK);
    switch (prior) {
        case HIGH_PRIORITY:
            Impl_->LaunchRange(rangeExec, queueSizeLimit, &Impl_->QueueSize, &Impl_->JobQueue);
            break;
        case MED_PRIORITY:
            Impl_->LaunchRange(rangeExec, queueSizeLimit, &Impl_->MPQueueSize, &Impl_->MedJobQueue);
            break;
        case LOW_PRIORITY:
            Impl_->LaunchRange(rangeExec, queueSizeLimit, &Impl_->LPQueueSize, &Impl_->LowJobQueue);
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

void NPar::ILocalExecutor::ExecRange(TLocallyExecutableFunction exec, int firstId, int lastId, int flags) {
    if (TryExecRangeSequentially(exec, firstId, lastId, flags)) {
        return;
    }
    ExecRange(new TFunctionWrapper(exec), firstId, lastId, flags);
}

void NPar::ILocalExecutor::ExecRangeWithThrow(TLocallyExecutableFunction exec, int firstId, int lastId, int flags) {
    Y_ABORT_UNLESS((flags & WAIT_COMPLETE) != 0, "ExecRangeWithThrow() requires WAIT_COMPLETE to wait if exceptions arise.");
    if (TryExecRangeSequentially(exec, firstId, lastId, flags)) {
        return;
    }
    TVector<NThreading::TFuture<void>> currentRun = ExecRangeWithFutures(exec, firstId, lastId, flags);
    for (auto& result : currentRun) {
        result.GetValueSync(); // Exception will be rethrown if exists. If several exception - only the one with minimal id is rethrown.
    }
}

TVector<NThreading::TFuture<void>>
NPar::ILocalExecutor::ExecRangeWithFutures(TLocallyExecutableFunction exec, int firstId, int lastId, int flags) {
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
    return AtomicGet(Impl_->QueueSize);
}

int NPar::TLocalExecutor::GetMPQueueSize() const noexcept {
    return AtomicGet(Impl_->MPQueueSize);
}

int NPar::TLocalExecutor::GetLPQueueSize() const noexcept {
    return AtomicGet(Impl_->LPQueueSize);
}

int NPar::TLocalExecutor::GetWorkerThreadId() const noexcept {
    return Impl_->WorkerThreadId;
}

int NPar::TLocalExecutor::GetThreadCount() const noexcept {
    return AtomicGet(Impl_->ThreadCount);
}

//////////////////////////////////////////////////////////////////////////
