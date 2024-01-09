#include <atomic>

#include <util/system/defaults.h>

#if defined(_unix_)
    #include <pthread.h>
#endif

#include <util/generic/vector.h>
#include <util/generic/intrlist.h>
#include <util/generic/yexception.h>
#include <util/generic/ylimits.h>
#include <util/generic/singleton.h>
#include <util/generic/fastqueue.h>

#include <util/stream/output.h>
#include <util/string/builder.h>

#include <util/system/event.h>
#include <util/system/mutex.h>
#include <util/system/condvar.h>
#include <util/system/thread.h>

#include <util/datetime/base.h>

#include "factory.h"
#include "pool.h"

namespace {
    class TThreadNamer {
    public:
        TThreadNamer(const IThreadPool::TParams& params)
            : ThreadName(params.ThreadName_)
            , EnumerateThreads(params.EnumerateThreads_)
        {
        }

        explicit operator bool() const {
            return !ThreadName.empty();
        }

        void SetCurrentThreadName() {
            if (EnumerateThreads) {
                Set(TStringBuilder() << ThreadName << (Index++));
            } else {
                Set(ThreadName);
            }
        }

    private:
        void Set(const TString& name) {
            TThread::SetCurrentThreadName(name.c_str());
        }

    private:
        TString ThreadName;
        bool EnumerateThreads = false;
        std::atomic<ui64> Index{0};
    };
}

TThreadFactoryHolder::TThreadFactoryHolder() noexcept
    : Pool_(SystemThreadFactory())
{
}

class TThreadPool::TImpl: public TIntrusiveListItem<TImpl>, public IThreadFactory::IThreadAble {
    using TTsr = IThreadPool::TTsr;
    using TJobQueue = TFastQueue<IObjectInQueue*>;
    using TThreadRef = THolder<IThreadFactory::IThread>;

public:
    inline TImpl(TThreadPool* parent, size_t thrnum, size_t maxqueue, const TParams& params)
        : Parent_(parent)
        , Blocking(params.Blocking_)
        , Catching(params.Catching_)
        , Namer(params)
        , ShouldTerminate(true)
        , MaxQueueSize(0)
        , ThreadCountExpected(0)
        , ThreadCountReal(0)
        , Forked(false)
    {
        TAtforkQueueRestarter::Get().RegisterObject(this);
        Start(thrnum, maxqueue);
    }

    inline ~TImpl() override {
        try {
            Stop();
        } catch (...) {
            // ¯\_(ツ)_/¯
        }

        TAtforkQueueRestarter::Get().UnregisterObject(this);
        Y_ASSERT(Tharr.empty());
    }

    inline bool Add(IObjectInQueue* obj) {
        if (ShouldTerminate.load()) {
            return false;
        }

        if (Tharr.empty()) {
            TTsr tsr(Parent_);
            obj->Process(tsr);

            return true;
        }

        with_lock (QueueMutex) {
            while (MaxQueueSize > 0 && Queue.Size() >= MaxQueueSize && !ShouldTerminate.load()) {
                if (!Blocking) {
                    return false;
                }
                QueuePopCond.Wait(QueueMutex);
            }

            if (ShouldTerminate.load()) {
                return false;
            }

            Queue.Push(obj);
        }

        QueuePushCond.Signal();

        return true;
    }

    inline size_t Size() const noexcept {
        auto guard = Guard(QueueMutex);

        return Queue.Size();
    }

    inline size_t GetMaxQueueSize() const noexcept {
        return MaxQueueSize;
    }

    inline size_t GetThreadCountExpected() const noexcept {
        return ThreadCountExpected;
    }

    inline size_t GetThreadCountReal() const noexcept {
        return ThreadCountReal;
    }

    inline void AtforkAction() noexcept Y_NO_SANITIZE("thread") {
        Forked = true;
    }

    inline bool NeedRestart() const noexcept {
        return Forked;
    }

private:
    inline void Start(size_t num, size_t maxque) {
        ShouldTerminate.store(false);
        MaxQueueSize = maxque;
        ThreadCountExpected = num;

        try {
            for (size_t i = 0; i < num; ++i) {
                Tharr.push_back(Parent_->Pool()->Run(this));
                ++ThreadCountReal;
            }
        } catch (...) {
            Stop();

            throw;
        }
    }

    inline void Stop() {
        ShouldTerminate.store(true);

        with_lock (QueueMutex) {
            QueuePopCond.BroadCast();
        }

        if (!NeedRestart()) {
            WaitForComplete();
        }

        Tharr.clear();
        ThreadCountExpected = 0;
        MaxQueueSize = 0;
    }

    inline void WaitForComplete() noexcept {
        with_lock (StopMutex) {
            while (ThreadCountReal) {
                with_lock (QueueMutex) {
                    QueuePushCond.Signal();
                }

                StopCond.Wait(StopMutex);
            }
        }
    }

    void DoExecute() override {
        THolder<TTsr> tsr(new TTsr(Parent_));

        if (Namer) {
            Namer.SetCurrentThreadName();
        }

        while (true) {
            IObjectInQueue* job = nullptr;

            with_lock (QueueMutex) {
                while (Queue.Empty() && !ShouldTerminate.load()) {
                    QueuePushCond.Wait(QueueMutex);
                }

                if (ShouldTerminate.load() && Queue.Empty()) {
                    tsr.Destroy();

                    break;
                }

                job = Queue.Pop();
            }

            QueuePopCond.Signal();

            if (Catching) {
                try {
                    try {
                        job->Process(*tsr);
                    } catch (...) {
                        Cdbg << "[mtp queue] " << CurrentExceptionMessage() << Endl;
                    }
                } catch (...) {
                    // ¯\_(ツ)_/¯
                }
            } else {
                job->Process(*tsr);
            }
        }

        FinishOneThread();
    }

    inline void FinishOneThread() noexcept {
        auto guard = Guard(StopMutex);

        --ThreadCountReal;
        StopCond.Signal();
    }

private:
    TThreadPool* Parent_;
    const bool Blocking;
    const bool Catching;
    TThreadNamer Namer;
    mutable TMutex QueueMutex;
    mutable TMutex StopMutex;
    TCondVar QueuePushCond;
    TCondVar QueuePopCond;
    TCondVar StopCond;
    TJobQueue Queue;
    TVector<TThreadRef> Tharr;
    std::atomic<bool> ShouldTerminate;
    size_t MaxQueueSize;
    size_t ThreadCountExpected;
    size_t ThreadCountReal;
    bool Forked;

    class TAtforkQueueRestarter {
    public:
        static TAtforkQueueRestarter& Get() {
            return *SingletonWithPriority<TAtforkQueueRestarter, 256>();
        }

        inline void RegisterObject(TImpl* obj) {
            auto guard = Guard(ActionMutex);

            RegisteredObjects.PushBack(obj);
        }

        inline void UnregisterObject(TImpl* obj) {
            auto guard = Guard(ActionMutex);

            obj->Unlink();
        }

    private:
        void ChildAction() {
            TTryGuard guard{ActionMutex};
            // If you get an error here, it means you've used fork(2) in multi-threaded environment and probably created thread pools often.
            // Don't use fork(2) in multi-threaded programs, don't create thread pools often.
            // The mutex is locked after fork iff the fork(2) call was concurrent with RegisterObject / UnregisterObject in another thread.
            Y_ABORT_UNLESS(guard.WasAcquired(), "Failed to acquire ActionMutex after fork");

            for (auto it = RegisteredObjects.Begin(); it != RegisteredObjects.End(); ++it) {
                it->AtforkAction();
            }
        }

        static void ProcessChildAction() {
            Get().ChildAction();
        }

        TIntrusiveList<TImpl> RegisteredObjects;
        TMutex ActionMutex;

    public:
        inline TAtforkQueueRestarter() {
#if defined(_bionic_)
//no pthread_atfork on android libc
#elif defined(_unix_)
            pthread_atfork(nullptr, nullptr, ProcessChildAction);
#endif
        }
    };
};

TThreadPool::~TThreadPool() = default;

size_t TThreadPool::Size() const noexcept {
    if (!Impl_.Get()) {
        return 0;
    }

    return Impl_->Size();
}

size_t TThreadPool::GetThreadCountExpected() const noexcept {
    if (!Impl_.Get()) {
        return 0;
    }

    return Impl_->GetThreadCountExpected();
}

size_t TThreadPool::GetThreadCountReal() const noexcept {
    if (!Impl_.Get()) {
        return 0;
    }

    return Impl_->GetThreadCountReal();
}

size_t TThreadPool::GetMaxQueueSize() const noexcept {
    if (!Impl_.Get()) {
        return 0;
    }

    return Impl_->GetMaxQueueSize();
}

bool TThreadPool::Add(IObjectInQueue* obj) {
    Y_ENSURE_EX(Impl_.Get(), TThreadPoolException() << TStringBuf("mtp queue not started"));

    if (Impl_->NeedRestart()) {
        Start(Impl_->GetThreadCountExpected(), Impl_->GetMaxQueueSize());
    }

    return Impl_->Add(obj);
}

void TThreadPool::Start(size_t thrnum, size_t maxque) {
    Impl_.Reset(new TImpl(this, thrnum, maxque, Params));
}

void TThreadPool::Stop() noexcept {
    Impl_.Destroy();
}

static std::atomic<long> MtpQueueCounter = 0;

class TAdaptiveThreadPool::TImpl {
public:
    class TThread: public IThreadFactory::IThreadAble {
    public:
        inline TThread(TImpl* parent)
            : Impl_(parent)
            , Thread_(Impl_->Parent_->Pool()->Run(this))
        {
        }

        inline ~TThread() override {
            Impl_->DecThreadCount();
        }

    private:
        void DoExecute() noexcept override {
            THolder<TThread> This(this);

            if (Impl_->Namer) {
                Impl_->Namer.SetCurrentThreadName();
            }

            {
                TTsr tsr(Impl_->Parent_);
                IObjectInQueue* obj;

                while ((obj = Impl_->WaitForJob()) != nullptr) {
                    if (Impl_->Catching) {
                        try {
                            try {
                                obj->Process(tsr);
                            } catch (...) {
                                Cdbg << Impl_->Name() << " " << CurrentExceptionMessage() << Endl;
                            }
                        } catch (...) {
                            // ¯\_(ツ)_/¯
                        }
                    } else {
                        obj->Process(tsr);
                    }
                }
            }
        }

    private:
        TImpl* Impl_;
        THolder<IThreadFactory::IThread> Thread_;
    };

    inline TImpl(TAdaptiveThreadPool* parent, const TParams& params)
        : Parent_(parent)
        , Catching(params.Catching_)
        , Namer(params)
        , ThrCount_(0)
        , AllDone_(false)
        , Obj_(nullptr)
        , Free_(0)
        , IdleTime_(TDuration::Max())
    {
        snprintf(Name_, sizeof(Name_), "[mtp queue %ld]", ++MtpQueueCounter);
    }

    inline ~TImpl() {
        Stop();
    }

    inline void SetMaxIdleTime(TDuration idleTime) {
        IdleTime_ = idleTime;
    }

    inline const char* Name() const noexcept {
        return Name_;
    }

    inline void Add(IObjectInQueue* obj) {
        with_lock (Mutex_) {
            while (Obj_ != nullptr) {
                CondFree_.Wait(Mutex_);
            }

            if (Free_ == 0) {
                AddThreadNoLock();
            }

            Obj_ = obj;

            Y_ENSURE_EX(!AllDone_, TThreadPoolException() << TStringBuf("adding to a stopped queue"));
        }

        CondReady_.Signal();
    }

    inline void AddThreads(size_t n) {
        with_lock (Mutex_) {
            while (n) {
                AddThreadNoLock();

                --n;
            }
        }
    }

    inline size_t Size() const noexcept {
        return ThrCount_.load();
    }

private:
    inline void IncThreadCount() noexcept {
        ++ThrCount_;
    }

    inline void DecThreadCount() noexcept {
        --ThrCount_;
    }

    inline void AddThreadNoLock() {
        IncThreadCount();

        try {
            new TThread(this);
        } catch (...) {
            DecThreadCount();

            throw;
        }
    }

    inline void Stop() noexcept {
        Mutex_.Acquire();

        AllDone_ = true;

        while (ThrCount_.load()) {
            Mutex_.Release();
            CondReady_.Signal();
            Mutex_.Acquire();
        }

        Mutex_.Release();
    }

    inline IObjectInQueue* WaitForJob() noexcept {
        Mutex_.Acquire();

        ++Free_;

        while (!Obj_ && !AllDone_) {
            if (!CondReady_.WaitT(Mutex_, IdleTime_)) {
                break;
            }
        }

        IObjectInQueue* ret = Obj_;
        Obj_ = nullptr;

        --Free_;

        Mutex_.Release();
        CondFree_.Signal();

        return ret;
    }

private:
    TAdaptiveThreadPool* Parent_;
    const bool Catching;
    TThreadNamer Namer;
    std::atomic<size_t> ThrCount_;
    TMutex Mutex_;
    TCondVar CondReady_;
    TCondVar CondFree_;
    bool AllDone_;
    IObjectInQueue* Obj_;
    size_t Free_;
    char Name_[64];
    TDuration IdleTime_;
};

TThreadPoolBase::TThreadPoolBase(const TParams& params)
    : TThreadFactoryHolder(params.Factory_)
    , Params(params)
{
}

#define DEFINE_THREAD_POOL_CTORS(type) \
    type::type(const TParams& params)  \
        : TThreadPoolBase(params)      \
    {                                  \
    }

DEFINE_THREAD_POOL_CTORS(TThreadPool)
DEFINE_THREAD_POOL_CTORS(TAdaptiveThreadPool)
DEFINE_THREAD_POOL_CTORS(TSimpleThreadPool)

TAdaptiveThreadPool::~TAdaptiveThreadPool() = default;

bool TAdaptiveThreadPool::Add(IObjectInQueue* obj) {
    Y_ENSURE_EX(Impl_.Get(), TThreadPoolException() << TStringBuf("mtp queue not started"));

    Impl_->Add(obj);

    return true;
}

void TAdaptiveThreadPool::Start(size_t, size_t) {
    Impl_.Reset(new TImpl(this, Params));
}

void TAdaptiveThreadPool::Stop() noexcept {
    Impl_.Destroy();
}

size_t TAdaptiveThreadPool::Size() const noexcept {
    if (Impl_.Get()) {
        return Impl_->Size();
    }

    return 0;
}

void TAdaptiveThreadPool::SetMaxIdleTime(TDuration interval) {
    Y_ENSURE_EX(Impl_.Get(), TThreadPoolException() << TStringBuf("mtp queue not started"));

    Impl_->SetMaxIdleTime(interval);
}

TSimpleThreadPool::~TSimpleThreadPool() {
    try {
        Stop();
    } catch (...) {
        // ¯\_(ツ)_/¯
    }
}

bool TSimpleThreadPool::Add(IObjectInQueue* obj) {
    Y_ENSURE_EX(Slave_.Get(), TThreadPoolException() << TStringBuf("mtp queue not started"));

    return Slave_->Add(obj);
}

void TSimpleThreadPool::Start(size_t thrnum, size_t maxque) {
    THolder<IThreadPool> tmp;
    TAdaptiveThreadPool* adaptive(nullptr);

    if (thrnum) {
        tmp.Reset(new TThreadPoolBinder<TThreadPool, TSimpleThreadPool>(this, Params));
    } else {
        adaptive = new TThreadPoolBinder<TAdaptiveThreadPool, TSimpleThreadPool>(this, Params);
        tmp.Reset(adaptive);
    }

    tmp->Start(thrnum, maxque);

    if (adaptive) {
        adaptive->SetMaxIdleTime(TDuration::Seconds(100));
    }

    Slave_.Swap(tmp);
}

void TSimpleThreadPool::Stop() noexcept {
    Slave_.Destroy();
}

size_t TSimpleThreadPool::Size() const noexcept {
    if (Slave_.Get()) {
        return Slave_->Size();
    }

    return 0;
}

namespace {
    class TOwnedObjectInQueue: public IObjectInQueue {
    private:
        THolder<IObjectInQueue> Owned;

    public:
        TOwnedObjectInQueue(THolder<IObjectInQueue> owned)
            : Owned(std::move(owned))
        {
        }

        void Process(void* data) override {
            THolder<TOwnedObjectInQueue> self(this);
            Owned->Process(data);
        }
    };
}

void IThreadPool::SafeAdd(IObjectInQueue* obj) {
    Y_ENSURE_EX(Add(obj), TThreadPoolException() << TStringBuf("can not add object to queue"));
}

void IThreadPool::SafeAddAndOwn(THolder<IObjectInQueue> obj) {
    Y_ENSURE_EX(AddAndOwn(std::move(obj)), TThreadPoolException() << TStringBuf("can not add to queue and own"));
}

bool IThreadPool::AddAndOwn(THolder<IObjectInQueue> obj) {
    auto owner = MakeHolder<TOwnedObjectInQueue>(std::move(obj));
    bool added = Add(owner.Get());
    if (added) {
        Y_UNUSED(owner.Release());
    }
    return added;
}

using IThread = IThreadFactory::IThread;
using IThreadAble = IThreadFactory::IThreadAble;

namespace {
    class TPoolThread: public IThread {
        class TThreadImpl: public IObjectInQueue, public TAtomicRefCount<TThreadImpl> {
        public:
            inline TThreadImpl(IThreadAble* func)
                : Func_(func)
            {
            }

            ~TThreadImpl() override = default;

            inline void WaitForStart() noexcept {
                StartEvent_.Wait();
            }

            inline void WaitForComplete() noexcept {
                CompleteEvent_.Wait();
            }

        private:
            void Process(void* /*tsr*/) override {
                TThreadImplRef This(this);

                {
                    StartEvent_.Signal();

                    try {
                        Func_->Execute();
                    } catch (...) {
                        // ¯\_(ツ)_/¯
                    }

                    CompleteEvent_.Signal();
                }
            }

        private:
            IThreadAble* Func_;
            TSystemEvent CompleteEvent_;
            TSystemEvent StartEvent_;
        };

        using TThreadImplRef = TIntrusivePtr<TThreadImpl>;

    public:
        inline TPoolThread(IThreadPool* parent)
            : Parent_(parent)
        {
        }

        ~TPoolThread() override {
            if (Impl_) {
                Impl_->WaitForStart();
            }
        }

    private:
        void DoRun(IThreadAble* func) override {
            TThreadImplRef impl(new TThreadImpl(func));

            Parent_->SafeAdd(impl.Get());
            Impl_.Swap(impl);
        }

        void DoJoin() noexcept override {
            if (Impl_) {
                Impl_->WaitForComplete();
                Impl_ = nullptr;
            }
        }

    private:
        IThreadPool* Parent_;
        TThreadImplRef Impl_;
    };
}

IThread* IThreadPool::DoCreate() {
    return new TPoolThread(this);
}

THolder<IThreadPool> CreateThreadPool(size_t threadsCount, size_t queueSizeLimit, const TThreadPoolParams& params) {
    THolder<IThreadPool> queue;
    if (threadsCount > 1) {
        queue.Reset(new TThreadPool(params));
    } else {
        queue.Reset(new TFakeThreadPool());
    }
    queue->Start(threadsCount, queueSizeLimit);
    return queue;
}
