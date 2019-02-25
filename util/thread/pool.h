#pragma once

#include "fwd.h"
#include "factory.h"

#include <util/system/yassert.h>
#include <util/system/defaults.h>
#include <util/generic/yexception.h>
#include <util/generic/ptr.h>
#include <util/generic/noncopyable.h>
#include <functional>

class TDuration;

struct IObjectInQueue {
    virtual ~IObjectInQueue() = default;

    /**
     * Supposed to be implemented by user, to define jobs processed
     * in multiple threads.
     *
     * @param threadSpecificResource is nullptr by default. But if you override
     * IThreadPool::CreateThreadSpecificResource, then result of
     * IThreadPool::CreateThreadSpecificResource is passed as threadSpecificResource
     * parameter.
     */
    virtual void Process(void* threadSpecificResource) = 0;
};

/**
 * Mighty class to add 'Pool' method to derived classes.
 * Useful only for creators of new queue classes.
 */
class TThreadFactoryHolder {
public:
    TThreadFactoryHolder() noexcept;

    inline TThreadFactoryHolder(IThreadFactory* pool) noexcept
        : Pool_(pool)
    {
    }

    inline ~TThreadFactoryHolder() = default;

    inline IThreadFactory* Pool() const noexcept {
        return Pool_;
    }

private:
    IThreadFactory* Pool_;
};

class TThreadPoolException: public yexception {
};

template<class T>
class TThrFuncObj : public IObjectInQueue {
public:
    TThrFuncObj(const T& func)
        : Func(func) {
    }

    TThrFuncObj(T&& func)
        : Func(std::move(func)) {
    }

    void Process(void*) override {
        THolder<TThrFuncObj> self(this);
        Func();
    }

private:
    T Func;
};

template<class T>
IObjectInQueue* MakeThrFuncObj(T&& func) {
    return new TThrFuncObj<std::remove_cv_t<std::remove_reference_t<T>>>(std::forward<T>(func));
}

/**
 * A queue processed simultaneously by several threads
 */
class IThreadPool: public IThreadFactory, public TNonCopyable {
public:
    ~IThreadPool() override = default;

    /**
     * Safe versions of Add*() functions. Behave exactly like as non-safe
     * version of Add*(), but use exceptions instead returning false
     */
    void SafeAdd(IObjectInQueue* obj);

    template<class T>
    void SafeAddFunc(T&& func) {
        Y_ENSURE_EX(AddFunc(std::forward<T>(func)), TThreadPoolException() << AsStringBuf("can not add function to queue"));
    }

    void SafeAddAndOwn(TAutoPtr<IObjectInQueue> obj);

    /**
     * Add object to queue, run ojb->Proccess in other threads.
     * Obj is not deleted after execution
     * @return true of obj is successfully added to queue
     * @return false if queue is full or shutting down
     */
    virtual bool Add(IObjectInQueue* obj) Y_WARN_UNUSED_RESULT = 0;

    template<class T>
    Y_WARN_UNUSED_RESULT bool AddFunc(T&& func) {
        THolder<IObjectInQueue> wrapper(MakeThrFuncObj(std::forward<T>(func)));
        bool added = Add(wrapper.Get());
        if (added) {
            Y_UNUSED(wrapper.Release());
        }
        return added;
    }

    bool AddAndOwn(TAutoPtr<IObjectInQueue> obj) Y_WARN_UNUSED_RESULT;
    virtual void Start(size_t threadCount, size_t queueSizeLimit = 0) = 0;
    /** Wait for completion of all scheduled objects, and then exit */
    virtual void Stop() noexcept = 0;
    /** Number of tasks currently in queue */
    virtual size_t Size() const noexcept = 0;

public:
    /**
     * RAII wrapper for Create/DestroyThreadSpecificResource.
     * Useful only for implementers of new IThreadPool queues.
     */
    class TTsr {
    public:
        inline TTsr(IThreadPool* q)
            : Q_(q)
            , Data_(Q_->CreateThreadSpecificResource())
        {
        }

        inline ~TTsr() {
            try {
                Q_->DestroyThreadSpecificResource(Data_);
            } catch (...) {
                // ¯\_(ツ)_/¯
            }
        }

        inline operator void*() noexcept {
            return Data_;
        }

    private:
        IThreadPool* Q_;
        void* Data_;
    };

    /**
     * CreateThreadSpecificResource and DestroyThreadSpecificResource
     * called from internals of (TAdaptiveThreadPool, TThreadPool, ...) implementation,
     * not by user of IThreadPool interface.
     * Created resource is passed to IObjectInQueue::Proccess function.
     */
    virtual void* CreateThreadSpecificResource() {
        return nullptr;
    }

    virtual void DestroyThreadSpecificResource(void* resource) {
        if (resource != nullptr) {
            Y_ASSERT(resource == nullptr);
        }
    }

private:
    IThread* DoCreate() override;
};

/**
 * Single-threaded implementation of IThreadPool, process tasks in same thread when
 * added.
 * Can be used to remove multithreading.
 */
class TFakeThreadPool: public IThreadPool {
public:
    bool Add(IObjectInQueue* pObj) override Y_WARN_UNUSED_RESULT {
        TTsr tsr(this);
        pObj->Process(tsr);

        return true;
    }

    void Start(size_t, size_t = 0) override {
    }

    void Stop() noexcept override {
    }

    size_t Size() const noexcept override {
        return 0;
    }
};

/** queue processed by fixed size thread pool */
class TThreadPool: public IThreadPool, public TThreadFactoryHolder {
public:
    enum EBlocking {
        NonBlockingMode,
        BlockingMode
    };

    enum ECatching {
        NonCatchingMode,
        CatchingMode
    };

    TThreadPool(EBlocking blocking = NonBlockingMode, ECatching catching = CatchingMode);
    TThreadPool(IThreadFactory* pool, EBlocking blocking = NonBlockingMode, ECatching catching = CatchingMode);
    ~TThreadPool() override;

    bool Add(IObjectInQueue* obj) override Y_WARN_UNUSED_RESULT;
    /**
      * @param queueSizeLimit means "unlimited" when = 0
      * @param threadCount means "single thread" when = 0
      */
    void Start(size_t threadCount, size_t queueSizeLimit = 0) override;
    void Stop() noexcept override;
    size_t Size() const noexcept override;
    size_t GetThreadCountExpected() const noexcept;
    size_t GetThreadCountReal() const noexcept;
    size_t GetMaxQueueSize() const noexcept;

private:
    class TImpl;
    THolder<TImpl> Impl_;

    const EBlocking Blocking;
    const ECatching Catching;
};

/**
 * Always create new thread for new task, when all existing threads are busy.
 * Maybe dangerous, number of threads is not limited.
 */
class TAdaptiveThreadPool: public IThreadPool, public TThreadFactoryHolder {
public:
    TAdaptiveThreadPool();
    TAdaptiveThreadPool(IThreadFactory* pool);
    ~TAdaptiveThreadPool() override;

    /**
     * If working thread waits task too long (more then interval parameter),
     * then the thread would be killed. Default value - infinity, all created threads
     * waits for new task forever, before Stop.
     */
    void SetMaxIdleTime(TDuration interval);

    bool Add(IObjectInQueue* obj) override Y_WARN_UNUSED_RESULT;
    /** @param thrnum, @param maxque are ignored */
    void Start(size_t thrnum = 0, size_t maxque = 0) override;
    void Stop() noexcept override;
    size_t Size() const noexcept override;

    class TImpl;

private:
    THolder<TImpl> Impl_;
};

/** Behave like TThreadPool or TAdaptiveThreadPool, choosen by thrnum parameter of Start()  */
class TSimpleThreadPool: public IThreadPool, public TThreadFactoryHolder {
public:
    TSimpleThreadPool();
    TSimpleThreadPool(IThreadFactory* pool);
    ~TSimpleThreadPool() override;

    bool Add(IObjectInQueue* obj) override Y_WARN_UNUSED_RESULT;
    /**
     * @parameter thrnum. If thrnum is 0, use TAdaptiveThreadPool with small
     * SetMaxIdleTime interval parameter. if thrnum is not 0, use non-blocking TThreadPool
     */
    void Start(size_t thrnum, size_t maxque = 0) override;
    void Stop() noexcept override;
    size_t Size() const noexcept override;

private:
    THolder<IThreadPool> Slave_;
};

/**
 * Helper to override virtual functions Create/DestroyThreadSpecificResource
 * from IThreadPool and implement them using functions with same name from
 * pointer to TSlave.
 */
template <class TQueueType, class TSlave>
class TThreadPoolBinder: public TQueueType {
public:
    inline TThreadPoolBinder(TSlave* slave)
        : Slave_(slave)
    {
    }

    template <class T1>
    inline TThreadPoolBinder(TSlave* slave, const T1& t1)
        : TQueueType(t1)
        , Slave_(slave)
    {
    }

    inline TThreadPoolBinder(TSlave& slave)
        : Slave_(&slave)
    {
    }

    ~TThreadPoolBinder() override {
        try {
            this->Stop();
        } catch (...) {
            // ¯\_(ツ)_/¯
        }
    }

    void* CreateThreadSpecificResource() override {
        return Slave_->CreateThreadSpecificResource();
    }

    void DestroyThreadSpecificResource(void* resource) override {
        Slave_->DestroyThreadSpecificResource(resource);
    }

private:
    TSlave* Slave_;
};

inline void Delete(TAutoPtr<IThreadPool> q) {
    if (q.Get()) {
        q->Stop();
    }
}

/**
 * Creates and starts TThreadPool if threadsCount > 1, or TFakeThreadPool otherwise
 * You could specify blocking and catching modes for TThreadPool only
 */
TAutoPtr<IThreadPool> CreateThreadPool(size_t threadsCount, size_t queueSizeLimit = 0, TThreadPool::EBlocking blocking = TThreadPool::EBlocking::NonBlockingMode, TThreadPool::ECatching catching = TThreadPool::ECatching::CatchingMode);
