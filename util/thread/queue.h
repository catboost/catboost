#pragma once

#include "pool.h"

#include <util/system/yassert.h>
#include <util/system/defaults.h>
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
     * IMtpQueue::CreateThreadSpecificResource, then result of
     * IMtpQueue::CreateThreadSpecificResource is passed as threadSpecificResource
     * parameter.
     */
    virtual void Process(void* threadSpecificResource) = 0;
};

/**
 * Mighty class to add 'Pool' method to derived classes.
 * Useful only for creators of new queue classes.
 */
class TThreadPoolHolder {
public:
    TThreadPoolHolder() noexcept;

    inline TThreadPoolHolder(IThreadPool* pool) noexcept
        : Pool_(pool)
    {
    }

    inline ~TThreadPoolHolder() = default;

    inline IThreadPool* Pool() const noexcept {
        return Pool_;
    }

private:
    IThreadPool* Pool_;
};

using TThreadFunction = std::function<void()>;

/**
 * A queue processed simultaneously by several threads
 */
class IMtpQueue: public IThreadPool, public TNonCopyable {
public:
    ~IMtpQueue() override = default;

    /**
     * Safe versions of Add*() functions. Behave exactly like as non-safe
     * version of Add*(), but use exceptions instead returning false
     */
    void SafeAdd(IObjectInQueue* obj);
    void SafeAddFunc(TThreadFunction func);
    void SafeAddAndOwn(TAutoPtr<IObjectInQueue> obj);

    /**
     * Add object to queue, run ojb->Proccess in other threads.
     * Obj is not deleted after execution
     * @return true of obj is successfully added to queue
     * @return false if queue is full or shutting down
     */
    virtual bool Add(IObjectInQueue* obj) Y_WARN_UNUSED_RESULT = 0;
    bool AddFunc(TThreadFunction func) Y_WARN_UNUSED_RESULT;
    bool AddAndOwn(TAutoPtr<IObjectInQueue> obj) Y_WARN_UNUSED_RESULT;
    virtual void Start(size_t threadCount, size_t queueSizeLimit = 0) = 0;
    /** Wait for completion of all scheduled objects, and then exit */
    virtual void Stop() noexcept = 0;
    /** Number of tasks currently in queue */
    virtual size_t Size() const noexcept = 0;

public:
    /**
     * RAII wrapper for Create/DestroyThreadSpecificResource.
     * Useful only for implementers of new IMtpQueue queues.
     */
    class TTsr {
    public:
        inline TTsr(IMtpQueue* q)
            : Q_(q)
            , Data_(Q_->CreateThreadSpecificResource())
        {
        }

        inline ~TTsr() {
            try {
                Q_->DestroyThreadSpecificResource(Data_);
            } catch (...) {
            }
        }

        inline operator void*() noexcept {
            return Data_;
        }

    private:
        IMtpQueue* Q_;
        void* Data_;
    };

    /**
     * CreateThreadSpecificResource and DestroyThreadSpecificResource
     * called from internals of (TAdaptiveMtpQueue, TMtpQueue, ...) implementation,
     * not by user of IMtpQueue interface.
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
 * Single-threaded implementation of IMtpQueue, process tasks in same thread when
 * added.
 * Can be used to remove multithreading.
 */
class TFakeMtpQueue: public IMtpQueue {
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
class TMtpQueue: public IMtpQueue, public TThreadPoolHolder {
public:
    enum EBlocking {
        NonBlockingMode,
        BlockingMode
    };

    enum ECatching {
        NonCatchingMode,
        CatchingMode
    };

    TMtpQueue(EBlocking blocking = NonBlockingMode, ECatching catching = CatchingMode);
    TMtpQueue(IThreadPool* pool, EBlocking blocking = NonBlockingMode, ECatching catching = CatchingMode);
    ~TMtpQueue() override;

    bool Add(IObjectInQueue* obj) override Y_WARN_UNUSED_RESULT;
    /**
      * @param queueSizeLimit means "unlimited" when = 0
      * @param threadCount means "single thread" when = 0
      */
    void Start(size_t threadCount, size_t queueSizeLimit = 0) override;
    void Stop() noexcept override;
    size_t Size() const noexcept override;

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
class TAdaptiveMtpQueue: public IMtpQueue, public TThreadPoolHolder {
public:
    TAdaptiveMtpQueue();
    TAdaptiveMtpQueue(IThreadPool* pool);
    ~TAdaptiveMtpQueue() override;

    /**
     * If working thread waits task too long (more then interval parameter),
     * then the thread would be killed. Default value - infinity, all created threads
     * waits for new task forever, before Stop.
     */
    void SetMaxIdleTime(TDuration interval);

    bool Add(IObjectInQueue* obj) override Y_WARN_UNUSED_RESULT;
    /** @param thrnum, @param maxque are ignored */
    void Start(size_t thrnum, size_t maxque = 0) override;
    void Stop() noexcept override;
    size_t Size() const noexcept override;

    class TImpl;

private:
    THolder<TImpl> Impl_;
};

/** Behave like TMtpQueue or TAdaptiveMtpQueue, choosen by thrnum parameter of Start()  */
class TSimpleMtpQueue: public IMtpQueue, public TThreadPoolHolder {
public:
    TSimpleMtpQueue();
    TSimpleMtpQueue(IThreadPool* pool);
    ~TSimpleMtpQueue() override;

    bool Add(IObjectInQueue* obj) override Y_WARN_UNUSED_RESULT;
    /**
     * @parameter thrnum. If thrnum is 0, use TAdaptiveMtpQueue with small
     * SetMaxIdleTime interval parameter. if thrnum is not 0, use non-blocking TMtpQueue
     */
    void Start(size_t thrnum, size_t maxque = 0) override;
    void Stop() noexcept override;
    size_t Size() const noexcept override;

private:
    THolder<IMtpQueue> Slave_;
};

/**
 * Helper to override virtual functions Create/DestroyThreadSpecificResource
 * from IMtpQueue and implement them using functions with same name from
 * pointer to TSlave.
 */
template <class TQueue, class TSlave>
class TMtpQueueBinder: public TQueue {
public:
    inline TMtpQueueBinder(TSlave* slave)
        : Slave_(slave)
    {
    }

    template <class T1>
    inline TMtpQueueBinder(TSlave* slave, const T1& t1)
        : TQueue(t1)
        , Slave_(slave)
    {
    }

    inline TMtpQueueBinder(TSlave& slave)
        : Slave_(&slave)
    {
    }

    ~TMtpQueueBinder() override {
        try {
            this->Stop();
        } catch (...) {
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

inline void Delete(TAutoPtr<IMtpQueue> q) {
    if (q.Get()) {
        q->Stop();
    }
}

/** creates and starts TMtpQueue if threadsCount > 1, or TFakeMtpQueue otherwise  */
TAutoPtr<IMtpQueue> CreateMtpQueue(size_t threadsCount, size_t queueSizeLimit = 0);
