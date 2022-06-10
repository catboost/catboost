#pragma once

#include "lfqueue.h"

#include <library/cpp/coroutine/engine/impl.h>
#include <library/cpp/coroutine/engine/network.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/pipe.h>

#ifdef _linux_
#include <sys/eventfd.h>
#endif

#if defined(_bionic_) && !defined(EFD_SEMAPHORE)
#define EFD_SEMAPHORE 1
#endif

namespace NNeh {
#ifdef _linux_
    class TSemaphoreEventFd {
    public:
        inline TSemaphoreEventFd() {
            F_ = eventfd(0, EFD_NONBLOCK | EFD_SEMAPHORE);
            if (F_ < 0) {
                ythrow TFileError() << "failed to create a eventfd";
            }
        }

        inline ~TSemaphoreEventFd() {
            close(F_);
        }

        inline size_t Acquire(TCont* c) {
            ui64 ev;
            return NCoro::ReadI(c, F_, &ev, sizeof ev).Processed();
        }

        inline void Release() {
            const static ui64 ev(1);
            (void)write(F_, &ev, sizeof ev);
        }

    private:
        int F_;
    };
#endif

    class TSemaphorePipe {
    public:
        inline TSemaphorePipe() {
            TPipeHandle::Pipe(S_[0], S_[1]);

            SetNonBlock(S_[0]);
            SetNonBlock(S_[1]);
        }

        inline size_t Acquire(TCont* c) {
            char ch;
            return NCoro::ReadI(c, S_[0], &ch, 1).Processed();
        }

        inline size_t Acquire(TCont* c, char* buff, size_t buflen) {
            return NCoro::ReadI(c, S_[0], buff, buflen).Processed();
        }

        inline void Release() {
            char ch = 13;
            S_[1].Write(&ch, 1);
        }

    private:
        TPipeHandle S_[2];
    };

    class TPipeQueueBase {
    public:
        inline void Enqueue(void* job) {
            Q_.Enqueue(job);
            S_.Release();
        }

        inline void* Dequeue(TCont* c, char* ch, size_t buflen) {
            void* ret = nullptr;

            while (!Q_.Dequeue(&ret) && S_.Acquire(c, ch, buflen)) {
            }

            return ret;
        }

        inline void* Dequeue() noexcept {
            void* ret = nullptr;

            Q_.Dequeue(&ret);

            return ret;
        }

    private:
        TLockFreeQueue<void*> Q_;
        TSemaphorePipe S_;
    };

    template <class T, size_t buflen = 1>
    class TPipeQueue {
    public:
        template <class TPtr>
        inline void EnqueueSafe(TPtr req) {
            Enqueue(req.Get());
            req.Release();
        }

        inline void Enqueue(T* req) {
            Q_.Enqueue(req);
        }

        template <class TPtr>
        inline void DequeueSafe(TCont* c, TPtr& ret) {
            ret.Reset(Dequeue(c));
        }

        inline T* Dequeue(TCont* c) {
            char ch[buflen];

            return (T*)Q_.Dequeue(c, ch, sizeof(ch));
        }

    protected:
        TPipeQueueBase Q_;
    };

    //optimized for avoiding unnecessary usage semaphore + use eventfd on linux
    template <class T>
    struct TOneConsumerPipeQueue {
        inline TOneConsumerPipeQueue()
            : Signaled_(0)
            , SkipWait_(0)
        {
        }

        inline void Enqueue(T* job) {
            Q_.Enqueue(job);

            AtomicSet(SkipWait_, 1);
            if (AtomicCas(&Signaled_, 1, 0)) {
                S_.Release();
            }
        }

        inline T* Dequeue(TCont* c) {
            T* ret = nullptr;

            while (!Q_.Dequeue(&ret)) {
                AtomicSet(Signaled_, 0);
                if (!AtomicCas(&SkipWait_, 0, 1)) {
                    if (!S_.Acquire(c)) {
                        break;
                    }
                }
                AtomicSet(Signaled_, 1);
            }

            return ret;
        }

        template <class TPtr>
        inline void EnqueueSafe(TPtr req) {
            Enqueue(req.Get());
            Y_UNUSED(req.Release());
        }

        template <class TPtr>
        inline void DequeueSafe(TCont* c, TPtr& ret) {
            ret.Reset(Dequeue(c));
        }

    protected:
        TLockFreeQueue<T*> Q_;
#ifdef _linux_
        TSemaphoreEventFd S_;
#else
        TSemaphorePipe S_;
#endif
        TAtomic Signaled_;
        TAtomic SkipWait_;
    };

    template <class T, size_t buflen = 1>
    struct TAutoPipeQueue: public TPipeQueue<T, buflen> {
        ~TAutoPipeQueue() {
            while (T* t = (T*)TPipeQueue<T, buflen>::Q_.Dequeue()) {
                delete t;
            }
        }
    };

    template <class T>
    struct TAutoOneConsumerPipeQueue: public TOneConsumerPipeQueue<T> {
        ~TAutoOneConsumerPipeQueue() {
            T* ret = nullptr;

            while (TOneConsumerPipeQueue<T>::Q_.Dequeue(&ret)) {
                delete ret;
            }
        }
    };
}
