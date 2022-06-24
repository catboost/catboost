#include "rq.h"
#include "lfqueue.h"

#include <library/cpp/threading/atomic/bool.h>

#include <util/system/tls.h>
#include <util/system/pipe.h>
#include <util/system/event.h>
#include <util/system/mutex.h>
#include <util/system/condvar.h>
#include <util/system/guard.h>
#include <util/network/socket.h>
#include <util/generic/deque.h>

using namespace NNeh;

namespace {
    class TBaseLockFreeRequestQueue: public IRequestQueue {
    public:
        void Clear() override {
            IRequestRef req;
            while (Q_.Dequeue(&req)) {
            }
        }

    protected:
        NNeh::TAutoLockFreeQueue<IRequest> Q_;
    };

    class TFdRequestQueue: public TBaseLockFreeRequestQueue {
    public:
        inline TFdRequestQueue() {
            TPipeHandle::Pipe(R_, W_);
            SetNonBlock(W_);
        }

        void Schedule(IRequestRef req) override {
            Q_.Enqueue(req);
            char ch = 42;
            W_.Write(&ch, 1);
        }

        IRequestRef Next() override {
            IRequestRef ret;

#if 0
            for (size_t i = 0; i < 20; ++i) {
                if (Q_.Dequeue(&ret)) {
                    return ret;
                }

                //asm volatile ("pause;");
            }
#endif

            while (!Q_.Dequeue(&ret)) {
                char ch;

                R_.Read(&ch, 1);
            }

            return ret;
        }

    private:
        TPipeHandle R_;
        TPipeHandle W_;
    };

    struct TNehFdEvent {
        inline TNehFdEvent() {
            TPipeHandle::Pipe(R, W);
            SetNonBlock(W);
        }

        inline void Signal() noexcept {
            char ch = 21;
            W.Write(&ch, 1);
        }

        inline void Wait() noexcept {
            char buf[128];
            R.Read(buf, sizeof(buf));
        }

        TPipeHandle R;
        TPipeHandle W;
    };

    template <class TEvent>
    class TEventRequestQueue: public TBaseLockFreeRequestQueue {
    public:
        void Schedule(IRequestRef req) override {
            Q_.Enqueue(req);
            E_.Signal();
        }

        IRequestRef Next() override {
            IRequestRef ret;

            while (!Q_.Dequeue(&ret)) {
                E_.Wait();
            }

            E_.Signal();

            return ret;
        }

    private:
        TEvent E_;
    };

    template <class TEvent>
    class TLazyEventRequestQueue: public TBaseLockFreeRequestQueue {
    public:
        void Schedule(IRequestRef req) override {
            Q_.Enqueue(req);
            if (C_.Val()) {
                E_.Signal();
            }
        }

        IRequestRef Next() override {
            IRequestRef ret;

            C_.Inc();
            while (!Q_.Dequeue(&ret)) {
                E_.Wait();
            }
            C_.Dec();

            if (Q_.Size() && C_.Val()) {
                E_.Signal();
            }

            return ret;
        }

    private:
        TEvent E_;
        TAtomicCounter C_;
    };

    class TCondVarRequestQueue: public IRequestQueue {
    public:
        void Clear() override {
            TGuard<TMutex> g(M_);
            Q_.clear();
        }

        void Schedule(IRequestRef req) override {
            {
                TGuard<TMutex> g(M_);

                Q_.push_back(req);
            }

            C_.Signal();
        }

        IRequestRef Next() override {
            TGuard<TMutex> g(M_);

            while (Q_.empty()) {
                C_.Wait(M_);
            }

            IRequestRef ret = *Q_.begin();
            Q_.pop_front();

            return ret;
        }

    private:
        TDeque<IRequestRef> Q_;
        TMutex M_;
        TCondVar C_;
    };

    class TBusyRequestQueue: public TBaseLockFreeRequestQueue {
    public:
        void Schedule(IRequestRef req) override {
            Q_.Enqueue(req);
        }

        IRequestRef Next() override {
            IRequestRef ret;

            while (!Q_.Dequeue(&ret)) {
            }

            return ret;
        }
    };

    class TSleepRequestQueue: public TBaseLockFreeRequestQueue {
    public:
        void Schedule(IRequestRef req) override {
            Q_.Enqueue(req);
        }

        IRequestRef Next() override {
            IRequestRef ret;

            while (!Q_.Dequeue(&ret)) {
                usleep(1);
            }

            return ret;
        }
    };

    struct TStupidEvent {
        inline TStupidEvent()
            : InWait(false)
        {
        }

        inline bool Signal() noexcept {
            const bool ret = InWait;
            Ev.Signal();

            return ret;
        }

        inline void Wait() noexcept {
            InWait = true;
            Ev.Wait();
            InWait = false;
        }

        TAutoEvent Ev;
        NAtomic::TBool InWait;
    };

    template <class TEvent>
    class TLFRequestQueue: public TBaseLockFreeRequestQueue {
        struct TLocalQueue: public TEvent {
        };

    public:
        void Schedule(IRequestRef req) override {
            Q_.Enqueue(req);

            for (TLocalQueue* lq = 0; FQ_.Dequeue(&lq) && !lq->Signal();) {
            }
        }

        IRequestRef Next() override {
            while (true) {
                IRequestRef ret;

                if (Q_.Dequeue(&ret)) {
                    return ret;
                }

                TLocalQueue* lq = LocalQueue();

                FQ_.Enqueue(lq);

                if (Q_.Dequeue(&ret)) {
                    TLocalQueue* besttry;

                    if (FQ_.Dequeue(&besttry)) {
                        if (besttry == lq) {
                            //huraay, get rid of spurious wakeup
                        } else {
                            FQ_.Enqueue(besttry);
                        }
                    }

                    return ret;
                }

                lq->Wait();
            }
        }

    private:
        static inline TLocalQueue* LocalQueue() noexcept {
            Y_POD_STATIC_THREAD(TLocalQueue*)
            lq((TLocalQueue*)0);

            if (!lq) {
                Y_STATIC_THREAD(TLocalQueue)
                slq;

                lq = &(TLocalQueue&)slq;
            }

            return lq;
        }

    private:
        TLockFreeStack<TLocalQueue*> FQ_;
    };
}

IRequestQueueRef NNeh::CreateRequestQueue() {
//return new TCondVarRequestQueue();
//return new TSleepRequestQueue();
//return new TBusyRequestQueue();
//return new TLFRequestQueue<TStupidEvent>();
#if defined(_freebsd_)
    return new TFdRequestQueue();
#endif
    //return new TFdRequestQueue();
    return new TLazyEventRequestQueue<TAutoEvent>();
    //return new TEventRequestQueue<TAutoEvent>();
    //return new TEventRequestQueue<TNehFdEvent>();
}
