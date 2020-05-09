#pragma once

#include "lfqueue.h"

#include <library/cpp/threading/atomic/bool.h>

#include <util/generic/vector.h>
#include <util/system/atomic.h>
#include <util/system/atomic_ops.h>
#include <util/system/event.h>
#include <util/system/spinlock.h>

namespace NNeh {
    template <class T>
    class TBlockedQueue: public TLockFreeQueue<T>, public TSystemEvent {
    public:
        inline TBlockedQueue() noexcept
            : TSystemEvent(TSystemEvent::rAuto)
        {
        }

        inline void Notify(T t) noexcept {
            this->Enqueue(t);
            Signal();
        }
    };

    class TWaitQueue {
    public:
        struct TWaitHandle {
            inline TWaitHandle() noexcept
                : Signalled(false)
                , Parent(nullptr)
            {
            }

            inline void Signal() noexcept {
                TGuard<TSpinLock> lock(M_);

                Signalled = true;

                if (Parent) {
                    Parent->Notify(this);
                }
            }

            inline void Register(TWaitQueue* parent) noexcept {
                TGuard<TSpinLock> lock(M_);

                Parent = parent;

                if (Signalled) {
                    if (Parent) {
                        Parent->Notify(this);
                    }
                }
            }

            NAtomic::TBool Signalled;
            TWaitQueue* Parent;
            TSpinLock M_;
        };

        inline ~TWaitQueue() {
            for (size_t i = 0; i < H_.size(); ++i) {
                H_[i]->Register(nullptr);
            }
        }

        inline void Register(TWaitHandle& ev) {
            H_.push_back(&ev);
            ev.Register(this);
        }

        template <class T>
        inline void Register(const T& ev) {
            Register(static_cast<TWaitHandle&>(*ev));
        }

        inline bool Wait(const TInstant& deadLine) noexcept {
            return Q_.WaitD(deadLine);
        }

        inline void Notify(TWaitHandle* wq) noexcept {
            Q_.Notify(wq);
        }

        inline bool Dequeue(TWaitHandle** wq) noexcept {
            return Q_.Dequeue(wq);
        }

    private:
        TBlockedQueue<TWaitHandle*> Q_;
        TVector<TWaitHandle*> H_;
    };

    typedef TWaitQueue::TWaitHandle TWaitHandle;

    template <class It, class T>
    static inline void WaitForMultipleObj(It b, It e, const TInstant& deadLine, T& func) {
        TWaitQueue hndl;

        while (b != e) {
            hndl.Register(*b++);
        }

        do {
            TWaitHandle* ret = nullptr;

            if (hndl.Dequeue(&ret)) {
                do {
                    func(ret);
                } while (hndl.Dequeue(&ret));

                return;
            }
        } while (hndl.Wait(deadLine));
    }

    struct TSignalled {
        inline TSignalled()
            : Signalled(false)
        {
        }

        inline void operator()(const TWaitHandle*) noexcept {
            Signalled = true;
        }

        bool Signalled;
    };

    static inline bool WaitForOne(TWaitHandle& wh, const TInstant& deadLine) {
        TSignalled func;

        WaitForMultipleObj(&wh, &wh + 1, deadLine, func);

        return func.Signalled;
    }
}
