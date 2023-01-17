#pragma once

#include "lfqueue.h"

#include <library/cpp/threading/atomic/bool.h>

#include <util/generic/vector.h>
#include <util/generic/scope.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <library/cpp/deprecated/atomic/atomic_ops.h>
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
        class TWaitHandle {
        public:
            void Signal() noexcept {
                TGuard<TSpinLock> lock(M_);

                Signalled_ = true;

                if (Parent_) {
                    Parent_->Notify(this);
                }
            }

            void Register(TWaitQueue* parent) noexcept {
                TGuard<TSpinLock> lock(M_);

                Parent_ = parent;

                if (Signalled_) {
                    if (Parent_) {
                        Parent_->Notify(this);
                    }
                }
            }

            bool Signalled() const {
                return Signalled_;
            }

            void ResetState() {
                TGuard<TSpinLock> lock(M_);

                Signalled_ = false;
            }
        private:
            NAtomic::TBool Signalled_ = false;
            TWaitQueue* Parent_ = nullptr;
            TSpinLock M_;
        };

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
    };

    typedef TWaitQueue::TWaitHandle TWaitHandle;

    template <class T>
    static inline void WaitForMultipleObj(TWaitQueue& hndl, const TInstant& deadLine, T& func) {
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
        TWaitQueue hndl;
        wh.Register(&hndl);

        Y_DEFER {
            wh.Register(nullptr);
        };

        WaitForMultipleObj(hndl, deadLine, func);

        return func.Signalled;
    }
}
