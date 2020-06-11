#pragma once

#include <util/thread/lfqueue.h>
#include <util/generic/ptr.h>

namespace NNeh {
    template <class T>
    class TAutoLockFreeQueue {
        struct TCounter : TAtomicCounter {
            inline void IncCount(const T* const&) {
                Inc();
            }

            inline void DecCount(const T* const&) {
                Dec();
            }
        };

    public:
        typedef TAutoPtr<T> TRef;

        inline ~TAutoLockFreeQueue() {
            TRef tmp;

            while (Dequeue(&tmp)) {
            }
        }

        inline bool Dequeue(TRef* t) {
            T* res = nullptr;

            if (Q_.Dequeue(&res)) {
                t->Reset(res);

                return true;
            }

            return false;
        }

        inline void Enqueue(TRef& t) {
            Q_.Enqueue(t.Get());
            Y_UNUSED(t.Release());
        }

        inline size_t Size() {
            return Q_.GetCounter().Val();
        }

    private:
        TLockFreeQueue<T*, TCounter> Q_;
    };
}
