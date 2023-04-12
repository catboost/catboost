#pragma once

#include <util/generic/ptr.h>
#include <library/cpp/deprecated/atomic/atomic.h>

namespace NNeh {
    //limited emulation shared_ptr/weak_ptr from boost lib.
    //the main value means the weak_ptr functionality, else recommended use types from util/generic/ptr.h

    //smart pointer counter shared between shared and weak ptrs.
    class TSPCounted: public TThrRefBase {
    public:
        inline TSPCounted() noexcept
            : C_(0)
        {
        }

        inline void Inc() noexcept {
            AtomicIncrement(C_);
        }

        //return false if C_ already 0, else increment and return true
        inline bool TryInc() noexcept {
            for (;;) {
                intptr_t curVal(AtomicGet(C_));

                if (!curVal) {
                    return false;
                }

                intptr_t newVal(curVal + 1);

                if (AtomicCas(&C_, newVal, curVal)) {
                    return true;
                }
            }
        }

        inline intptr_t Dec() noexcept {
            return AtomicDecrement(C_);
        }

        inline intptr_t Value() const noexcept {
            return AtomicGet(C_);
        }

    private:
        TAtomic C_;
    };

    typedef TIntrusivePtr<TSPCounted> TSPCountedRef;

    class TWeakCount;

    class TSPCount {
    public:
        TSPCount(TSPCounted* c = nullptr) noexcept
            : C_(c)
        {
        }

        inline void Swap(TSPCount& r) noexcept {
            DoSwap(C_, r.C_);
        }

        inline size_t UseCount() const noexcept {
            if (!C_) {
                return 0;
            }
            return C_->Value();
        }

        inline bool operator!() const noexcept {
            return !C_;
        }

        inline TSPCounted* GetCounted() const noexcept {
            return C_.Get();
        }

        inline void Reset() noexcept {
            if (!!C_) {
                C_.Drop();
            }
        }

    protected:
        TIntrusivePtr<TSPCounted> C_;
    };

    class TSharedCount: public TSPCount {
    public:
        inline TSharedCount() noexcept {
        }

        /// @throws std::bad_alloc
        inline explicit TSharedCount(const TSharedCount& r)
            : TSPCount(r.C_.Get())
        {
            if (!!C_) {
                (C_->Inc());
            }
        }

        //'c' must exist and has already increased ref
        inline explicit TSharedCount(TSPCounted* c) noexcept
            : TSPCount(c)
        {
        }

    public:
        /// @throws std::bad_alloc
        inline void Inc() {
            if (!C_) {
                TSPCountedRef(new TSPCounted()).Swap(C_);
            }
            C_->Inc();
        }

        inline bool TryInc() noexcept {
            if (!C_) {
                return false;
            }
            return C_->TryInc();
        }

        inline intptr_t Dec() noexcept {
            if (!C_) {
                Y_ASSERT(0);
                return 0;
            }
            return C_->Dec();
        }

        void Drop() noexcept {
            C_.Drop();
        }

    protected:
        template <class Y>
        friend class TSharedPtrB;

        // 'c' MUST BE already incremented
        void Assign(TSPCounted* c) noexcept {
            TSPCountedRef(c).Swap(C_);
        }

    private:
        TSharedCount& operator=(const TSharedCount&); //disable
    };

    class TWeakCount: public TSPCount {
    public:
        inline TWeakCount() noexcept {
        }

        inline explicit TWeakCount(const TWeakCount& r) noexcept
            : TSPCount(r.GetCounted())
        {
        }

        inline explicit TWeakCount(const TSharedCount& r) noexcept
            : TSPCount(r.GetCounted())
        {
        }

    private:
        TWeakCount& operator=(const TWeakCount&); //disable
    };

    template <class T>
    class TWeakPtrB;

    template <class T>
    class TSharedPtrB {
    public:
        inline TSharedPtrB() noexcept
            : T_(nullptr)
        {
        }

        /// @throws std::bad_alloc
        inline TSharedPtrB(T* t)
            : T_(nullptr)
        {
            if (t) {
                THolder<T> h(t);
                C_.Inc();
                T_ = h.Release();
            }
        }

        inline TSharedPtrB(const TSharedPtrB<T>& r) noexcept
            : T_(r.T_)
            , C_(r.C_)
        {
            Y_ASSERT((!!T_ && !!C_.UseCount()) || (!T_ && !C_.UseCount()));
        }

        inline TSharedPtrB(const TWeakPtrB<T>& r) noexcept
            : T_(r.T_)
        {
            if (T_) {
                TSPCounted* spc = r.C_.GetCounted();

                if (spc && spc->TryInc()) {
                    C_.Assign(spc);
                } else { //obsolete ptr
                    T_ = nullptr;
                }
            }
        }

        inline ~TSharedPtrB() {
            Reset();
        }

        TSharedPtrB& operator=(const TSharedPtrB<T>& r) noexcept {
            TSharedPtrB<T>(r).Swap(*this);
            return *this;
        }

        TSharedPtrB& operator=(const TWeakPtrB<T>& r) noexcept {
            TSharedPtrB<T>(r).Swap(*this);
            return *this;
        }

        void Swap(TSharedPtrB<T>& r) noexcept {
            DoSwap(T_, r.T_);
            DoSwap(C_, r.C_);
            Y_ASSERT((!!T_ && !!UseCount()) || (!T_ && !UseCount()));
        }

        inline bool operator!() const noexcept {
            return !T_;
        }

        inline T* Get() noexcept {
            return T_;
        }

        inline T* operator->() noexcept {
            return T_;
        }

        inline T* operator->() const noexcept {
            return T_;
        }

        inline T& operator*() noexcept {
            return *T_;
        }

        inline T& operator*() const noexcept {
            return *T_;
        }

        inline void Reset() noexcept {
            if (T_) {
                if (C_.Dec() == 0) {
                    delete T_;
                }
                T_ = nullptr;
                C_.Drop();
            }
        }

        inline size_t UseCount() const noexcept {
            return C_.UseCount();
        }

    protected:
        template <class Y>
        friend class TWeakPtrB;

        T* T_;
        TSharedCount C_;
    };

    template <class T>
    class TWeakPtrB {
    public:
        inline TWeakPtrB() noexcept
            : T_(nullptr)
        {
        }

        inline TWeakPtrB(const TWeakPtrB<T>& r) noexcept
            : T_(r.T_)
            , C_(r.C_)
        {
        }

        inline TWeakPtrB(const TSharedPtrB<T>& r) noexcept
            : T_(r.T_)
            , C_(r.C_)
        {
        }

        TWeakPtrB& operator=(const TWeakPtrB<T>& r) noexcept {
            TWeakPtrB(r).Swap(*this);
            return *this;
        }

        TWeakPtrB& operator=(const TSharedPtrB<T>& r) noexcept {
            TWeakPtrB(r).Swap(*this);
            return *this;
        }

        inline void Swap(TWeakPtrB<T>& r) noexcept {
            DoSwap(T_, r.T_);
            DoSwap(C_, r.C_);
        }

        inline void Reset() noexcept {
            T_ = 0;
            C_.Reset();
        }

        inline size_t UseCount() const noexcept {
            return C_.UseCount();
        }

    protected:
        template <class Y>
        friend class TSharedPtrB;

        T* T_;
        TWeakCount C_;
    };

}
