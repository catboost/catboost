#pragma once

#include <util/generic/noncopyable.h>
#include <util/system/defaults.h>

template <class T>
struct TCommonLockOps {
    static inline void Acquire(T* t) noexcept {
        t->Acquire();
    }

    static inline void Release(T* t) noexcept {
        t->Release();
    }
};

template <class T>
struct TTryLockOps: public TCommonLockOps<T> {
    static inline bool TryAcquire(T* t) noexcept {
        return t->TryAcquire();
    }
};

//must be used with great care
template <class TOps>
struct TInverseLockOps: public TOps {
    template <class T>
    static inline void Acquire(T* t) noexcept {
        TOps::Release(t);
    }

    template <class T>
    static inline void Release(T* t) noexcept {
        TOps::Acquire(t);
    }
};

template <class T, class TOps = TCommonLockOps<T>>
class TGuard: public TNonCopyable {
public:
    inline TGuard(const T& t) noexcept {
        Init(&t);
    }

    inline TGuard(const T* t) noexcept {
        Init(t);
    }

    inline TGuard(TGuard&& g) noexcept
        : T_(g.T_)
    {
        g.T_ = nullptr;
    }

    inline ~TGuard() {
        Release();
    }

    inline void Release() noexcept {
        if (WasAcquired()) {
            TOps::Release(T_);
            T_ = nullptr;
        }
    }

    explicit inline operator bool() const noexcept {
        return WasAcquired();
    }

    inline bool WasAcquired() const noexcept {
        return T_ != nullptr;
    }

    inline T* GetMutex() const noexcept {
        return T_;
    }

private:
    inline void Init(const T* t) noexcept {
        T_ = const_cast<T*>(t);
        TOps::Acquire(T_);
    }

private:
    T* T_;
};

/*
 * {
 *     auto guard = Guard(Lock_);
 *     some code under guard
 * }
 */
template <class T>
static inline TGuard<T> Guard(const T& t) {
    return {&t};
}

/*
 * with_lock (Lock_) {
 *     some code under guard
 * }
 */
#define with_lock(X)                                              \
    if (auto Y_GENERATE_UNIQUE_ID(__guard) = ::Guard(X); false) { \
    } else

/*
 * auto guard = Guard(Lock_);
 * ... some code under lock
 * {
 *     auto unguard = Unguard(guard);
 *     ... some code not under lock
 * }
 * ... some code under lock
 */
template <class T, class TOps = TCommonLockOps<T>>
using TInverseGuard = TGuard<T, TInverseLockOps<TOps>>;

template <class T, class TOps>
static inline TInverseGuard<T, TOps> Unguard(const TGuard<T, TOps>& guard) {
    return {guard.GetMutex()};
}

template <class T>
static inline TInverseGuard<T> Unguard(const T& mutex) {
    return {&mutex};
}

template <class T, class TOps = TTryLockOps<T>>
class TTryGuard: public TNonCopyable {
public:
    inline TTryGuard(const T& t) noexcept {
        Init(&t);
    }

    inline TTryGuard(const T* t) noexcept {
        Init(t);
    }

    inline TTryGuard(TTryGuard&& g) noexcept
        : T_(g.T_)
    {
        g.T_ = nullptr;
    }

    inline ~TTryGuard() {
        Release();
    }

    inline void Release() noexcept {
        if (WasAcquired()) {
            TOps::Release(T_);
            T_ = nullptr;
        }
    }

    inline bool WasAcquired() const noexcept {
        return T_ != nullptr;
    }

    explicit inline operator bool() const noexcept {
        return WasAcquired();
    }

private:
    inline void Init(const T* t) noexcept {
        T_ = nullptr;
        T* tMutable = const_cast<T*>(t);
        if (TOps::TryAcquire(tMutable)) {
            T_ = tMutable;
        }
    }

private:
    T* T_;
};
