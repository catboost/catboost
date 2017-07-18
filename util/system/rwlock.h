#pragma once

#include "guard.h"
#include "defaults.h"

#include <util/generic/ptr.h>

class TRWMutex {
public:
    TRWMutex();
    ~TRWMutex();

    void AcquireRead() noexcept;
    bool TryAcquireRead() noexcept;
    void ReleaseRead() noexcept;

    void AcquireWrite() noexcept;
    bool TryAcquireWrite() noexcept;
    void ReleaseWrite() noexcept;

    void Release() noexcept;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

template <class T>
struct TReadGuardOps {
    static inline void Acquire(T* t) noexcept {
        t->AcquireRead();
    }

    static inline void Release(T* t) noexcept {
        t->ReleaseRead();
    }
};

template <class T>
struct TTryReadGuardOps: public TReadGuardOps<T> {
    static inline bool TryAcquire(T* t) noexcept {
        return t->TryAcquireRead();
    }
};

template <class T>
struct TWriteGuardOps {
    static inline void Acquire(T* t) noexcept {
        t->AcquireWrite();
    }

    static inline void Release(T* t) noexcept {
        t->ReleaseWrite();
    }
};

template <class T>
struct TTryWriteGuardOps: public TWriteGuardOps<T> {
    static inline bool TryAcquire(T* t) noexcept {
        return t->TryAcquireWrite();
    }
};

template <class T>
using TReadGuardBase = TGuard<T, TReadGuardOps<T>>;
template <class T>
using TTryReadGuardBase = TTryGuard<T, TTryReadGuardOps<T>>;

template <class T>
using TWriteGuardBase = TGuard<T, TWriteGuardOps<T>>;
template <class T>
using TTryWriteGuardBase = TTryGuard<T, TTryWriteGuardOps<T>>;

using TReadGuard = TReadGuardBase<TRWMutex>;
using TTryReadGuard = TTryReadGuardBase<TRWMutex>;

using TWriteGuard = TWriteGuardBase<TRWMutex>;
using TTryWriteGuard = TTryWriteGuardBase<TRWMutex>;
