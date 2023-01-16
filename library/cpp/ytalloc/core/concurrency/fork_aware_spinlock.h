#pragma once

#include <util/system/spinlock.h>

namespace NYT::NConcurrency {

////////////////////////////////////////////////////////////////////////////////

//! Wraps TSpinLock and additionally acquires a global read lock preventing
//! concurrent forks from happening.
class TForkAwareSpinLock
{
public:
    TForkAwareSpinLock() = default;
    TForkAwareSpinLock(const TForkAwareSpinLock&) = delete;
    TForkAwareSpinLock& operator =(const TForkAwareSpinLock&) = delete;

    void Acquire() noexcept;
    void Release() noexcept;

    bool IsLocked() noexcept;

    using TAtForkHandler = void(*)(void*);
    static void AtFork(
        void* cookie,
        TAtForkHandler prepare,
        TAtForkHandler parent,
        TAtForkHandler child);

private:
    TAdaptiveLock SpinLock_;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NConcurrency
