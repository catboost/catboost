#pragma once

#include "guard.h"
#include "defaults.h"

#include <util/generic/ptr.h>
#include <util/generic/noncopyable.h>

class TFakeMutex: public TNonCopyable {
public:
    inline void Acquire() noexcept {
    }

    inline bool TryAcquire() noexcept {
        return true;
    }

    inline void Release() noexcept {
    }

    inline void lock() noexcept {
        Acquire();
    }

    inline bool try_lock() noexcept {
        return TryAcquire();
    }

    inline void unlock() noexcept {
        Release();
    }

    ~TFakeMutex() = default;
};

class TMutex {
public:
    TMutex();
    TMutex(TMutex&&) noexcept;
    ~TMutex();

    void Acquire() noexcept;
    bool TryAcquire() noexcept;
    void Release() noexcept;

    inline void lock() noexcept {
        Acquire();
    }

    inline bool try_lock() noexcept {
        return TryAcquire();
    }

    inline void unlock() noexcept {
        Release();
    }

    //return opaque pointer to real handler
    void* Handle() const noexcept;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};
