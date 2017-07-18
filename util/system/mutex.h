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

    ~TFakeMutex() = default;
};

class TSysMutex {
public:
    TSysMutex();
    TSysMutex(TSysMutex&&);
    ~TSysMutex();

    void Acquire() noexcept;
    bool TryAcquire() noexcept;
    void Release() noexcept;

    //return opaque pointer to real handler
    void* Handle() const noexcept;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

class TMutex: public TSysMutex {
};
