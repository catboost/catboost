#pragma once

#include <library/cpp/deprecated/atomic/atomic.h>

namespace NAtomic {
    class TBool {
    public:
        TBool() noexcept = default;

        TBool(bool val) noexcept
            : Val_(val)
        {
        }

        TBool(const TBool& src) noexcept {
            AtomicSet(Val_, AtomicGet(src.Val_));
        }

        operator bool() const noexcept {
            return AtomicGet(Val_);
        }

        const TBool& operator=(bool val) noexcept {
            AtomicSet(Val_, val);
            return *this;
        }

        const TBool& operator=(const TBool& src) noexcept {
            AtomicSet(Val_, AtomicGet(src.Val_));
            return *this;
        }

    private:
        TAtomic Val_ = 0;
    };
}
