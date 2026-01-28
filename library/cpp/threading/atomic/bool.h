#pragma once

#include <atomic>

namespace NAtomic {
    class TBool {
    public:
        TBool() noexcept = default;

        TBool(bool val) noexcept
            : Val_(val)
        {
        }

        TBool(const TBool& src) noexcept
            : Val_(src.Val_.load())
        {
        }

        operator bool() const noexcept {
            return Val_.load();
        }

        const TBool& operator=(bool val) noexcept {
            Val_.store(val);
            return *this;
        }

        const TBool& operator=(const TBool& src) noexcept {
            Val_.store(src.Val_.load());
            return *this;
        }

    private:
        std::atomic<bool> Val_ = false;
    };
} // namespace NAtomic
