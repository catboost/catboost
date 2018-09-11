#pragma once

#include <util/system/compiler.h>
#include <util/system/defaults.h>

#include <utility>

namespace NPrivate {
    template <typename F>
    class TScopeGuard {
    public:
        TScopeGuard(const F& function)
            : Function_{function}
        {
        }

        TScopeGuard(F&& function)
            : Function_{std::move(function)}
        {
        }

        TScopeGuard(TScopeGuard&&) = default;
        TScopeGuard(const TScopeGuard&) = default;

        ~TScopeGuard() {
            Function_();
        }

    private:
        F Function_;
    };

    struct TMakeGuardHelper {
        template <class F>
        TScopeGuard<F> operator | (F&& function) const {
            return std::forward<F>(function);
        }
    };
}

#define Y_SCOPE_EXIT(...) const auto Y_GENERATE_UNIQUE_ID(scopeGuard) Y_DECLARE_UNUSED = ::NPrivate::TMakeGuardHelper{} | [__VA_ARGS__]() mutable -> void
#define Y_DEFER Y_SCOPE_EXIT(&)
