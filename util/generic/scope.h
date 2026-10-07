#pragma once

#include <util/system/compiler.h>
#include <util/system/defaults.h>

#include <optional>
#include <type_traits>
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
} // namespace NPrivate

// \brief `Y_SCOPE_EXIT(captures) { body };`
//
// General implementaion of RAII idiom (resource acquisition is initialization). Executes code in the `body` block
// upon return from the current scope.
//
// @note expects `body` to provide no-throw guarantee, otherwise whenever an exception
// is thrown and leaves the outermost block of `body`, the function `std::terminate` is called.
// @see http://drdobbs.com/184403758 for detailed motivation.
#define Y_SCOPE_EXIT(...) const ::NPrivate::TScopeGuard Y_GENERATE_UNIQUE_ID(scopeGuard) Y_DECLARE_UNUSED = [__VA_ARGS__]() mutable -> void

// \brief `Y_DEFER { body };`
//
// Same as `Y_SCOPE_EXIT` but doesn't require user to provide capture-list explicitly (it
// implicitly uses `[&]` capture). Have same requirements for `body`.
//
// Inspired by `defer` statement in languages like Swift and Go.
//
// \code
// auto item = s.pop();
// bool ok = false;
// Y_DEFER { if (!ok) { s.push(std::move(item)); } };
// ... try handle `item` ...
// ok = true;
// \endcode
#define Y_DEFER Y_SCOPE_EXIT(&)

// A RAII scope guard.
// By default, invokes the provided callback when destroyed.
// The callback can be invoked earlier by `CallNow()` or cancelled by `Drop()`,
// in these cases the destructor becomes no-op.
template <typename F>
class TDeferredOnceFunction {
    static_assert(std::is_nothrow_invocable_v<F>);

public:
    TDeferredOnceFunction(const F& function)
        : Function_{function}
    {
    }

    TDeferredOnceFunction(F&& function)
        : Function_{std::move(function)}
    {
    }

    ~TDeferredOnceFunction() {
        if (Function_) {
            (*Function_)();
        }
    }

    TDeferredOnceFunction(const TDeferredOnceFunction&) = delete;
    TDeferredOnceFunction& operator=(const TDeferredOnceFunction&) = delete;

    TDeferredOnceFunction(TDeferredOnceFunction&& other) {
        // Cannot use Function_.swap because F may be not move-assignable (e.g. a lambda with captures)
        if (other.Function_.has_value()) {
            Function_.emplace(std::move(*other.Function_));
            other.Function_.reset();
        }
    }

    // Not sure if there are any valid usecases for assignment.
    TDeferredOnceFunction& operator=(TDeferredOnceFunction&&) = delete;

    void CallNow() && noexcept {
        (*Function_)();
        Function_.reset();
    }

    void Drop() && noexcept {
        Function_.reset();
    }

private:
    std::optional<F> Function_; // Not TMaybe, because scope.h is used in builds with disabled exceptions
};
