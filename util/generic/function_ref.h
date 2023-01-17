#pragma once

#include <util/generic/function.h>
#include <util/system/yassert.h>

#include <functional>

namespace NPrivate {

    template <typename Signature>
    struct TIsNoexcept;

    template <typename Ret, typename... Args>
    struct TIsNoexcept<Ret(Args...)> {
        static constexpr bool Value = false;
    };

    template <typename Ret, typename... Args>
    struct TIsNoexcept<Ret(Args...) noexcept> {
        static constexpr bool Value = true;
    };

} // namespace NPrivate

template <typename Signature, bool IsNoexcept = NPrivate::TIsNoexcept<Signature>::Value>
class TFunctionRef;

template <typename Ret, typename... Args, bool IsNoexcept>
class TFunctionRef<Ret(Args...) noexcept(IsNoexcept), IsNoexcept> {
public:
    using TSignature = Ret(Args...) noexcept(IsNoexcept);

private:
    union TErasedCallable {
        const void* Functor;
        void (*Function)();
    };
    using TProxy = Ret (*)(TErasedCallable callable, Args...);

    // Making this a lambda inside TFunctionRef ctor caused:
    // "error: cannot compile this forwarded non-trivially copyable parameter yet"
    // on clang-win-i686-release.
    //
    // Using correct noexcept specifiers here (noexcept(IsNoexcept)) caused miscompilation on clang:
    // https://github.com/llvm/llvm-project/issues/55280.
    template <typename Functor>
    static Ret InvokeErasedFunctor(TErasedCallable callable, Args... args) {
        auto& ref = *static_cast<const std::remove_reference_t<Functor>*>(callable.Functor);
        return static_cast<Ret>(std::invoke(ref, std::forward<Args>(args)...));
    }

    template <typename Function>
    static Ret InvokeErasedFunction(TErasedCallable callable, Args... args) {
        auto* function = reinterpret_cast<Function*>(callable.Function);
        return static_cast<Ret>(std::invoke(function, std::forward<Args>(args)...));
    }

    template <class F>
    static constexpr bool IsInvocableUsing = std::conditional_t<
        IsNoexcept,
        std::is_nothrow_invocable_r<Ret, F, Args...>,
        std::is_invocable_r<Ret, F, Args...>>::value;

    // clang-format off
    template <class Callable>
    static constexpr bool IsSuitableFunctor =
        IsInvocableUsing<Callable>
        && !std::is_function_v<Callable>
        && !std::is_same_v<std::remove_cvref_t<Callable>, TFunctionRef>;

    template <class Callable>
    static constexpr bool IsSuitableFunction =
        IsInvocableUsing<Callable>
        && std::is_function_v<Callable>;
    // clang-format on

public:
    // Function ref should not be default constructible.
    // While the function ref can have empty state (for example, Proxy_ == nullptr),
    // It does not make sense in common usage cases.
    TFunctionRef() = delete;

    // Construct function ref from a functor.
    template <typename Functor, typename = std::enable_if_t<IsSuitableFunctor<Functor>>>
    TFunctionRef(Functor&& functor) noexcept
        : Callable_{
              .Functor = std::addressof(functor),
          }
        , Proxy_{InvokeErasedFunctor<Functor>}
    {
    }

    // Construct function ref from a function pointer.
    template <typename Function, typename = std::enable_if_t<IsSuitableFunction<Function>>>
    TFunctionRef(Function* function) noexcept
        : Callable_{
              .Function = reinterpret_cast<void (*)()>(function),
          }
        , Proxy_{InvokeErasedFunction<Function>}
    {
    }

    // Copy ctors & assignment.
    // Just copy pointers.
    TFunctionRef(const TFunctionRef& rhs) noexcept = default;
    TFunctionRef& operator=(const TFunctionRef& rhs) noexcept = default;

    Ret operator()(Args... args) const noexcept(IsNoexcept) {
        return Proxy_(Callable_, std::forward<Args>(args)...);
    }

private:
    TErasedCallable Callable_;
    TProxy Proxy_ = nullptr;
};

namespace NPrivate {

    template <typename Callable, typename Signature = typename TCallableTraits<Callable>::TSignature>
    struct TIsNothrowInvocable;

    template <typename Callable, typename Ret, typename... Args>
    struct TIsNothrowInvocable<Callable, Ret(Args...)> {
        static constexpr bool IsNoexcept = std::is_nothrow_invocable_r_v<Ret, Callable, Args...>;
        using TSignature = Ret(Args...) noexcept(IsNoexcept);
    };

    template <typename Callable>
    struct TCallableTraitsWithNoexcept {
        using TSignature = typename TIsNothrowInvocable<Callable>::TSignature;
    };

} // namespace NPrivate

template <typename Callable>
TFunctionRef(Callable&&) -> TFunctionRef<typename NPrivate::TCallableTraitsWithNoexcept<Callable>::TSignature>;

template <typename Function>
TFunctionRef(Function*) -> TFunctionRef<Function>;
