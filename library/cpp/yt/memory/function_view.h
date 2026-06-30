#pragma once

#include <library/cpp/yt/misc/concepts.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

// Non-owning type-erasure container.
/*
    Example:

    template <class T>
    class TSerializedObject
    {
    public:
        explicit TSerializedObject(T value)
            : Object_(value)
        { }

        void Lock(TFunctionView<void(const T&)> callback)
        {
            auto guard = Guard(SpinLock_);
            callback(Object_);
        }

    private:
        TSpinLock SpinLock_;
        T Object_;
    };

    int main()
    {
        TSerializedObject<int> object(42);

        // object.Lock([] (const int& value) {
        //     fmt::println("Value is {}", value);
        // });
        // ^ CE -- cannot pass rvalue.

        auto callback = [] (const int& value) {
            fmt::println("Value is {}", value);
        };

        object.Lock(callback); // <- prints "Value is 42".
    }
*/
template <class TSignature>
class TFunctionView;

////////////////////////////////////////////////////////////////////////////////

// TODO(arkady-e1ppa): Support pointer-to-member-function?
template <class T, class TSignature>
concept CTypeErasable =
    CInvocable<T, TSignature> &&
    (!std::same_as<T, TFunctionView<TSignature>>);

////////////////////////////////////////////////////////////////////////////////

template <class TResult, bool NoExcept, class... TArgs>
class TFunctionView<TResult(TArgs...) noexcept(NoExcept)>
{
public:
    using TSignature = TResult(TArgs...) noexcept(NoExcept);

    TFunctionView() = default;

    template <CTypeErasable<TSignature> TConcrete>
    TFunctionView(TConcrete& concreteRef) noexcept;

    template <CTypeErasable<TSignature> TConcrete>
    TFunctionView(TConcrete* concretePtr) noexcept;

    TResult operator()(TArgs... args) noexcept(NoExcept);

    explicit operator bool() const noexcept;

    TFunctionView Release() noexcept;

    bool IsValid() const noexcept;
    void Reset() noexcept;

    bool operator==(const TFunctionView& other) const & = default;

private:
    // NB: Technically, this is UB according to C standard, which
    // was not changed for C++ standard.
    // This is so because it is allowed to have
    // function pointers to be modelled by entities
    // different from object pointers.
    // No reasonable system architecture (e.g. x86 or ARM
    // or any other POSIX compliant one) does this.
    // No reasonable compiler (clang/gcc) does anything with this.
    // Accounting for such requirement would cause this class
    // to have std::variant-like storage which would make this class
    // weight more. Thus, we have decided to keep it this way,
    // since we are safe on x86 or ARM + clang.
    using TErasedPtr = void*;
    using TErasedInvoke = TResult(*)(TArgs..., TErasedPtr);

    TErasedPtr Ptr_ = nullptr;
    TErasedInvoke Invoke_ = nullptr;

    template <class TConcrete>
    static TResult ConcreteInvoke(TArgs... args, TErasedPtr ptr) noexcept(NoExcept);
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define FUNCTION_VIEW_INL_H_
#include "function_view-inl.h"
#undef FUNCTION_VIEW_INL_H_
