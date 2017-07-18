#pragma once

#include "maybe.h"
#include "function.h"

template <class T>
class TLazyValue {
public:
    using TInitializer = std::function<T()>;

    template <typename... TArgs>
    TLazyValue(TArgs&&... args)
        : Initializer(std::forward<TArgs>(args)...)
    {
    }

    explicit operator bool() const noexcept {
        return Defined();
    }

    bool Defined() const noexcept {
        return ValueHolder.Defined();
    }

    const T& GetRef() const {
        if (!Defined()) {
            InitDefault();
        }
        return *ValueHolder;
    }

    const T& operator*() const {
        return GetRef();
    }

    const T* operator->() const {
        return &GetRef();
    }

    void InitDefault() const {
        Y_ASSERT(Initializer);
        ValueHolder = Initializer();
    }

private:
    mutable TMaybe<T> ValueHolder;
    TInitializer Initializer;
};

template <typename F>
TLazyValue<TFunctionResult<F>> MakeLazy(F&& f) {
    return {std::forward<F>(f)};
}
