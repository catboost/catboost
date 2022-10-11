#pragma once

#include "maybe.h"
#include "function.h"

template <class T>
class TLazyValueBase {
public:
    using TInitializer = std::function<T()>;

    TLazyValueBase() = default;

    TLazyValueBase(TInitializer initializer)
        : Initializer(std::move(initializer))
    {
    }

    bool WasLazilyInitialized() const noexcept {
        return ValueHolder.Defined();
    }

    const T& GetRef() const {
        if (!WasLazilyInitialized()) {
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

// we need this to get implicit construction TLazyValue from lambda
// and save default copy constructor and operator= for type TLazyValue
template <class T>
class TLazyValue: public TLazyValueBase<T> {
public:
    template <typename... TArgs>
    TLazyValue(TArgs&&... args)
        : TLazyValueBase<T>(std::forward<TArgs>(args)...)
    {
    }
};

template <typename F>
TLazyValue<TFunctionResult<F>> MakeLazy(F&& f) {
    return {std::forward<F>(f)};
}
