#pragma once

#include <util/datetime/base.h>
#include <util/generic/function.h>
#include <util/generic/maybe.h> // TODO: remove
#include <util/generic/ptr.h>
#include <util/generic/singleton.h>
#include <util/generic/string.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/system/event.h>
#include <util/system/spinlock.h>

namespace NThreading {

////////////////////////////////////////////////////////////////////////////////

struct TFutureException: public yexception {};

template <typename T>
class TFuture;

template <typename T>
class TPromise;

// creates unset promise
template <typename T> TPromise<T> NewPromise();
TPromise<void> NewPromise();

// creates preset future
template <typename T> TFuture<T> MakeFuture(const T& value);
template <typename T> TFuture<std::remove_reference_t<T>> MakeFuture(T&& value);
template <typename T> TFuture<T> MakeFuture();
TFuture<void> MakeFuture();

// waits for all futures
TFuture<void> WaitAll(const TFuture<void>& f1);
TFuture<void> WaitAll(const TFuture<void>& f1, const TFuture<void>& f2);
template <typename TContainer> TFuture<void> WaitAll(const TContainer& futures);

// waits for any future
TFuture<void> WaitAny(const TFuture<void>& f1);
TFuture<void> WaitAny(const TFuture<void>& f1, const TFuture<void>& f2);
template <typename TContainer> TFuture<void> WaitAny(const TContainer& futures);

////////////////////////////////////////////////////////////////////////////////

namespace NImpl {
    template <typename T>
    class TFutureState;

    template <typename T>
    struct TFutureType {
        using TType = T;
    };

    template <typename T>
    struct TFutureType<TFuture<T>> {
        using TType = typename TFutureType<T>::TType;
    };
}   // namespace NImpl

template <typename F>
using TFutureType = typename NImpl::TFutureType<F>::TType;

////////////////////////////////////////////////////////////////////////////////

template <typename T>
class TFuture {
    using TFutureState = NImpl::TFutureState<T>;

private:
    TIntrusivePtr<TFutureState> State;

public:
    TFuture();
    TFuture(const TFuture<T>& other);
    TFuture(const TIntrusivePtr<TFutureState>& state);

    TFuture<T>& operator =(const TFuture<T>& other);
    void Swap(TFuture<T>& other);

    bool Initialized() const;

    bool HasValue() const;
    const T& GetValue(TDuration timeout = TDuration::Zero()) const;

    bool HasException() const;

    void Wait() const;
    bool Wait(TDuration timeout) const;
    bool Wait(TInstant deadline) const;

    template <typename F>
    const TFuture<T>& Subscribe(F&& callback) const;

    template <typename F>
    TFuture<TFutureType<TFunctionResult<F>>> Apply(F&& func) const;

    TFuture<void> IgnoreResult() const;

private:
    void EnsureInitialized() const;
};

////////////////////////////////////////////////////////////////////////////////

template <>
class TFuture<void> {
    using TFutureState = NImpl::TFutureState<void>;

private:
    TIntrusivePtr<TFutureState> State;

public:
    TFuture();
    TFuture(const TFuture<void>& other);
    TFuture(const TIntrusivePtr<TFutureState>& state);

    TFuture<void>& operator =(const TFuture<void>& other);
    void Swap(TFuture<void>& other);

    bool Initialized() const;

    bool HasValue() const;
    void GetValue(TDuration timeout = TDuration::Zero()) const;

    bool HasException() const;

    void Wait() const;
    bool Wait(TDuration timeout) const;
    bool Wait(TInstant deadline) const;

    template <typename F>
    const TFuture<void>& Subscribe(F&& callback) const;

    template <typename F>
    TFuture<TFutureType<TFunctionResult<F>>> Apply(F&& func) const;

    template <typename R>
    TFuture<R> Return(const R& value) const;

private:
    void EnsureInitialized() const;
};

////////////////////////////////////////////////////////////////////////////////

template <typename T>
class TPromise {
    using TFutureState = NImpl::TFutureState<T>;

private:
    TIntrusivePtr<TFutureState> State;

public:
    TPromise();
    TPromise(const TPromise<T>& other);
    TPromise(const TIntrusivePtr<TFutureState>& state);

    TPromise<T>& operator=(const TPromise<T>& other);
    void Swap(TPromise<T>& other);

    bool Initialized() const;

    bool HasValue() const;
    const T& GetValue() const;

    void SetValue(const T& value);
    void SetValue(T&& value);

    bool HasException() const;
    void SetException(const TString& e);
    void SetException(std::exception_ptr e);

    TFuture<T> GetFuture() const;
    operator TFuture<T>() const;

private:
    void EnsureInitialized() const;
};

////////////////////////////////////////////////////////////////////////////////

template <>
class TPromise<void> {
    using TFutureState = NImpl::TFutureState<void>;

private:
    TIntrusivePtr<TFutureState> State;

public:
    TPromise();
    TPromise(const TPromise<void>& other);
    TPromise(const TIntrusivePtr<TFutureState>& state);

    TPromise<void>& operator=(const TPromise<void>& other);
    void Swap(TPromise<void>& other);

    bool Initialized() const;

    bool HasValue() const;
    void GetValue() const;

    void SetValue();

    bool HasException() const;
    void SetException(const TString& e);
    void SetException(std::exception_ptr e);

    TFuture<void> GetFuture() const;
    operator TFuture<void>() const;

private:
    void EnsureInitialized() const;
};

}   // namespace NThreading

#define INCLUDE_FUTURE_INL_H
#include "future-inl.h"
#undef INCLUDE_FUTURE_INL_H
