#pragma once

#include "fwd.h"

#include <library/cpp/deprecated/atomic/atomic.h>

#include <util/datetime/base.h>
#include <util/generic/function.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/system/event.h>
#include <util/system/spinlock.h>

namespace NThreading {
    ////////////////////////////////////////////////////////////////////////////////

    struct TFutureException: public yexception {};

    // creates unset promise
    template <typename T>
    TPromise<T> NewPromise();
    TPromise<void> NewPromise();

    // creates preset future
    template <typename T>
    TFuture<T> MakeFuture(const T& value);
    template <typename T>
    TFuture<std::remove_reference_t<T>> MakeFuture(T&& value);
    template <typename T>
    TFuture<T> MakeFuture();
    template <typename T>
    TFuture<T> MakeErrorFuture(std::exception_ptr exception);
    TFuture<void> MakeFuture();

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

        template <typename F, typename T>
        struct TFutureCallResult {
            // NOTE: separate class for msvc compatibility
            using TType = decltype(std::declval<F&>()(std::declval<const TFuture<T>&>()));
        };
    }

    template <typename F>
    using TFutureType = typename NImpl::TFutureType<F>::TType;

    template <typename F, typename T>
    using TFutureCallResult = typename NImpl::TFutureCallResult<F, T>::TType;

    //! Type of the future/promise state identifier
    class TFutureStateId;

    ////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    class TFuture {
        using TFutureState = NImpl::TFutureState<T>;

    private:
        TIntrusivePtr<TFutureState> State;

    public:
        using value_type = T;

        TFuture() noexcept = default;
        TFuture(const TFuture<T>& other) noexcept = default;
        TFuture(TFuture<T>&& other) noexcept = default;
        TFuture(const TIntrusivePtr<TFutureState>& state) noexcept;

        TFuture<T>& operator=(const TFuture<T>& other) noexcept = default;
        TFuture<T>& operator=(TFuture<T>&& other) noexcept = default;
        void Swap(TFuture<T>& other);

        bool Initialized() const;

        bool HasValue() const;
        const T& GetValue(TDuration timeout = TDuration::Zero()) const;
        const T& GetValueSync() const;
        T ExtractValue(TDuration timeout = TDuration::Zero());
        T ExtractValueSync();

        void TryRethrow() const;
        bool HasException() const;

        // returns true if exception or value was set.
        //   allows to check readiness without locking cheker-thread
        //   NOTE: returns true even if value was extracted from promise
        //   good replace for HasValue() || HasException()
        bool IsReady() const;

        void Wait() const;
        bool Wait(TDuration timeout) const;
        bool Wait(TInstant deadline) const;

        template <typename F>
        const TFuture<T>& Subscribe(F&& callback) const;

        // precondition: EnsureInitialized() passes
        // postcondition: std::terminate is highly unlikely
        template <typename F>
        const TFuture<T>& NoexceptSubscribe(F&& callback) const noexcept;

        template <typename F>
        TFuture<TFutureType<TFutureCallResult<F, T>>> Apply(F&& func) const;

        TFuture<void> IgnoreResult() const;

        //! If the future is initialized returns the future state identifier. Otherwise returns an empty optional
        /** The state identifier is guaranteed to be unique during the future state lifetime and could be reused after its death
        **/
        TMaybe<TFutureStateId> StateId() const noexcept;

        void EnsureInitialized() const;
    };

    ////////////////////////////////////////////////////////////////////////////////

    template <>
    class TFuture<void> {
        using TFutureState = NImpl::TFutureState<void>;

    private:
        TIntrusivePtr<TFutureState> State = nullptr;

    public:
        using value_type = void;

        TFuture() noexcept = default;
        TFuture(const TFuture<void>& other) noexcept = default;
        TFuture(TFuture<void>&& other) noexcept = default;
        TFuture(const TIntrusivePtr<TFutureState>& state) noexcept;

        TFuture<void>& operator=(const TFuture<void>& other) noexcept = default;
        TFuture<void>& operator=(TFuture<void>&& other) noexcept = default;
        void Swap(TFuture<void>& other);

        bool Initialized() const;

        bool HasValue() const;
        void GetValue(TDuration timeout = TDuration::Zero()) const;
        void GetValueSync() const;

        void TryRethrow() const;
        bool HasException() const;

        // returns true if exception or value was set.
        //   allows to check readiness without locking cheker-thread
        //   good replace for HasValue() || HasException()
        bool IsReady() const;

        void Wait() const;
        bool Wait(TDuration timeout) const;
        bool Wait(TInstant deadline) const;

        template <typename F>
        const TFuture<void>& Subscribe(F&& callback) const;

        // precondition: EnsureInitialized() passes
        // postcondition: std::terminate is highly unlikely
        template <typename F>
        const TFuture<void>& NoexceptSubscribe(F&& callback) const noexcept;

        template <typename F>
        TFuture<TFutureType<TFutureCallResult<F, void>>> Apply(F&& func) const;

        template <typename R>
        TFuture<std::remove_cvref_t<R>> Return(R&& value) const;

        TFuture<void> IgnoreResult() const {
            return *this;
        }

        //! If the future is initialized returns the future state identifier. Otherwise returns an empty optional
        /** The state identifier is guaranteed to be unique during the future state lifetime and could be reused after its death
        **/
        TMaybe<TFutureStateId> StateId() const noexcept;

        void EnsureInitialized() const;
    };

    ////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    class TPromise {
        using TFutureState = NImpl::TFutureState<T>;

    private:
        TIntrusivePtr<TFutureState> State = nullptr;

    public:
        TPromise() noexcept = default;
        TPromise(const TPromise<T>& other) noexcept = default;
        TPromise(TPromise<T>&& other) noexcept = default;
        TPromise(const TIntrusivePtr<TFutureState>& state) noexcept;

        TPromise<T>& operator=(const TPromise<T>& other) noexcept = default;
        TPromise<T>& operator=(TPromise<T>&& other) noexcept = default;
        void Swap(TPromise<T>& other);

        bool Initialized() const;

        bool HasValue() const;
        const T& GetValue() const;
        T ExtractValue();

        void SetValue(const T& value);
        void SetValue(T&& value);

        bool TrySetValue(const T& value);
        bool TrySetValue(T&& value);

        void TryRethrow() const;
        bool HasException() const;

        // returns true if exception or value was set.
        //   allows to check readiness without locking cheker-thread
        //   NOTE: returns true even if value was extracted from promise
        //   good replace for HasValue() || HasException()
        bool IsReady() const;
        void SetException(const TString& e);
        void SetException(std::exception_ptr e);
        bool TrySetException(std::exception_ptr e);

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
        TPromise() noexcept = default;
        TPromise(const TPromise<void>& other) noexcept = default;
        TPromise(TPromise<void>&& other) noexcept = default;
        TPromise(const TIntrusivePtr<TFutureState>& state) noexcept;

        TPromise<void>& operator=(const TPromise<void>& other) noexcept = default;
        TPromise<void>& operator=(TPromise<void>&& other) noexcept = default;
        void Swap(TPromise<void>& other);

        bool Initialized() const;

        bool HasValue() const;
        void GetValue() const;

        void SetValue();
        bool TrySetValue();

        void TryRethrow() const;
        bool HasException() const;

        // returns true if exception or value was set.
        //   allows to check readiness without locking cheker-thread
        //   good replace for HasValue() || HasException()
        bool IsReady() const;
        void SetException(const TString& e);
        void SetException(std::exception_ptr e);
        bool TrySetException(std::exception_ptr e);

        TFuture<void> GetFuture() const;
        operator TFuture<void>() const;

    private:
        void EnsureInitialized() const;
    };

}

#define INCLUDE_FUTURE_INL_H
#include "future-inl.h"
#undef INCLUDE_FUTURE_INL_H
