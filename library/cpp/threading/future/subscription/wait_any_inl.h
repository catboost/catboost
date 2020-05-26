#pragma once

#if !defined(INCLUDE_LIBRARY_THREADING_FUTURE_WAIT_ANY_INL_H)
#error "you should never include wait_any_inl.h directly"
#endif

#include "subscription.h"

#include <initializer_list>

namespace NThreading::NWait {

namespace NPrivate {

class TWaitAny final : public NThreading::NPrivate::TWait<TWaitAny> {
private:
    static constexpr bool RevertOnSignaled = true;

    using TBase = NThreading::NPrivate::TWait<TWaitAny>;
    friend TBase;

private:
    TWaitAny(TSubscriptionManagerPtr manager)
        : TBase(std::move(manager))
    {
    }

    template <typename TFutures>
    void BeforeSubscribe(TFutures const& futures) {
        Y_ENSURE(std::size(futures) > 0, "Futures set cannot be empty");
    }

    template <typename T>
    void Set(TFuture<T> const& future) {
        with_lock (TBase::Lock) {
            TBase::Unsubscribe();
            try {
                future.TryRethrow();
                TBase::Promise.TrySetValue();
            } catch (...) {
                TBase::Promise.TrySetException(std::current_exception());
            }
        }
    }
};

}

template <typename TFutures, typename TCallbackExecutor = NThreading::NPrivate::TNoexceptExecutor>
TFuture<void> WaitAny(TFutures const& futures, TSubscriptionManagerPtr manager, TCallbackExecutor&& executor) {
    return NThreading::NPrivate::Wait<NPrivate::TWaitAny>(futures, std::move(manager), std::forward<TCallbackExecutor>(executor));
}

template <typename T, typename TCallbackExecutor = NThreading::NPrivate::TNoexceptExecutor>
TFuture<void> WaitAny(std::initializer_list<TFuture<T> const> futures, TSubscriptionManagerPtr manager, TCallbackExecutor&& executor) {
    return NThreading::NPrivate::Wait<NPrivate::TWaitAny>(futures, std::move(manager), std::forward<TCallbackExecutor>(executor));
}

template <typename T, typename TCallbackExecutor = NThreading::NPrivate::TNoexceptExecutor>
TFuture<void> WaitAny(TFuture<T> const& future1, TFuture<T> const& future2, TSubscriptionManagerPtr manager, TCallbackExecutor&& executor) {
    return NThreading::NPrivate::Wait<NPrivate::TWaitAny>(future1, future2, std::move(manager), std::forward<TCallbackExecutor>(executor));
}

}
