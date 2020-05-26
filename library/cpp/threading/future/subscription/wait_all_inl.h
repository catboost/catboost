#pragma once

#if !defined(INCLUDE_LIBRARY_THREADING_FUTURE_WAIT_ALL_INL_H)
#error "you should never include wait_all_inl.h directly"
#endif

#include "subscription.h"

#include <initializer_list>

namespace NThreading::NWait {

namespace NPrivate {

class TWaitAll final : public NThreading::NPrivate::TWait<TWaitAll> {
private:
    size_t Count = 0;
    std::exception_ptr Exception;

    static constexpr bool RevertOnSignaled = false;

    using TBase = NThreading::NPrivate::TWait<TWaitAll>;
    friend TBase;

private:
    TWaitAll(TSubscriptionManagerPtr manager)
        : TBase(std::move(manager))
        , Count(0)
        , Exception()
    {
    }

    template <typename TFutures>
    void BeforeSubscribe(TFutures const& futures) {
        Count = std::size(futures);
        Y_ENSURE(Count > 0, "It is meaningless to use this class with empty futures set");
    }

    template <typename T>
    void Set(TFuture<T> const& future) {
        with_lock (TBase::Lock) {
            if (!Exception) {
                try {
                    future.TryRethrow();
                } catch (...) {
                    Exception = std::current_exception();
                }
            }

            if (--Count == 0) {
                // there is no need to call Unsubscribe here since all futures are signaled
                Y_ASSERT(!TBase::Promise.HasValue() && !TBase::Promise.HasException());
                if (Exception) {
                    TBase::Promise.SetException(std::move(Exception));
                } else {
                    TBase::Promise.SetValue();
                }
            }
        }
    }
};

}

template <typename TFutures, typename TCallbackExecutor = NThreading::NPrivate::TNoexceptExecutor>
TFuture<void> WaitAll(TFutures const& futures, TSubscriptionManagerPtr manager, TCallbackExecutor&& executor) {
    return NThreading::NPrivate::Wait<NPrivate::TWaitAll>(futures, std::move(manager), std::forward<TCallbackExecutor>(executor));
}

template <typename T, typename TCallbackExecutor = NThreading::NPrivate::TNoexceptExecutor>
TFuture<void> WaitAll(std::initializer_list<TFuture<T> const> futures, TSubscriptionManagerPtr manager, TCallbackExecutor&& executor) {
    return NThreading::NPrivate::Wait<NPrivate::TWaitAll>(futures, std::move(manager), std::forward<TCallbackExecutor>(executor));
}

template <typename T, typename TCallbackExecutor = NThreading::NPrivate::TNoexceptExecutor>
TFuture<void> WaitAll(TFuture<T> const& future1, TFuture<T> const& future2, TSubscriptionManagerPtr manager, TCallbackExecutor&& executor) {
    return NThreading::NPrivate::Wait<NPrivate::TWaitAll>(future1, future2, std::move(manager), std::forward<TCallbackExecutor>(executor));
}

}
