#pragma once

#if !defined(INCLUDE_LIBRARY_THREADING_FUTURE_WAIT_ALL_OR_EXCEPTION_INL_H)
#error "you should never include wait_all_or_exception_inl.h directly"
#endif

#include "subscription.h"

#include <initializer_list>

namespace NThreading::NWait {

namespace NPrivate {

class TWaitAllOrException final : public NThreading::NPrivate::TWait<TWaitAllOrException>
{
private:
    size_t Count = 0;

    static constexpr bool RevertOnSignaled = false;

    using TBase = NThreading::NPrivate::TWait<TWaitAllOrException>;
    friend TBase;

private:
    TWaitAllOrException(TSubscriptionManagerPtr manager)
        : TBase(std::move(manager))
        , Count(0)
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
            try {
                future.TryRethrow();
                if (--Count == 0) {
                    // there is no need to call Unsubscribe here since all futures are signaled
                    TBase::Promise.SetValue();
                }
            } catch (...) {
                Y_ASSERT(!TBase::Promise.HasValue());
                TBase::Unsubscribe();
                if (!TBase::Promise.HasException()) {
                    TBase::Promise.SetException(std::current_exception());
                }
            }
        }
    }
};

}

template <typename TFutures, typename TCallbackExecutor = NThreading::NPrivate::TNoexceptExecutor>
TFuture<void> WaitAllOrException(TFutures const& futures, TSubscriptionManagerPtr manager, TCallbackExecutor&& executor) {
    return NThreading::NPrivate::Wait<NPrivate::TWaitAllOrException>(futures, std::move(manager), std::forward<TCallbackExecutor>(executor));
}

template <typename T, typename TCallbackExecutor = NThreading::NPrivate::TNoexceptExecutor>
TFuture<void> WaitAllOrException(std::initializer_list<TFuture<T> const> futures, TSubscriptionManagerPtr manager
                                    , TCallbackExecutor&& executor)
{
    return NThreading::NPrivate::Wait<NPrivate::TWaitAllOrException>(futures, std::move(manager), std::forward<TCallbackExecutor>(executor));
}
template <typename T, typename TCallbackExecutor = NThreading::NPrivate::TNoexceptExecutor>
TFuture<void> WaitAllOrException(TFuture<T> const& future1, TFuture<T> const& future2, TSubscriptionManagerPtr manager
                                    , TCallbackExecutor&& executor)
{
    return NThreading::NPrivate::Wait<NPrivate::TWaitAllOrException>(future1, future2, std::move(manager)
                                                                        , std::forward<TCallbackExecutor>(executor));
}

}
