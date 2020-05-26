#pragma once

#if !defined(INCLUDE_LIBRARY_THREADING_FUTURE_SUBSCRIPTION_INL_H)
#error "you should never include subscription-inl.h directly"
#endif

namespace NThreading {

namespace NPrivate {

template <typename T>
TFutureStateId CheckedStateId(TFuture<T> const& future) {
    auto const id = future.StateId();
    if (id.Defined()) {
        return *id;
    }
    ythrow TFutureException() << "Future state should be initialized";
}

}

template <typename T, typename F, typename TCallbackExecutor>
inline TSubscriptionManager::TSubscription::TSubscription(TFuture<T> future, F&& callback, TCallbackExecutor&& executor)
    : Callback(
            [future = std::move(future), callback = std::forward<F>(callback), executor = std::forward<TCallbackExecutor>(executor)]() mutable {
                executor(std::as_const(future), callback);
            })
{
}

template <typename T, typename F, typename TCallbackExecutor>
inline std::optional<TSubscriptionId> TSubscriptionManager::Subscribe(TFuture<T> const& future, F&& callback, TCallbackExecutor&& executor) {
    auto stateId = NPrivate::CheckedStateId(future);
    with_lock(Lock) {
        auto const status = TrySubscribe(future, std::forward<F>(callback), stateId, std::forward<TCallbackExecutor>(executor));
        switch (status) {
            case ECallbackStatus::Subscribed:
                return TSubscriptionId(stateId, Revision);
            case ECallbackStatus::ExecutedSynchronously:
                return {};
            default:
                Y_FAIL("Unexpected callback status");
        }
    }
}

template <typename TFutures, typename F, typename TCallbackExecutor>
inline TVector<TSubscriptionId> TSubscriptionManager::Subscribe(TFutures const& futures, F&& callback, bool revertOnSignaled
                                                                , TCallbackExecutor&& executor)
{
    return SubscribeImpl(futures, std::forward<F>(callback), revertOnSignaled, std::forward<TCallbackExecutor>(executor));
}

template <typename T, typename F, typename TCallbackExecutor>
inline TVector<TSubscriptionId> TSubscriptionManager::Subscribe(std::initializer_list<TFuture<T> const> futures, F&& callback
                                                                , bool revertOnSignaled, TCallbackExecutor&& executor)
{
    return SubscribeImpl(futures, std::forward<F>(callback), revertOnSignaled, std::forward<TCallbackExecutor>(executor));
}

template <typename T, typename F, typename TCallbackExecutor>
inline TSubscriptionManager::ECallbackStatus TSubscriptionManager::TrySubscribe(TFuture<T> const& future, F&& callback, TFutureStateId stateId
                                                                                , TCallbackExecutor&& executor)
{
    TSubscription subscription(future, std::forward<F>(callback), std::forward<TCallbackExecutor>(executor));
    auto const it = Subscriptions.find(stateId);
    auto const revision = ++Revision;
    if (it == std::end(Subscriptions)) {
        auto const success = Subscriptions.emplace(stateId, THashMap<ui64, TSubscription>{ { revision, std::move(subscription) } }).second;
        Y_VERIFY(success);
        auto self = TSubscriptionManagerPtr(this);
        future.Subscribe([self, stateId](TFuture<T> const&) { self->OnCallback(stateId); });
        if (Subscriptions.find(stateId) == std::end(Subscriptions)) {
            return ECallbackStatus::ExecutedSynchronously;
        }
    } else {
        Y_VERIFY(it->second.emplace(revision, std::move(subscription)).second);
    }
    return ECallbackStatus::Subscribed;
}

template <typename TFutures, typename F, typename TCallbackExecutor>
inline TVector<TSubscriptionId> TSubscriptionManager::SubscribeImpl(TFutures const& futures, F const& callback, bool revertOnSignaled
                                                                    , TCallbackExecutor const& executor)
{
    TVector<TSubscriptionId> results;
    results.reserve(std::size(futures));
    // resolve all state ids to minimize processing under the lock
    for (auto const& f : futures) {
        results.push_back(TSubscriptionId(NPrivate::CheckedStateId(f), 0));
    }
    with_lock(Lock) {
        size_t i = 0;
        for (auto const& f : futures) {
            auto& r = results[i];
            auto const status = TrySubscribe(f, callback, r.StateId(), executor);
            switch (status) {
                case ECallbackStatus::Subscribed:
                    break;
                case ECallbackStatus::ExecutedSynchronously:
                    if (revertOnSignaled) {
                        // revert
                        results.crop(i);
                        UnsubscribeImpl(results);
                        return {};
                    }
                    break;
                default:
                    Y_FAIL("Unexpected callback status");
            }
            r.SetSubId(Revision);
            ++i;
        }
    }
    return results;
}

}
