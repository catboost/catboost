#pragma once

#include "subscription.h"

#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/system/spinlock.h>


#include <initializer_list>

namespace NThreading::NPrivate {

template <typename TDerived>
class TWait : public TThrRefBase {
private:
    TSubscriptionManagerPtr Manager;
    TVector<TSubscriptionId> Subscriptions;
    bool Unsubscribed = false;

protected:
    TAdaptiveLock Lock;
    TPromise<void> Promise;

public:
    template <typename TFutures, typename TCallbackExecutor>
    static TFuture<void> Make(TFutures const& futures, TSubscriptionManagerPtr manager, TCallbackExecutor&& executor) {
        TIntrusivePtr<TDerived> w(new TDerived(std::move(manager)));
        w->Subscribe(futures, std::forward<TCallbackExecutor>(executor));
        return w->Promise.GetFuture();
    }

protected:
    TWait(TSubscriptionManagerPtr manager)
        : Manager(std::move(manager))
        , Subscriptions()
        , Unsubscribed(false)
        , Lock()
        , Promise(NewPromise())
    {
        Y_ENSURE(Manager != nullptr);
    }

protected:
    //! Unsubscribes all existing subscriptions
    /** Lock should be acquired!
    **/
    void Unsubscribe() noexcept {
        if (Unsubscribed) {
            return;
        }
        Unsubscribe(Subscriptions);
        Subscriptions.clear();
    }

private:
    //! Performs a subscription to the given futures
    /** Lock should not be acquired!
        @param future - The futures to subscribe to
        @param callback - The callback to call for each future
    **/
    template <typename TFutures, typename TCallbackExecutor>
    void Subscribe(TFutures const& futures, TCallbackExecutor&& executor) {
        auto self = TIntrusivePtr<TDerived>(static_cast<TDerived*>(this));
        self->BeforeSubscribe(futures);
        auto callback = [self = std::move(self)](const auto& future) mutable {
            self->Set(future);
        };
        auto subscriptions = Manager->Subscribe(futures, callback, TDerived::RevertOnSignaled, std::forward<TCallbackExecutor>(executor));
        if (subscriptions.empty()) {
            return;
        }
        with_lock (Lock) {
            if (Unsubscribed) {
                Unsubscribe(subscriptions);
            } else {
                Subscriptions = std::move(subscriptions);
            }
        }
    }

    void Unsubscribe(TVector<TSubscriptionId>& subscriptions) noexcept {
        Manager->Unsubscribe(subscriptions);
        Unsubscribed = true;
    }
};

template <typename TWaiter, typename TFutures, typename TCallbackExecutor>
TFuture<void> Wait(TFutures const& futures, TSubscriptionManagerPtr manager, TCallbackExecutor&& executor) {
    switch (std::size(futures)) {
        case 0:
            return MakeFuture();
        case 1:
            return std::begin(futures)->IgnoreResult();
        default:
            return TWaiter::Make(futures, std::move(manager), std::forward<TCallbackExecutor>(executor));
    }
}

template <typename TWaiter, typename T, typename TCallbackExecutor>
TFuture<void> Wait(std::initializer_list<TFuture<T> const> futures, TSubscriptionManagerPtr manager, TCallbackExecutor&& executor) {
    switch (std::size(futures)) {
        case 0:
            return MakeFuture();
        case 1:
            return std::begin(futures)->IgnoreResult();
        default:
            return TWaiter::Make(futures, std::move(manager), std::forward<TCallbackExecutor>(executor));
    }
}


template <typename TWaiter, typename T, typename TCallbackExecutor>
TFuture<void> Wait(TFuture<T> const& future1, TFuture<T> const& future2, TSubscriptionManagerPtr manager, TCallbackExecutor&& executor) {
    return TWaiter::Make(std::initializer_list<TFuture<T> const>({ future1, future2 }), std::move(manager)
                            , std::forward<TCallbackExecutor>(executor));
}

}
