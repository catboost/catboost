#pragma once

#include <library/cpp/threading/future/future.h>

#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/system/mutex.h>

#include <functional>
#include <optional>
#include <utility>

namespace NThreading {

namespace NPrivate {

struct TNoexceptExecutor {
    template <typename T, typename F>
    void operator()(TFuture<T> const& future, F&& callee) const noexcept {
        return callee(future);
    }
};

}

class TSubscriptionManager;

using TSubscriptionManagerPtr = TIntrusivePtr<TSubscriptionManager>;

//! A subscription id
class TSubscriptionId {
private:
    TFutureStateId StateId_;
    ui64 SubId_; // Secondary id to make the whole subscription id unique

    friend class TSubscriptionManager;

public:
    TFutureStateId StateId() const noexcept {
        return StateId_;
    }

    ui64 SubId() const noexcept {
        return SubId_;
    }

private:
    TSubscriptionId(TFutureStateId stateId, ui64 subId)
        : StateId_(stateId)
        , SubId_(subId)
    {
    }

    void SetSubId(ui64 subId) noexcept {
        SubId_ = subId;
    }
};

bool operator==(TSubscriptionId const& l, TSubscriptionId const& r) noexcept;
bool operator!=(TSubscriptionId const& l, TSubscriptionId const& r) noexcept;

//! The subscription manager manages subscriptions to futures
/** It provides an ability to create (and drop) multiple subscriptions to any future
    with just a single underlying subscription per future.

    When a future is signaled all its subscriptions are removed.
    So, there no need to call Unsubscribe for subscriptions to already signaled futures.

    Warning!!! For correct operation this class imposes the following requirement to futures/promises:
    Any used future must be signaled (value or exception set) before the future state destruction.
    Otherwise subscriptions and futures may happen.
    Current future design does not provide the required guarantee. But that should be fixed soon.
**/
class TSubscriptionManager final : public TAtomicRefCount<TSubscriptionManager> {
private:
    //! A single subscription
    class TSubscription {
    private:
        std::function<void()> Callback;

    public:
        template <typename T, typename F, typename TCallbackExecutor>
        TSubscription(TFuture<T> future, F&& callback, TCallbackExecutor&& executor);

        void operator()();
    };

    struct TFutureStateIdHash {
        size_t operator()(TFutureStateId const id) const noexcept {
            auto const value = id.Value();
            return ::hash<decltype(value)>()(value);
        }
    };

private:
    THashMap<TFutureStateId, THashMap<ui64, TSubscription>, TFutureStateIdHash> Subscriptions;
    ui64 Revision = 0;
    TMutex Lock;

public:
    //! Creates a new subscription manager instance
    static TSubscriptionManagerPtr NewInstance();

    //! The default subscription manager instance
    static TSubscriptionManagerPtr Default();

    //! Attempts to subscribe the callback to the future
    /** Subscription should succeed if the future is not signaled yet.
        Otherwise the callback will be called synchronously and nullopt will be returned

        @param future - The future to subscribe to
        @param callback - The callback to attach
        @return The subscription id on success, nullopt if the future has been signaled already
    **/
    template <typename T, typename F, typename TCallbackExecutor = NPrivate::TNoexceptExecutor>
    std::optional<TSubscriptionId> Subscribe(TFuture<T> const& future, F&& callback
                                                , TCallbackExecutor&& executor = NPrivate::TNoexceptExecutor());

    //! Drops the subscription with the given id
    /** @param id - The subscription id
    **/
    void Unsubscribe(TSubscriptionId id);

    //! Attempts to subscribe the callback to the set of futures
    /** @param futures - The futures to subscribe to
        @param callback - The callback to attach
        @param revertOnSignaled - Shows whether to stop and revert the subscription process if one of the futures is in signaled state
        @return The vector of subscription ids if no revert happened or an empty vector otherwise
                A subscription id will be valid even if a corresponding future has been signaled
    **/
    template <typename TFutures, typename F, typename TCallbackExecutor = NPrivate::TNoexceptExecutor>
    TVector<TSubscriptionId> Subscribe(TFutures const& futures, F&& callback, bool revertOnSignaled = false
                                                    , TCallbackExecutor&& executor = NPrivate::TNoexceptExecutor());

    //! Attempts to subscribe the callback to the set of futures
    /** @param futures - The futures to subscribe to
        @param callback - The callback to attach
        @param revertOnSignaled - Shows whether to stop and revert the subscription process if one of the futures is in signaled state
        @return The vector of subscription ids if no revert happened or an empty vector otherwise
                A subscription id will be valid even if a corresponding future has been signaled
    **/
    template <typename T, typename F, typename TCallbackExecutor = NPrivate::TNoexceptExecutor>
    TVector<TSubscriptionId> Subscribe(std::initializer_list<TFuture<T> const> futures, F&& callback, bool revertOnSignaled = false
                                                    , TCallbackExecutor&& executor = NPrivate::TNoexceptExecutor());

    //! Drops the subscriptions with the given ids
    /** @param ids - The subscription ids
    **/
    void Unsubscribe(TVector<TSubscriptionId> const& ids);

private:
    enum class ECallbackStatus {
        Subscribed, //! A subscription has been created. The callback will be called asynchronously.
        ExecutedSynchronously //! A callback has been called synchronously. No subscription has been created
    };

private:
    //! .ctor
    TSubscriptionManager() = default;
    //! Processes a callback from a future
    void OnCallback(TFutureStateId stateId) noexcept;
    //! Attempts to create a subscription
    /** This method should be called under the lock
    **/
    template <typename T, typename F, typename TCallbackExecutor>
    ECallbackStatus TrySubscribe(TFuture<T> const& future, F&& callback, TFutureStateId stateId, TCallbackExecutor&& executor);
    //! Batch subscribe implementation
    template <typename TFutures, typename F, typename TCallbackExecutor>
    TVector<TSubscriptionId> SubscribeImpl(TFutures const& futures, F const& callback, bool revertOnSignaled
                                                        , TCallbackExecutor const& executor);
    //! Unsubscribe implementation
    /** This method should be called under the lock
    **/
    void UnsubscribeImpl(TSubscriptionId id);
    //! Batch unsubscribe implementation
    /** This method should be called under the lock
    **/
    void UnsubscribeImpl(TVector<TSubscriptionId> const& ids);
};

}

#define INCLUDE_LIBRARY_THREADING_FUTURE_SUBSCRIPTION_INL_H
#include "subscription-inl.h"
#undef INCLUDE_LIBRARY_THREADING_FUTURE_SUBSCRIPTION_INL_H
