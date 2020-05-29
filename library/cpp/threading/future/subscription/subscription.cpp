#include "subscription.h"

namespace NThreading {

bool operator==(TSubscriptionId const& l, TSubscriptionId const& r) noexcept {
    return l.StateId() == r.StateId() && l.SubId() == r.SubId();
}

bool operator!=(TSubscriptionId const& l, TSubscriptionId const& r) noexcept {
    return !(l == r);
}

void TSubscriptionManager::TSubscription::operator()() {
    Callback();
}

TSubscriptionManagerPtr TSubscriptionManager::NewInstance() {
    return new TSubscriptionManager();
}

TSubscriptionManagerPtr TSubscriptionManager::Default() {
    static auto instance = NewInstance();
    return instance;
}

void TSubscriptionManager::Unsubscribe(TSubscriptionId id) {
    with_lock(Lock) {
        UnsubscribeImpl(id);
    }
}

void TSubscriptionManager::Unsubscribe(TVector<TSubscriptionId> const& ids) {
    with_lock(Lock) {
        UnsubscribeImpl(ids);
    }
}

void TSubscriptionManager::OnCallback(TFutureStateId stateId) noexcept {
    THashMap<ui64, TSubscription> subscriptions;
    with_lock(Lock) {
        auto const it = Subscriptions.find(stateId);
        Y_VERIFY(it != Subscriptions.end(), "The callback has been triggered more than once");
        subscriptions.swap(it->second);
        Subscriptions.erase(it);
    }
    for (auto& [_, subscription] : subscriptions) {
        subscription();
    }
}

void TSubscriptionManager::UnsubscribeImpl(TSubscriptionId id) {
    auto const it = Subscriptions.find(id.StateId());
    if (it == std::end(Subscriptions)) {
        return;
    }
    it->second.erase(id.SubId());
}

void TSubscriptionManager::UnsubscribeImpl(TVector<TSubscriptionId> const& ids) {
    for (auto const& id : ids) {
        UnsubscribeImpl(id);
    }
}

}
