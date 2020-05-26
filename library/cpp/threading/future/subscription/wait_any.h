#pragma once

#include "wait.h"

namespace NThreading::NWait {

template <typename TFutures, typename TCallbackExecutor>
TFuture<void> WaitAny(TFutures const& futures, TSubscriptionManagerPtr manager = TSubscriptionManager::Default()
                        , TCallbackExecutor&& executor = TCallbackExecutor());

template <typename T, typename TCallbackExecutor>
TFuture<void> WaitAny(std::initializer_list<TFuture<T> const> futures, TSubscriptionManagerPtr manager = TSubscriptionManager::Default()
                        , TCallbackExecutor&& executor = TCallbackExecutor());

template <typename T, typename TCallbackExecutor>
TFuture<void> WaitAny(TFuture<T> const& future1, TFuture<T> const& future2, TSubscriptionManagerPtr manager = TSubscriptionManager::Default()
                        , TCallbackExecutor&& executor = TCallbackExecutor());

}

#define INCLUDE_LIBRARY_THREADING_FUTURE_WAIT_ANY_INL_H
#include "wait_any_inl.h"
#undef INCLUDE_LIBRARY_THREADING_FUTURE_WAIT_ANY_INL_H
