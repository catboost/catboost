#pragma once

#include "wait.h"

namespace NThreading::NWait {

template <typename TFutures, typename TCallbackExecutor>
TFuture<void> WaitAll(TFutures const& futures, TSubscriptionManagerPtr manager = TSubscriptionManager::Default()
                        , TCallbackExecutor&& executor = TCallbackExecutor());

template <typename T, typename TCallbackExecutor>
TFuture<void> WaitAll(std::initializer_list<TFuture<T> const> futures, TSubscriptionManagerPtr manager = TSubscriptionManager::Default()
                        , TCallbackExecutor&& executor = TCallbackExecutor());

template <typename T, typename TCallbackExecutor>
TFuture<void> WaitAll(TFuture<T> const& future1, TFuture<T> const& future2, TSubscriptionManagerPtr manager = TSubscriptionManager::Default()
                        , TCallbackExecutor&& executor = TCallbackExecutor());

}

#define INCLUDE_LIBRARY_THREADING_FUTURE_WAIT_ALL_INL_H
#include "wait_all_inl.h"
#undef INCLUDE_LIBRARY_THREADING_FUTURE_WAIT_ALL_INL_H
