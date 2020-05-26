#pragma once

#include "wait.h"

namespace NThreading::NWait {

template <typename TFutures, typename TCallbackExecutor>
TFuture<void> WaitAllOrException(TFutures const& futures, TSubscriptionManagerPtr manager = TSubscriptionManager::Default()
                                    , TCallbackExecutor&& executor = TCallbackExecutor());

template <typename T, typename TCallbackExecutor>
TFuture<void> WaitAllOrException(std::initializer_list<TFuture<T> const> futures
                                    , TSubscriptionManagerPtr manager = TSubscriptionManager::Default()
                                    , TCallbackExecutor&& executor = TCallbackExecutor());

template <typename T, typename TCallbackExecutor>
TFuture<void> WaitAllOrException(TFuture<T> const& future1, TFuture<T> const& future2
                                    , TSubscriptionManagerPtr manager = TSubscriptionManager::Default()
                                    , TCallbackExecutor&& executor = TCallbackExecutor());

}

#define INCLUDE_LIBRARY_THREADING_FUTURE_WAIT_ALL_OR_EXCEPTION_INL_H
#include "wait_all_or_exception_inl.h"
#undef INCLUDE_LIBRARY_THREADING_FUTURE_WAIT_ALL_OR_EXCEPTION_INL_H
