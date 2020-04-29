#pragma once

#include "future.h"

#include <util/generic/function.h>
#include <util/thread/pool.h>

namespace NThreading {
    /**
 * @brief Asynchronously executes @arg func in @arg queue returning a future for the result.
 *
 * @arg func should be a callable object with signature T().
 * @arg queue where @arg will be executed
 * @returns For @arg func with signature T() the function returns TFuture<T> unless T is TFuture<U>.
 *          In this case the function returns TFuture<U>.
 *
 * If you want to use another queue for execution just write an overload, @see ExtensionExample
 * unittest.
 */
    template <typename Func>
    TFuture<TFutureType<TFunctionResult<Func>>> Async(Func&& func, IThreadPool& queue) {
        auto promise = NewPromise<TFutureType<TFunctionResult<Func>>>();
        auto lambda = [promise, func = std::forward<Func>(func)]() mutable {
            NImpl::SetValue(promise, func);
        };
        queue.SafeAddFunc(std::move(lambda));

        return promise.GetFuture();
    }

}
