#pragma once

#include <library/cpp/threading/future/future.h>

#include <coroutine>

template<typename... Args>
struct std::coroutine_traits<NThreading::TFuture<void>, Args...> {
    struct promise_type {

        NThreading::TFuture<void> get_return_object() {
            return Promise_.GetFuture();
        }

        std::suspend_never initial_suspend() { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }

        void unhandled_exception() {
            Promise_.SetException(std::current_exception());
        }

        void return_void() {
            Promise_.SetValue();
        }

    private:
        NThreading::TPromise<void> Promise_ = NThreading::NewPromise();
    };
};

template<typename T, typename... Args>
struct std::coroutine_traits<NThreading::TFuture<T>, Args...> {
    struct promise_type {
        NThreading::TFuture<T> get_return_object() {
            return Promise_.GetFuture();
        }

        std::suspend_never initial_suspend() { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }

        void unhandled_exception() {
            Promise_.SetException(std::current_exception());
        }

        void return_value(auto&& val) {
            Promise_.SetValue(std::forward<decltype(val)>(val));
        }

    private:
        NThreading::TPromise<T> Promise_ = NThreading::NewPromise<T>();
    };
};

namespace NThreading {

    template<typename T>
    struct TFutureAwaitable {
        NThreading::TFuture<T> Future;

        TFutureAwaitable(NThreading::TFuture<T> future) noexcept
            : Future{std::move(future)}
        {
        }

        bool await_ready() const noexcept {
            return Future.IsReady();
        }

        void await_suspend(auto h) noexcept {
            /*
             * This library assumes that resume never throws an exception.
             * This assumption is made due to the fact that the users of these library in most cases do not need to write their own coroutine handlers,
             * and all coroutine handlers provided by the library do not throw exception from resume.
             *
             * WARNING: do not change subscribe to apply or something other here, creating an extra future state degrades performance.
             */
            Future.NoexceptSubscribe(
                [h](auto) mutable noexcept {
                    h();
                }
            );
        }

        decltype(auto) await_resume() {
            return Future.GetValue();
        }
    };

} // namespace NThreading

template<typename T>
auto operator co_await(const NThreading::TFuture<T>& future) {
    return NThreading::TFutureAwaitable{future};
}

namespace NThreading {

    template<typename T>
    auto AsAwaitable(const NThreading::TFuture<T>& fut) {
        return operator co_await(fut);
    }

} // namespace NThreading
