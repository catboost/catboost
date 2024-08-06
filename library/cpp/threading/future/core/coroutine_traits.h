#pragma once

#include <library/cpp/threading/future/future.h>

#include <coroutine>

template <typename... Args>
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

template <typename T, typename... Args>
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

    template <typename T, bool Extracting = false>
    struct TFutureAwaitable {
        NThreading::TFuture<T> Future;

        TFutureAwaitable(const NThreading::TFuture<T>& future) noexcept requires (!Extracting)
            : Future{future}
        {
        }

        TFutureAwaitable(NThreading::TFuture<T>&& future) noexcept
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
            if constexpr (Extracting && !std::is_same_v<T, void>) {  // Future<void> has only GetValue()
                return Future.ExtractValue();
            } else {
                return Future.GetValue();
            }
        }
    };

    template <typename T>
    using TExtractingFutureAwaitable = TFutureAwaitable<T, true>;

} // namespace NThreading

template <typename T>
auto operator co_await(const NThreading::TFuture<T>& future) noexcept {
    return NThreading::TFutureAwaitable{future};
}

template <typename T>
auto operator co_await(NThreading::TFuture<T>&& future) noexcept {
    // Not TExtractongFutureAwaitable, because TFuture works like std::shared_future.
    // auto value = co_await GetCachedFuture();
    // If GetCachedFuture stores a future in some cache and returns its copies,
    // then subsequent uses of co_await will return a moved-from value.
    return NThreading::TFutureAwaitable{std::move(future)};
}

namespace NThreading {

    template <typename T>
    auto AsAwaitable(const NThreading::TFuture<T>& fut) noexcept {
        return TFutureAwaitable(fut);
    }

    template <typename T>
    auto AsExtractingAwaitable(NThreading::TFuture<T>&& fut) noexcept {
        return TExtractingFutureAwaitable<T>(std::move(fut));
    }

} // namespace NThreading
