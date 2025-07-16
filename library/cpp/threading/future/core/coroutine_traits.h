#pragma once

#include <library/cpp/threading/future/future.h>

#include <coroutine>
#include <utility>

template <typename... Args>
struct std::coroutine_traits<NThreading::TFuture<void>, Args...> {
    struct promise_type {

        NThreading::TFuture<void> get_return_object() noexcept {
            return NThreading::TFuture<void>(State_);
        }

        struct TFinalSuspend {
            bool await_ready() noexcept { return false; }
            void await_resume() noexcept { /* never called */ }
            void await_suspend(std::coroutine_handle<promise_type> self) noexcept {
                auto state = std::move(self.promise().State_);
                // We must destroy the coroutine before running callbacks
                // This will make sure argument copies are destroyed before the caller is resumed
                self.destroy();
                state->RunCallbacks();
            }
        };

        std::suspend_never initial_suspend() noexcept { return {}; }
        TFinalSuspend final_suspend() noexcept { return {}; }

        void unhandled_exception() {
            bool success = State_->TrySetException(std::current_exception(), /* deferCallbacks */ true);
            Y_ASSERT(success && "value already set");
        }

        void return_void() {
            bool success = State_->TrySetValue(/* deferCallbacks */ true);
            Y_ASSERT(success && "value already set");
        }

    private:
        TIntrusivePtr<NThreading::NImpl::TFutureState<void>> State_{new NThreading::NImpl::TFutureState<void>()};
    };
};

template <typename T, typename... Args>
struct std::coroutine_traits<NThreading::TFuture<T>, Args...> {
    struct promise_type {

        static_assert(
            !std::derived_from<T, std::exception>,
            "TFuture<std::exception can not be used in coroutines"
        );

        NThreading::TFuture<T> get_return_object() noexcept {
            return NThreading::TFuture<T>(State_);
        }

        struct TFinalSuspend {
            bool await_ready() noexcept { return false; }
            void await_resume() noexcept { /* never called */ }
            void await_suspend(std::coroutine_handle<promise_type> self) noexcept {
                auto state = std::move(self.promise().State_);
                // We must destroy the coroutine before running callbacks
                // This will make sure argument copies are destroyed before the caller is resumed
                self.destroy();
                state->RunCallbacks();
            }
        };

        std::suspend_never initial_suspend() noexcept { return {}; }
        TFinalSuspend final_suspend() noexcept { return {}; }

        void unhandled_exception() {
            bool success = State_->TrySetException(std::current_exception(), /* deferCallbacks */ true);
            Y_ASSERT(success && "value already set");
        }

        template <typename E>
        requires std::derived_from<std::remove_cvref_t<E>, std::exception>
        void return_value(E&& err) {
            // Allow co_return std::exception instances in order to avoid stack unwinding
            bool success = State_->TrySetException(
                std::make_exception_ptr(std::forward<E>(err)),
                /* deferCallbacks */ true
            );
            Y_ASSERT(success && "value already set");
        }

        void return_value(auto&& val) {
            bool success = State_->TrySetValue(std::forward<decltype(val)>(val), /* deferCallbacks */ true);
            Y_ASSERT(success && "value already set");
        }

    private:
        TIntrusivePtr<NThreading::NImpl::TFutureState<T>> State_{new NThreading::NImpl::TFutureState<T>()};
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

    template <typename T>
    auto AsAwaitable(const NThreading::TFuture<T>& fut) noexcept {
        return TFutureAwaitable(fut);
    }

    template <typename T>
    auto AsExtractingAwaitable(NThreading::TFuture<T>&& fut) noexcept {
        return TExtractingFutureAwaitable<T>(std::move(fut));
    }

} // namespace NThreading
