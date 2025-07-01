#pragma once

#include <util/generic/string.h>
#include <util/generic/vector.h>

#include <coroutine>
#include <utility>

// Note: this namespace must be separate from NThreading and global namespace
namespace NSimpleTestTask {

    /**
     * The simplest possible coroutine type for testing
     */
    struct TSimpleTask {
        struct promise_type {
            TSimpleTask get_return_object() noexcept {
                return {};
            }
            std::suspend_never initial_suspend() noexcept {
                return {};
            }
            std::suspend_never final_suspend() noexcept {
                return {};
            }
            void unhandled_exception() {
                throw;
            }
            void return_void() noexcept {
            }
        };
    };

    /**
     * This coroutine awaits a generic awaitable
     *
     * When operator co_await for awaitables is not defined before this header
     * is included it will be searched using argument dependent lookup.
     */
    template <class TAwaitable>
    TSimpleTask RunAwaitable(TVector<TString>& result, TAwaitable&& awaitable) {
        result.push_back("before co_await");
        co_await std::forward<TAwaitable>(awaitable);
        result.push_back("after co_await");
    }

} // namespace NSimpleTestTask
