#include <library/cpp/testing/gtest/gtest.h>
#include <library/cpp/testing/gtest_extensions/gtest_extensions.h>
#include <library/cpp/threading/future/future.h>
#include <library/cpp/threading/future/core/coroutine_traits.h>

#include <util/generic/vector.h>
#include <util/generic/scope.h>
#include <util/system/platform.h>


TEST(TestFutureTraits, Simple) {
    TVector<TString> result;

    auto coroutine1 = [&result]() -> NThreading::TFuture<size_t> {
        result.push_back("coroutine1");
        co_return 1;
    };

    NThreading::TPromise<size_t> coroutine2SuspendPromise = NThreading::NewPromise<size_t>();
    auto coroutine2 = [&result, coroutine2SuspendFuture = coroutine2SuspendPromise.GetFuture()]() -> NThreading::TFuture<size_t> {
        result.push_back("coroutine2");

        result.push_back("pre_coroutine2_suspend_future");
        size_t futureResult = co_await coroutine2SuspendFuture;
        result.push_back("post_coroutine2_suspend_future");

        co_return 2 + futureResult;
    };

    auto coroutineAll = [&]() -> NThreading::TFuture<size_t> {
        Y_DEFER {
            result.push_back("coroutine_all_destroy");
        };

        result.push_back("pre_coroutine1");
        size_t coroutine1Res = co_await coroutine1();
        result.push_back("post_coroutine1");

        result.push_back("pre_coroutine2");
        size_t coroutine2Res = co_await coroutine2();
        result.push_back("post_coroutine2");

        co_return coroutine1Res + coroutine2Res;
    };

    NThreading::TFuture<size_t> coroutineAllFuture = coroutineAll();
    EXPECT_FALSE(coroutineAllFuture.HasValue());
    EXPECT_FALSE(coroutineAllFuture.HasException());
    EXPECT_THAT(
        result,
        ::testing::ContainerEq(
            TVector<TString>({
                "pre_coroutine1",
                "coroutine1",
                "post_coroutine1",

                "pre_coroutine2",
                "coroutine2",
                "pre_coroutine2_suspend_future"
            })
        )
    );

    coroutine2SuspendPromise.SetValue(3u);
    EXPECT_TRUE(coroutineAllFuture.HasValue());
    EXPECT_EQ(coroutineAllFuture.GetValue(), 6u);
    EXPECT_THAT(
        result,
        ::testing::ContainerEq(
            TVector<TString>({
                "pre_coroutine1",
                "coroutine1",
                "post_coroutine1",

                "pre_coroutine2",
                "coroutine2",
                "pre_coroutine2_suspend_future",
                "post_coroutine2_suspend_future",
                "post_coroutine2",

                "coroutine_all_destroy"
            })
        )
    );
}

TEST(TestFutureTraits, Exception) {
    TVector<TString> result;

    auto coroutine1 = [&result]() -> NThreading::TFuture<size_t> {
        result.push_back("coroutine1");
        co_return 1;
    };

    auto coroutine2 = [&result]() -> NThreading::TFuture<size_t> {
        result.push_back("coroutine2");
        ythrow yexception() << "coroutine2 exception";
    };

    auto coroutineAll = [&]() -> NThreading::TFuture<size_t> {
        Y_DEFER {
            result.push_back("coroutine_all_destroy");
        };

        result.push_back("pre_coroutine1");
        size_t coroutine1Res = co_await coroutine1();
        result.push_back("post_coroutine1");

        result.push_back("pre_coroutine2");
        size_t coroutine2Res = co_await coroutine2();
        result.push_back("post_coroutine2");

        co_return coroutine1Res + coroutine2Res;
    };

    EXPECT_THROW_MESSAGE_HAS_SUBSTR(
        coroutineAll().GetValueSync(),
        yexception,
        "coroutine2 exception"
    );
    EXPECT_THAT(
        result,
        ::testing::ContainerEq(
            TVector<TString>({
                "pre_coroutine1",
                "coroutine1",
                "post_coroutine1",

                "pre_coroutine2",
                "coroutine2",

                "coroutine_all_destroy"
            })
        )
    );
}

TEST(TestFutureTraits, CrashOnExceptionInCoroutineHandlerResume) {
    EXPECT_DEATH(
        {
            struct TBadPromise;
            struct TBadCoroutine : std::coroutine_handle<TBadPromise> {
                using promise_type = TBadPromise;
            };

            struct TBadPromise {
                TBadCoroutine get_return_object() {
                    return {TBadCoroutine::from_promise(*this)};
                }

                std::suspend_never initial_suspend() noexcept {
                    return {};
                }
                std::suspend_never final_suspend() noexcept {
                    return {};
                }
                void return_void() {
                }
                void unhandled_exception() {
                    throw;
                }
            };

            auto badCoroutine = []() -> TBadCoroutine {
                ythrow yexception() << "bad coroutine exception";
            };
            // Sanity check
            EXPECT_THROW_MESSAGE_HAS_SUBSTR(
                badCoroutine(),
                yexception,
                "bad coroutine exception"
            );

            NThreading::TPromise<void> promise = NThreading::NewPromise<void>();
            auto badCoroutineWithFutureAwait = [future = promise.GetFuture()]() -> TBadCoroutine {
                co_await future;
                ythrow yexception() << "bad coroutine with future await exception";
            };

            badCoroutineWithFutureAwait();
            promise.SetValue();
        },
#if defined(_win_)
        ".*"
#else
        "bad coroutine with future await exception"
#endif
    );
}

TEST(ExtractingFutureAwaitable, Simple) {
    NThreading::TPromise<THolder<size_t>> suspendPromise = NThreading::NewPromise<THolder<size_t>>();
    auto coro = [](NThreading::TFuture<THolder<size_t>> future) -> NThreading::TFuture<THolder<size_t>> {
        auto value = co_await NThreading::AsExtractingAwaitable(std::move(future));
        co_return value;
    };

    NThreading::TFuture<THolder<size_t>> getHolder = coro(suspendPromise.GetFuture());
    EXPECT_FALSE(getHolder.HasValue());
    EXPECT_FALSE(getHolder.HasException());
    suspendPromise.SetValue(MakeHolder<size_t>(42));

    EXPECT_TRUE(getHolder.HasValue());
    auto holder = getHolder.ExtractValue();
    ASSERT_NE(holder, nullptr);
    EXPECT_EQ(*holder, 42u);
}
