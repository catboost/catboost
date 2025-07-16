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

TEST(TestFutureTraits, ErrorViaThrow) {
    TVector<TString> result;

    auto coroutineReturnValue = [&result]() -> NThreading::TFuture<size_t> {
        result.push_back("coroutine_return_value");
        co_return 1;
    };

    auto coroutineThrow = [&result]() -> NThreading::TFuture<size_t> {
        result.push_back("coroutine_throw");
        ythrow yexception() << "coroutine exception";
    };

    auto coroutineReturnValueThrow = [&]() -> NThreading::TFuture<size_t> {
        Y_DEFER {
            result.push_back("coroutine_all_destroy");
        };

        result.push_back("pre_coroutine_return_value");
        size_t res1 = co_await coroutineReturnValue();
        result.push_back("post_coroutine_return_value");

        result.push_back("pre_coroutine_throw");
        size_t res2 = co_await coroutineThrow();
        result.push_back("post_coroutine_throw");

        co_return res1 + res2;
    };

    EXPECT_THROW_MESSAGE_HAS_SUBSTR(
        coroutineReturnValueThrow().GetValueSync(),
        yexception,
        "coroutine exception"
    );
    EXPECT_THAT(
        result,
        ::testing::ContainerEq(
            TVector<TString>({
                "pre_coroutine_return_value",
                "coroutine_return_value",
                "post_coroutine_return_value",

                "pre_coroutine_throw",
                "coroutine_throw",

                "coroutine_all_destroy"
            })
        )
    );
}

TEST(TestFutureTraits, ErrorViaReturnException) {

    TVector<TString> result;

    auto coroutineReturnValue = [&result]() -> NThreading::TFuture<size_t> {
        result.push_back("coroutine_return_value");
        co_return 1;
    };

    auto coroutineReturnException = [&result]() -> NThreading::TFuture<size_t> {
        result.push_back("coroutine_return_exception");
        co_return std::runtime_error("exception_to_return");

        static std::runtime_error another("another_exception_not_to_return");
        co_return another;
    };

    auto coroutineReturnValueReturnException = [&]() -> NThreading::TFuture<size_t> {
        Y_DEFER {
            result.push_back("coroutine_all_destroy");
        };

        result.push_back("pre_coroutine_return_value");
        size_t res1 = co_await coroutineReturnValue();
        result.push_back("post_coroutine_return_value");

        result.push_back("pre_coroutine_return_exception");
        size_t res2 = co_await coroutineReturnException();
        result.push_back("post_coroutine_return_exception");

        co_return res1 + res2;
    };

    EXPECT_THROW_MESSAGE_HAS_SUBSTR(
        coroutineReturnValueReturnException().GetValueSync(),
        std::runtime_error,
        "exception_to_return"
    );
    EXPECT_THAT(
        result,
        ::testing::ContainerEq(
            TVector<TString>({
                "pre_coroutine_return_value",
                "coroutine_return_value",
                "post_coroutine_return_value",

                "pre_coroutine_return_exception",
                "coroutine_return_exception",

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

TEST(TestFutureTraits, DestructorOrder) {
    class TTrackedValue {
    public:
        TTrackedValue(TVector<TString>& result, TString name)
            : Result(result)
            , Name(std::move(name))
        {
            Result.push_back(Name + " constructed");
        }

        TTrackedValue(TTrackedValue&& rhs)
            : Result(rhs.Result)
            , Name(std::move(rhs.Name))
        {
            Result.push_back(Name + " moved");
            rhs.Name.clear();
        }

        ~TTrackedValue() {
            if (!Name.empty()) {
                Result.push_back(Name + " destroyed");
            }
        }

    private:
        TVector<TString>& Result;
        TString Name;
    };

    TVector<TString> result;
    NThreading::TPromise<void> promise = NThreading::NewPromise<void>();
    NThreading::TFuture<void> future = promise.GetFuture();

    auto coroutine1 = [&](TTrackedValue arg) -> NThreading::TFuture<TString> {
        TTrackedValue a(result, "local a");
        result.push_back("before co_await future");
        co_await future;
        result.push_back("after co_await future");
        Y_UNUSED(arg);
        co_return "42";
    };

    auto coroutine2 = [&]() -> NThreading::TFuture<void> {
        TTrackedValue b(result, "local b");
        result.push_back("before co_await coroutine1(...)");
        TString value = co_await coroutine1(TTrackedValue(result, "arg"));
        result.push_back("after co_await coroutine1(...)");
        result.push_back("value = " + value);
    };

    result.push_back("before coroutine2()");
    auto future2 = coroutine2();
    result.push_back("after coroutine2()");
    EXPECT_FALSE(future2.HasValue() || future2.HasException());
    future2.Subscribe([&](const auto&) {
        result.push_back("in coroutine2() callback");
    });

    promise.SetValue();
    EXPECT_TRUE(future2.HasValue());

    EXPECT_THAT(
        result,
        ::testing::ContainerEq(
            TVector<TString>({
                "before coroutine2()",
                "local b constructed",
                "before co_await coroutine1(...)",
                "arg constructed",
                "arg moved",
                "local a constructed",
                "before co_await future",
                "after coroutine2()",
                "after co_await future",
                "local a destroyed",
                "arg destroyed",
                "after co_await coroutine1(...)",
                "value = 42",
                "local b destroyed",
                "in coroutine2() callback",
            })
        )
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
