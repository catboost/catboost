#include "simple_task.h"

#include <library/cpp/testing/gtest/gtest.h>
#include <library/cpp/testing/gtest_extensions/gtest_extensions.h>
#include <library/cpp/threading/future/future.h>
#include <library/cpp/threading/future/core/coroutine_traits.h>

TEST(TestFutureTraits, ArgumentDependentLookup) {
    TVector<TString> result;
    NThreading::TPromise<void> promise = NThreading::NewPromise<void>();
    NThreading::TFuture<void> future = promise.GetFuture();

    NSimpleTestTask::RunAwaitable(result, std::move(future));

    promise.SetValue();

    EXPECT_THAT(
        result,
        ::testing::ContainerEq(
            TVector<TString>({
                "before co_await",
                "after co_await",
            })));
}
