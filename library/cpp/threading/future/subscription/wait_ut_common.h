#pragma once

#include <library/cpp/threading/future/future.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/vector.h>

#include <functional>
#include <type_traits>

namespace NThreading::NTest {

namespace NPrivate {

void ExecuteAndWait(TVector<std::function<void()>> jobs, TFuture<void> waiter, size_t threads);

template <typename TPromises, typename FSetter>
void SetConcurrentAndWait(TPromises&& promises, FSetter&& setter, TFuture<void> waiter, size_t threads = 8) {
    TVector<std::function<void()>> jobs;
    jobs.reserve(std::size(promises));
    for (auto& p : promises) {
        jobs.push_back([p, setter]() mutable {setter(p); });
    }
    ExecuteAndWait(std::move(jobs), std::move(waiter), threads);
}

template <typename T>
auto MakePromise() {
    if constexpr (std::is_same_v<T, void>) {
        return NewPromise();
    }
    return NewPromise<T>();
}

}

template <typename T, typename FWaiterFactory, typename FSetterFactory, typename FChecker>
void TestManyStress(FWaiterFactory&& waiterFactory, FSetterFactory&& setterFactory, FChecker&& checker) {
    for (size_t i : { 1, 2, 4, 8, 16, 32, 64, 128, 256 }) {
        TVector<TPromise<T>> promises;
        TVector<TFuture<T>> futures;
        promises.reserve(i);
        futures.reserve(i);
        for (size_t j = 0; j < i; ++j) {
            auto promise = NPrivate::MakePromise<T>();
            futures.push_back(promise.GetFuture());
            promises.push_back(std::move(promise));
        }
        auto waiter = waiterFactory(futures);
        NPrivate::SetConcurrentAndWait(std::move(promises), [valueSetter = setterFactory(i)](auto&& p) { valueSetter(p); }
                                        , waiter);
        checker(waiter);
    }
}

}
