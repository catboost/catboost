#include "future.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/noncopyable.h>
#include <util/generic/xrange.h>
#include <util/thread/pool.h>

#include <atomic>
#include <exception>

using NThreading::NewPromise;
using NThreading::TFuture;
using NThreading::TPromise;
using NThreading::TWaitPolicy;

namespace {
    // Wait* implementation without optimizations, to test TWaitGroup better
    template <class WaitPolicy, class TContainer>
    TFuture<void> WaitNoOpt(const TContainer& futures) {
        NThreading::TWaitGroup<WaitPolicy> wg;
        for (const auto& fut : futures) {
            wg.Add(fut);
        }

        return std::move(wg).Finish();
    }

    class TRelaxedBarrier {
    public:
        explicit TRelaxedBarrier(i64 size)
            : Waiting_{size} {
        }

        void Arrive() {
            // barrier is not for synchronization, just to ensure good timings, so
            // std::memory_order_relaxed is enough
            Waiting_.fetch_add(-1, std::memory_order_relaxed);

            while (Waiting_.load(std::memory_order_relaxed)) {
            }

            Y_ASSERT(Waiting_.load(std::memory_order_relaxed) >= 0);
        }

    private:
        std::atomic<i64> Waiting_;
    };

    THolder<TThreadPool> MakePool() {
        auto pool = MakeHolder<TThreadPool>(TThreadPool::TParams{}.SetBlocking(false).SetCatching(false));
        pool->Start(8);
        return pool;
    }

    template <class T>
    TVector<TFuture<T>> ToFutures(const TVector<TPromise<T>>& promises) {
        TVector<TFuture<void>> futures;

        for (auto&& p : promises) {
            futures.emplace_back(p);
        }

        return futures;
    }

    struct TStateSnapshot {
        i64 Started = -1;
        i64 StartedException = -1;
        const TVector<TFuture<void>>* Futures = nullptr;
    };

    // note: std::memory_order_relaxed should be enough everywhere, because TFuture::SetValue must provide the
    // needed synchronization
    template <class TFactory>
    void RunWaitTest(TFactory global) {
        auto pool = MakePool();

        const auto exception = std::make_exception_ptr(42);

        for (auto numPromises : xrange(1, 5)) {
            for (auto loopIter : xrange(1024 * 64)) {
                const auto numParticipants = numPromises + 1;

                TRelaxedBarrier barrier{numParticipants};

                std::atomic<i64> started = 0;
                std::atomic<i64> startedException = 0;
                std::atomic<i64> completed = 0;

                TVector<TPromise<void>> promises;
                for (auto i : xrange(numPromises)) {
                    Y_UNUSED(i);
                    promises.push_back(NewPromise());
                }

                const auto futures = ToFutures(promises);

                auto snapshotter = [&] {
                    return TStateSnapshot{
                        .Started = started.load(std::memory_order_relaxed),
                        .StartedException = startedException.load(std::memory_order_relaxed),
                        .Futures = &futures,
                    };
                };

                for (auto i : xrange(numPromises)) {
                    pool->SafeAddFunc([&, i] {
                        barrier.Arrive();

                        // subscribers must observe effects of this operation
                        // after .Set*
                        started.fetch_add(1, std::memory_order_relaxed);

                        if ((loopIter % 4 == 0) && i == 0) {
                            startedException.fetch_add(1, std::memory_order_relaxed);
                            promises[i].SetException(exception);
                        } else {
                            promises[i].SetValue();
                        }

                        completed.fetch_add(1, std::memory_order_release);
                    });
                }

                pool->SafeAddFunc([&] {
                    auto local = global(snapshotter);

                    barrier.Arrive();

                    local();

                    completed.fetch_add(1, std::memory_order_release);
                });

                while (completed.load() != numParticipants) {
                }
            }
        }
    }
}

Y_UNIT_TEST_SUITE(TFutureMultiThreadedTest) {
    Y_UNIT_TEST(WaitAll) {
        RunWaitTest(
            [](auto snapshotter) {
                return [=]() {
                    auto* futures = snapshotter().Futures;

                    auto all = WaitNoOpt<TWaitPolicy::TAll>(*futures);

                    // tests safety part
                    all.Subscribe([=] (auto&& all) {
                        TStateSnapshot snap = snapshotter();

                        // value safety: all is set => every future is set
                        UNIT_ASSERT(all.HasValue() <= ((snap.Started == (i64)snap.Futures->size()) && !snap.StartedException));

                        // safety for hasException: all is set => every future is set and some has exception
                        UNIT_ASSERT(all.HasException() <= ((snap.Started == (i64)snap.Futures->size()) && snap.StartedException > 0));
                    });

                    // test liveness
                    all.Wait();
                };
            });
    }

    Y_UNIT_TEST(WaitAny) {
        RunWaitTest(
            [](auto snapshotter) {
                return [=]() {
                    auto* futures = snapshotter().Futures;

                    auto any = WaitNoOpt<TWaitPolicy::TAny>(*futures);

                    // safety: any is ready => some f is ready
                    any.Subscribe([=](auto&&) {
                        UNIT_ASSERT(snapshotter().Started > 0);
                    });

                    // do we need better multithreaded liveness tests?
                    any.Wait();
                };
            });
    }

    Y_UNIT_TEST(WaitExceptionOrAll) {
        RunWaitTest(
            [](auto snapshotter) {
                return [=]() {
                    NThreading::WaitExceptionOrAll(*snapshotter().Futures)
                        .Subscribe([=](auto&&) {
                            auto* futures = snapshotter().Futures;

                            auto exceptionOrAll = WaitNoOpt<TWaitPolicy::TExceptionOrAll>(*futures);

                            exceptionOrAll.Subscribe([snapshotter](auto&& exceptionOrAll) {
                                TStateSnapshot snap = snapshotter();

                                // safety for hasException: exceptionOrAll has exception => some has exception
                                UNIT_ASSERT(exceptionOrAll.HasException() ? snap.StartedException > 0 : true);

                                // value safety: exceptionOrAll has value => all have value
                                UNIT_ASSERT(exceptionOrAll.HasValue() == ((snap.Started == (i64)snap.Futures->size()) && !snap.StartedException));
                            });

                            // do we need better multithreaded liveness tests?
                            exceptionOrAll.Wait();
                        });
                };
            });
    }
}

