#include "future.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/noncopyable.h>
#include <util/generic/xrange.h>
#include <util/thread/pool.h>

#include <atomic>
#include <exception>
#include <limits>

using NThreading::NewPromise;
using NThreading::TFuture;
using NThreading::TPromise;

namespace {
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
        const TVector<TFuture<void>>* Futures = nullptr;
    };

    // note: std::memory_order_relaxed should be enough everywhere, because TFuture::SetValue must provide the
    // needed synchronization
    template <class TFactory>
    void RunWaitTest(TFactory global) {
        auto pool = MakePool();

        const auto exception = std::make_exception_ptr(42);

        for (auto numPromises : xrange(2, 5)) {
            for (auto loopIter : xrange(1024 * 64)) {
                const auto numParticipants = numPromises + 1;

                TRelaxedBarrier barrier{numParticipants};

                std::atomic<i64> started = 0;
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
                    NThreading::WaitAll(*snapshotter().Futures)
                        .Subscribe([=](auto&&) {
                            TStateSnapshot snap = snapshotter();
                            UNIT_ASSERT_VALUES_EQUAL(snap.Started, snap.Futures->size());
                        });
                };
            });
    }

    Y_UNIT_TEST(WaitAny) {
        std::atomic<i64> lowest = std::numeric_limits<i64>::max();

        RunWaitTest(
            [lowest = &lowest](auto snapshotter) mutable {
                return [=]() {
                    NThreading::WaitAny(*snapshotter().Futures)
                        .Subscribe([=](auto&&) {
                            TStateSnapshot snap = snapshotter();

                            auto l = lowest->load(std::memory_order_relaxed);
                            auto current = snap.Started;
                            if (current < l) {
                                lowest->store(current, std::memory_order_relaxed);
                            }
                        });
                };
            });

        UNIT_ASSERT_VALUES_EQUAL(lowest.load(), 1);
    }

    Y_UNIT_TEST(WaitExceptionOrAll) {
        std::atomic<i64> lowest = 1024;

        RunWaitTest(
            [lowest = &lowest](auto snapshotter) mutable {
                return [=]() {
                    NThreading::WaitExceptionOrAll(*snapshotter().Futures)
                        .Subscribe([=](auto&&) {
                            TStateSnapshot snap = snapshotter();

                            auto l = lowest->load(std::memory_order_relaxed);
                            auto current = snap.Started;
                            if (current < l) {
                                lowest->store(current, std::memory_order_relaxed);
                            }

                            bool allStarted = current == (i64)snap.Futures->size();
                            bool anyException = false;

                            for (auto&& fut : *snap.Futures) {
                                anyException = anyException || fut.HasException();
                            }

                            UNIT_ASSERT(allStarted || anyException);
                        });
                };
            });

        UNIT_ASSERT_VALUES_EQUAL(lowest.load(), 1);
    }
}

