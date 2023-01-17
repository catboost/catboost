#include "async_semaphore.h"
#include "async.h"

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/threading/cancellation/operation_cancelled_exception.h>

#include <util/generic/scope.h>
#include <util/generic/vector.h>
#include <util/thread/pool.h>

using namespace NThreading;

Y_UNIT_TEST_SUITE(TSemaphoreAsync) {
    Y_UNIT_TEST(SimplyAquired) {
        const size_t MAX_IN_PROGRESS = 5;

        TSimpleThreadPool pool(TThreadPool::TParams().SetCatching(false));
        pool.Start(MAX_IN_PROGRESS * 2);

        TVector<TFuture<size_t>> futures;
        auto semaphore = TAsyncSemaphore::Make(MAX_IN_PROGRESS);
        for (size_t i = 0; i < 100; ++i) {
            auto f = semaphore->AcquireAsync()
                .Apply([&pool, i](const auto& f) -> TFuture<size_t> {
                    return Async([i, semaphore = f.GetValue()] {
                        auto guard = semaphore->MakeAutoRelease();
                        Sleep(TDuration::MilliSeconds(100));
                        return i;
                    }, pool);
                });
            futures.push_back(f);
        }

        for (size_t i = 0; i < 100; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(futures[i].GetValueSync(), i);
        }
    }

    Y_UNIT_TEST(AutoReleasedOnException) {
        auto semaphore = TAsyncSemaphore::Make(1);

        auto lock = semaphore->AcquireAsync();
        UNIT_ASSERT(lock.HasValue());
        auto waitingLock = semaphore->AcquireAsync();
        UNIT_ASSERT(!waitingLock.HasValue() && !waitingLock.HasException());

        auto future = lock.Apply([](const auto& f) {
            auto guard = f.GetValue()->MakeAutoRelease();

            ythrow yexception() << "oops";
        });

        UNIT_ASSERT(future.HasException());
        UNIT_ASSERT(waitingLock.HasValue());
    }

    Y_UNIT_TEST(LimitsParallelism) {
        const size_t MAX_IN_PROGRESS = 5;

        TSimpleThreadPool pool(TThreadPool::TParams().SetCatching(false));
        pool.Start(MAX_IN_PROGRESS * 2);

        std::atomic_uint64_t inProgress = 0;

        TVector<TFuture<size_t>> futures;
        auto semaphore = TAsyncSemaphore::Make(MAX_IN_PROGRESS);
        for (size_t i = 0; i < 100; ++i) {
            auto f = semaphore->AcquireAsync()
                .Apply([&, i](const auto&) -> TFuture<size_t> {
                    auto currentInProgress = inProgress.fetch_add(1) + 1;

                    UNIT_ASSERT_GT(currentInProgress, 0);
                    UNIT_ASSERT_LE(currentInProgress, MAX_IN_PROGRESS);

                    return Async([i] {
                        Sleep(TDuration::MilliSeconds(100));
                        return i;
                    }, pool);
                });
            f.IgnoreResult().Subscribe([&](const auto&) {
                auto currentInProgress = inProgress.fetch_sub(1) - 1;

                UNIT_ASSERT_GE(currentInProgress, 0);
                UNIT_ASSERT_LE(currentInProgress, MAX_IN_PROGRESS);

                semaphore->Release();
            });
            futures.push_back(f);
        }

        WaitAll(futures).Wait();

        UNIT_ASSERT_EQUAL(inProgress.load(), 0);
    }

    Y_UNIT_TEST(AcquisitionOrder) {
        const size_t MAX_IN_PROGRESS = 5;

        TSimpleThreadPool pool(TThreadPool::TParams().SetCatching(false));
        pool.Start(MAX_IN_PROGRESS * 2);

        std::atomic_size_t latestId = 0;

        TVector<TFuture<size_t>> futures;
        auto semaphore = TAsyncSemaphore::Make(MAX_IN_PROGRESS);
        for (size_t i = 0; i < 100; ++i) {
            auto f = semaphore->AcquireAsync()
                .Apply([&](const auto& f) -> size_t {
                    auto guard = f.GetValue()->MakeAutoRelease();

                    auto currentId = latestId.fetch_add(1);

                    return currentId;
                });
            futures.push_back(f);
        }

        for (size_t i = 0; i < 100; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(futures[i].GetValueSync(), i);
        }
    }

    Y_UNIT_TEST(Cancel) {
        auto semaphore = TAsyncSemaphore::Make(1);
        auto firstLock = semaphore->AcquireAsync();
        auto canceledLock = semaphore->AcquireAsync();

        UNIT_ASSERT(firstLock.HasValue());

        UNIT_ASSERT(!canceledLock.HasValue());
        UNIT_ASSERT(!canceledLock.HasException());

        semaphore->Cancel();

        UNIT_ASSERT_EXCEPTION(canceledLock.TryRethrow(), TOperationCancelledException);

        UNIT_ASSERT_NO_EXCEPTION(firstLock.GetValue()->Release());
    }

    Y_UNIT_TEST(AcquireAfterCancel) {
        auto semaphore = TAsyncSemaphore::Make(1);

        semaphore->Cancel();

        auto lock = semaphore->AcquireAsync();

        UNIT_ASSERT_EXCEPTION(lock.TryRethrow(), TOperationCancelledException);
    }

    Y_UNIT_TEST(AutoReleaseDeferReleaseReleasesOnException) {
        auto semaphore = TAsyncSemaphore::Make(1);

        auto lock = semaphore->AcquireAsync();
        UNIT_ASSERT(lock.HasValue());
        auto waitingLock = semaphore->AcquireAsync();
        UNIT_ASSERT(!waitingLock.HasValue() && !waitingLock.HasException());

        auto asyncWork = lock.Apply([](const auto& lock) {
            lock.TryRethrow();

            ythrow yexception() << "oops";
        });

        asyncWork.Subscribe(semaphore->MakeAutoRelease().DeferRelease());

        UNIT_ASSERT(asyncWork.HasException());
        UNIT_ASSERT(waitingLock.HasValue());
    }

    Y_UNIT_TEST(AutoReleaseNotCopyable) {
        UNIT_ASSERT(!std::is_copy_constructible_v<TAsyncSemaphore::TAutoRelease>);
        UNIT_ASSERT(!std::is_copy_assignable_v<TAsyncSemaphore::TAutoRelease>);
    }
}
