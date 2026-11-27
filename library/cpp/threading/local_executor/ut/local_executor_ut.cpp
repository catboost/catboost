#include <library/cpp/threading/local_executor/local_executor.h>
#include <library/cpp/threading/future/future.h>

#include <library/cpp/testing/unittest/registar.h>
#include <util/system/mutex.h>
#include <util/system/rwlock.h>
#include <util/generic/algorithm.h>

#include <atomic>

using namespace NPar;

class TTestException: public yexception {
};

static const int DefaultThreadsCount = 41;
static const int DefaultRangeSize = 999;

Y_UNIT_TEST_SUITE(ExecRangeWithFutures){
    bool AllOf(const TVector<int>& vec, int value){
        return AllOf(vec, [value](int element) { return value == element; });
}

void AsyncRunAndWaitFuturesReady(int rangeSize, int threads) {
    TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threads);
    std::atomic<bool> signal = false;
    TVector<int> data(rangeSize, 0);
    TVector<NThreading::TFuture<void>> futures = localExecutor.ExecRangeWithFutures([&signal, &data](int i) {
        UNIT_ASSERT(data[i] == 0);
        while (!signal.load())
            ;
        data[i] += 1;
    },
                                                                                    0, rangeSize, TLocalExecutor::HIGH_PRIORITY);
    UNIT_ASSERT(AllOf(data, 0));
    for (auto& future : futures)
        UNIT_ASSERT(!future.HasValue());
    signal.store(true);
    for (auto& future : futures) {
        future.GetValueSync();
    }
    UNIT_ASSERT(AllOf(data, 1));
}

Y_UNIT_TEST(AsyncRunRangeAndWaitFuturesReady) {
    AsyncRunAndWaitFuturesReady(DefaultRangeSize, DefaultThreadsCount);
}

Y_UNIT_TEST(AsyncRunOneTaskAndWaitFuturesReady) {
    AsyncRunAndWaitFuturesReady(1, DefaultThreadsCount);
}

Y_UNIT_TEST(AsyncRunRangeAndWaitFuturesReadyOneExtraThread) {
    AsyncRunAndWaitFuturesReady(DefaultRangeSize, 1);
}

Y_UNIT_TEST(AsyncRunOneThreadAndWaitFuturesReadyOneExtraThread) {
    AsyncRunAndWaitFuturesReady(1, 1);
}

Y_UNIT_TEST(AsyncRunTwoRangesAndWaitFuturesReady) {
    TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(DefaultThreadsCount);
    std::atomic<bool> signal = false;
    TVector<int> data1(DefaultRangeSize, 0);
    TVector<NThreading::TFuture<void>> futures1 = localExecutor.ExecRangeWithFutures([&signal, &data1](int i) {
        UNIT_ASSERT(data1[i] == 0);
        while (!signal.load())
            ;
        data1[i] += 1;
    },
                                                                                     0, DefaultRangeSize, TLocalExecutor::HIGH_PRIORITY);
    TVector<int> data2(DefaultRangeSize, 0);
    TVector<NThreading::TFuture<void>> futures2 = localExecutor.ExecRangeWithFutures([&signal, &data2](int i) {
        UNIT_ASSERT(data2[i] == 0);
        while (!signal.load())
            ;
        data2[i] += 2;
    },
                                                                                     0, DefaultRangeSize, TLocalExecutor::HIGH_PRIORITY);
    UNIT_ASSERT(AllOf(data1, 0));
    UNIT_ASSERT(AllOf(data2, 0));
    signal.store(true);
    for (int i = 0; i < DefaultRangeSize; ++i) {
        futures1[i].GetValueSync();
        futures2[i].GetValueSync();
    }
    UNIT_ASSERT(AllOf(data1, 1));
    UNIT_ASSERT(AllOf(data2, 2));
}

void AsyncRunRangeAndWaitExceptions(int rangeSize, int threadsCount) {
    TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadsCount);
    std::atomic<bool> signal = false;
    TVector<int> data(rangeSize, 0);
    TVector<NThreading::TFuture<void>> futures = localExecutor.ExecRangeWithFutures([&signal, &data](int i) {
        UNIT_ASSERT(data[i] == 0);
        while (!signal.load())
            ;
        data[i] += 1;
        throw 10000 + i;
    },
                                                                                    0, rangeSize, TLocalExecutor::HIGH_PRIORITY);
    UNIT_ASSERT(AllOf(data, 0));
    UNIT_ASSERT(futures.ysize() == rangeSize);
    signal.store(true);
    int exceptionsCaught = 0;
    for (int i = 0; i < rangeSize; ++i) {
        try {
            futures[i].GetValueSync();
        } catch (int& e) {
            if (e == 10000 + i) {
                ++exceptionsCaught;
            }
        }
    }
    UNIT_ASSERT(exceptionsCaught == rangeSize);
    UNIT_ASSERT(AllOf(data, 1));
}

Y_UNIT_TEST(AsyncRunRangeAndWaitExceptions) {
    AsyncRunRangeAndWaitExceptions(DefaultRangeSize, DefaultThreadsCount);
}

Y_UNIT_TEST(AsyncRunOneTaskAndWaitExceptions) {
    AsyncRunRangeAndWaitExceptions(1, DefaultThreadsCount);
}

Y_UNIT_TEST(AsyncRunRangeAndWaitExceptionsOneExtraThread) {
    AsyncRunRangeAndWaitExceptions(DefaultRangeSize, 1);
}

Y_UNIT_TEST(AsyncRunOneTaskAndWaitExceptionsOneExtraThread) {
    AsyncRunRangeAndWaitExceptions(1, 1);
}

Y_UNIT_TEST(AsyncRunTwoRangesAndWaitExceptions) {
    TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(DefaultThreadsCount);
    std::atomic<bool> signal = false;
    TVector<int> data1(DefaultRangeSize, 0);
    TVector<NThreading::TFuture<void>> futures1 = localExecutor.ExecRangeWithFutures([&signal, &data1](int i) {
        UNIT_ASSERT(data1[i] == 0);
        while (!signal.load())
            ;
        data1[i] += 1;
        throw 15000 + i;
    },
                                                                                     0, DefaultRangeSize, TLocalExecutor::LOW_PRIORITY);
    TVector<int> data2(DefaultRangeSize, 0);
    TVector<NThreading::TFuture<void>> futures2 = localExecutor.ExecRangeWithFutures([&signal, &data2](int i) {
        UNIT_ASSERT(data2[i] == 0);
        while (!signal.load())
            ;
        data2[i] += 2;
        throw 16000 + i;
    },
                                                                                     0, DefaultRangeSize, TLocalExecutor::HIGH_PRIORITY);

    UNIT_ASSERT(AllOf(data1, 0));
    UNIT_ASSERT(AllOf(data2, 0));
    UNIT_ASSERT(futures1.size() == DefaultRangeSize);
    UNIT_ASSERT(futures2.size() == DefaultRangeSize);
    signal.store(true);
    int exceptionsCaught = 0;
    for (int i = 0; i < DefaultRangeSize; ++i) {
        try {
            futures1[i].GetValueSync();
        } catch (int& e) {
            if (e == 15000 + i) {
                ++exceptionsCaught;
            }
        }
        try {
            futures2[i].GetValueSync();
        } catch (int& e) {
            if (e == 16000 + i) {
                ++exceptionsCaught;
            }
        }
    }
    UNIT_ASSERT(exceptionsCaught == 2 * DefaultRangeSize);
    UNIT_ASSERT(AllOf(data1, 1));
    UNIT_ASSERT(AllOf(data2, 2));
}

void RunRangeAndCheckExceptionsWithWaitComplete(int rangeSize, int threadsCount) {
    TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadsCount);
    TVector<int> data(rangeSize, 0);
    TVector<NThreading::TFuture<void>> futures = localExecutor.ExecRangeWithFutures([&data](int i) {
        UNIT_ASSERT(data[i] == 0);
        data[i] += 1;
        throw 30000 + i;
    },
                                                                                    0, rangeSize, TLocalExecutor::EFlags::WAIT_COMPLETE);
    UNIT_ASSERT(AllOf(data, 1));
    int exceptionsCaught = 0;
    for (int i = 0; i < rangeSize; ++i) {
        try {
            futures[i].GetValueSync();
        } catch (int& e) {
            if (e == 30000 + i) {
                ++exceptionsCaught;
            }
        }
    }
    UNIT_ASSERT(exceptionsCaught == rangeSize);
    UNIT_ASSERT(AllOf(data, 1));
}

Y_UNIT_TEST(RunRangeAndCheckExceptionsWithWaitComplete) {
    RunRangeAndCheckExceptionsWithWaitComplete(DefaultRangeSize, DefaultThreadsCount);
}

Y_UNIT_TEST(RunOneAndCheckExceptionsWithWaitComplete) {
    RunRangeAndCheckExceptionsWithWaitComplete(1, DefaultThreadsCount);
}

Y_UNIT_TEST(RunRangeAndCheckExceptionsWithWaitCompleteOneExtraThread) {
    RunRangeAndCheckExceptionsWithWaitComplete(DefaultRangeSize, 1);
}

Y_UNIT_TEST(RunOneAndCheckExceptionsWithWaitCompleteOneExtraThread) {
    RunRangeAndCheckExceptionsWithWaitComplete(1, 1);
}

Y_UNIT_TEST(RunRangeAndCheckExceptionsWithWaitCompleteZeroExtraThreads) {
    RunRangeAndCheckExceptionsWithWaitComplete(DefaultRangeSize, 0);
}

Y_UNIT_TEST(RunOneAndCheckExceptionsWithWaitCompleteZeroExtraThreads) {
    RunRangeAndCheckExceptionsWithWaitComplete(1, 0);
}
}

Y_UNIT_TEST_SUITE(ExecRangeWithThrow){
    void RunParallelWhichThrowsTTestException(int rangeStart, int rangeSize, int threadsCount, int flags, std::atomic<int>& processed){
        processed.store(0);
TLocalExecutor localExecutor;
localExecutor.RunAdditionalThreads(threadsCount);
localExecutor.ExecRangeWithThrow([&processed](int) {
    ++processed;
    throw TTestException();
},
                                 rangeStart, rangeStart + rangeSize, flags);
}

Y_UNIT_TEST(RunParallelWhichThrowsTTestException) {
    std::atomic<int> processed = 0;
    UNIT_ASSERT_EXCEPTION(
        RunParallelWhichThrowsTTestException(10, 40, DefaultThreadsCount,
                                             TLocalExecutor::EFlags::WAIT_COMPLETE, processed),
        TTestException);
    UNIT_ASSERT(processed.load() == 40);
}

void ThrowAndCatchTTestException(int rangeSize, int threadsCount, int flags) {
    std::atomic<int> processed = 0;
    UNIT_ASSERT_EXCEPTION(
        RunParallelWhichThrowsTTestException(0, rangeSize, threadsCount, flags, processed),
        TTestException);
    UNIT_ASSERT(processed.load() == rangeSize);
}

Y_UNIT_TEST(ThrowAndCatchTTestExceptionLowPriority) {
    ThrowAndCatchTTestException(DefaultRangeSize, DefaultThreadsCount,
                                TLocalExecutor::EFlags::WAIT_COMPLETE | TLocalExecutor::EFlags::LOW_PRIORITY);
}

Y_UNIT_TEST(ThrowAndCatchTTestExceptionMedPriority) {
    ThrowAndCatchTTestException(DefaultRangeSize, DefaultThreadsCount,
                                TLocalExecutor::EFlags::WAIT_COMPLETE | TLocalExecutor::EFlags::MED_PRIORITY);
}

Y_UNIT_TEST(ThrowAndCatchTTestExceptionHighPriority) {
    ThrowAndCatchTTestException(DefaultRangeSize, DefaultThreadsCount,
                                TLocalExecutor::EFlags::WAIT_COMPLETE | TLocalExecutor::EFlags::HIGH_PRIORITY);
}

Y_UNIT_TEST(ThrowAndCatchTTestExceptionWaitComplete) {
    ThrowAndCatchTTestException(DefaultRangeSize, DefaultThreadsCount,
                                TLocalExecutor::EFlags::WAIT_COMPLETE);
}

Y_UNIT_TEST(RethrowExeptionSequentialWaitComplete) {
    ThrowAndCatchTTestException(DefaultRangeSize, 0,
                                TLocalExecutor::EFlags::WAIT_COMPLETE);
}

Y_UNIT_TEST(RethrowExeptionOneExtraThreadWaitComplete) {
    ThrowAndCatchTTestException(DefaultRangeSize, 1,
                                TLocalExecutor::EFlags::WAIT_COMPLETE);
}

void ThrowsTTestExceptionFromNested(TLocalExecutor& localExecutor) {
    localExecutor.ExecRangeWithThrow([](int) {
        throw TTestException();
    },
                                     0, 10, TLocalExecutor::EFlags::WAIT_COMPLETE);
}

void CatchTTestExceptionFromNested(std::atomic<int>& processed1, std::atomic<int>& processed2) {
    TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(DefaultThreadsCount);
    localExecutor.ExecRangeWithThrow([&processed1, &processed2, &localExecutor](int) {
        ++processed1;
        UNIT_ASSERT_EXCEPTION(
            ThrowsTTestExceptionFromNested(localExecutor),
            TTestException);
        ++processed2;
    },
                                     0, DefaultRangeSize, TLocalExecutor::EFlags::WAIT_COMPLETE);
}

Y_UNIT_TEST(NestedParallelExceptionsDoNotLeak) {
    std::atomic<int> processed1 = 0;
    std::atomic<int> processed2 = 0;
    UNIT_ASSERT_NO_EXCEPTION(
        CatchTTestExceptionFromNested(processed1, processed2));
    UNIT_ASSERT_EQUAL(processed1.load(), DefaultRangeSize);
    UNIT_ASSERT_EQUAL(processed2.load(), DefaultRangeSize);
}
}

Y_UNIT_TEST_SUITE(ExecLargeRangeWithThrow){

    constexpr int LARGE_COUNT = 128 * (1 << 20);

    static auto IsValue(char v) {
        return [=](char c) { return c == v; };
    }

    Y_UNIT_TEST(ExecLargeRangeNoExceptions) {
        TVector<char> tasks(LARGE_COUNT);

        TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(DefaultThreadsCount);

        localExecutor.ExecRangeBlockedWithThrow([&tasks](int i) {
            tasks[i] = 1;
        }, 0, tasks.size(), 0, TLocalExecutor::EFlags::WAIT_COMPLETE);
        UNIT_ASSERT(AllOf(tasks, IsValue(1)));


        localExecutor.ExecRangeBlockedWithThrow([&tasks](int i) {
            tasks[i] += 1;
        }, 0, tasks.size(), 128, TLocalExecutor::EFlags::WAIT_COMPLETE);
        UNIT_ASSERT(AllOf(tasks, IsValue(2)));
    }

    Y_UNIT_TEST(ExecLargeRangeWithException) {
        TVector<char> tasks(LARGE_COUNT);

        TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(DefaultThreadsCount);

        Fill(tasks.begin(), tasks.end(), 0);
        UNIT_ASSERT_EXCEPTION(
            localExecutor.ExecRangeBlockedWithThrow([&tasks](int i) {
                tasks[i] += 1;
                if (i == LARGE_COUNT / 2) {
                    throw TTestException();
                }
            }, 0, tasks.size(), 0, TLocalExecutor::EFlags::WAIT_COMPLETE),
            TTestException
        );
    }
}
