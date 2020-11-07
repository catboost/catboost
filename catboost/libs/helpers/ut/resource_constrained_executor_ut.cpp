#include <catboost/libs/helpers/resource_constrained_executor.h>

#include <catboost/libs/helpers/exception.h>

#include <util/datetime/base.h>
#include <util/generic/xrange.h>
#include <util/stream/output.h>
#include <util/string/cast.h>
#include <util/system/guard.h>
#include <util/system/mutex.h>

#include <array>

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TResourceConstrainedExecutor) {
    Y_UNIT_TEST(TestDoNothing) {
        NCB::TResourceConstrainedExecutor executor("Memory", 0, true, &NPar::LocalExecutor());
    }

    void SimpleTestCase(size_t threadsCount, bool runInsideLocalExecutor, bool lenientMode) {
        // task type is also it's resource consumption (and work time, emulated by sleep)
        constexpr size_t TASK_TYPE_COUNT = 11;
        size_t resourceQuota = lenientMode ? 2 : 10;
        size_t maxResourceUsageInLenientMode = TASK_TYPE_COUNT-1;

        std::array<size_t, TASK_TYPE_COUNT> tasksPerType{ {3, 7, 4, 2, 1, 6, 2, 5, 7, 9, 5} };

        TMutex M; // protects resourceConsumption and counters
        size_t resourceConsumption = 0;
        std::array<size_t, TASK_TYPE_COUNT> counters{};

        auto func = [&] (NPar::ILocalExecutor& localExecutor) {
            {
                NCB::TResourceConstrainedExecutor executor(
                    "Memory",
                    resourceQuota,
                    lenientMode,
                    &localExecutor
                );
                for (auto taskType : xrange(TASK_TYPE_COUNT)) {
                    for (auto i : xrange(tasksPerType[taskType])) {
                        Y_UNUSED(i);
                        executor.Add(
                            {
                                taskType,
                                [&, taskType]() {
                                    with_lock(M) {
                                        resourceConsumption += taskType;
                                        UNIT_ASSERT(
                                            resourceConsumption <=
                                            (lenientMode ? maxResourceUsageInLenientMode : resourceQuota)
                                        );
                                        ++counters[taskType];
                                    }
                                    Sleep(TDuration::MicroSeconds(10*taskType));
                                    with_lock(M) {
                                        resourceConsumption -= taskType;
                                    }
                                }
                            }
                        );
                    }
                }
            }

            for (auto taskType : xrange(TASK_TYPE_COUNT)) {
                UNIT_ASSERT(counters[taskType] == tasksPerType[taskType]);
            }
        };

        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(threadsCount);
        if (runInsideLocalExecutor) {
            localExecutor.ExecRange(
                [&localExecutor, &func](int id) { Y_UNUSED(id); func(localExecutor); },
                0,
                1,
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
        } else {
            func(localExecutor);
        }
    }

    Y_UNIT_TEST(TestSimple) {
        for (auto threadsCount : {1, 2, 10}) {
            for (auto runInsideLocalExecutor : {true, false}) {
                for (auto lenientMode : {true, false}) {
                    SimpleTestCase(threadsCount, runInsideLocalExecutor, lenientMode);
                }
            }
        }
    }

    Y_UNIT_TEST(TestImpossibleResourceRequest) {
        {
            NCB::TResourceConstrainedExecutor executor("Memory", 0, false, &NPar::LocalExecutor());
            UNIT_ASSERT_EXCEPTION(executor.Add({1, [](){;}}), TCatBoostException);
        }
        {
            NCB::TResourceConstrainedExecutor executor("Memory", 5, false, &NPar::LocalExecutor());
            UNIT_ASSERT_EXCEPTION(executor.Add({6, [](){;}}), TCatBoostException);
        }
    }

    Y_UNIT_TEST(TestExceptions) {
        for (auto withExecTasks : {true, false}) {
            {
                UNIT_ASSERT_EXCEPTION(
                    [&]() {
                        NPar::TLocalExecutor localExecutor;
                        localExecutor.RunAdditionalThreads(1);

                        NCB::TResourceConstrainedExecutor executor("Memory", 2, false, &localExecutor);
                        executor.Add({1, [](){;}});
                        executor.Add({1, [](){;}});
                        executor.Add({2, [](){ ythrow TCatBoostException(); }});

                        if (withExecTasks) {
                            executor.ExecTasks();
                        }
                    }(),
                    TCatBoostException
                );
            }
            {
                UNIT_ASSERT_EXCEPTION(
                    [&]() {
                        NPar::TLocalExecutor localExecutor;
                        localExecutor.RunAdditionalThreads(3);

                        NCB::TResourceConstrainedExecutor executor("Memory", 2, false, &localExecutor);
                        executor.Add({1, [](){;}});
                        executor.Add({1, [](){ FromString<int>("2.9"); }});
                        executor.Add({0, [](){ ythrow TCatBoostException(); }});

                        if (withExecTasks) {
                            executor.ExecTasks();
                        }
                    }(),
                    std::exception
                );
            }
        }
    }
}

