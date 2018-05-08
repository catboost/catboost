#include <library/unittest/registar.h>
#include <library/threading/local_executor/local_executor.h>
#include <library/threading/local_executor/waitable_registry.h>

#include <util/generic/vector.h>
#include <util/thread/pool.h>

static int GetPriority(size_t i) {
    static const int priorities[] = {
        NPar::TLocalExecutor::HIGH_PRIORITY,
        NPar::TLocalExecutor::MED_PRIORITY,
        NPar::TLocalExecutor::LOW_PRIORITY
    };

    return priorities[i % 3];
}

static TAutoPtr<IThreadPool::IThread> AddUnrelatedJobs(NPar::TLocalExecutor& e) {
    return SystemThreadPool()->Run([&e]{
        e.Exec([](int) {
            // do nothing
        }, 0, NPar::TLocalExecutor::HIGH_PRIORITY);
    });
}

Y_UNIT_TEST_SUITE(WaitableRegistryTests) {
    Y_UNIT_TEST(TestOneTaskAddThreadsBeforeExec) {
        NPar::TLocalExecutor e;
        const auto r = NPar::MakeDefaultWaitableRegistry();
        e.RunAdditionalThreads(3);

        bool flag = false;
        e.Exec(r->Register([&flag](int){ flag = true; }), 0, NPar::TLocalExecutor::HIGH_PRIORITY);

        r->Wait();

        UNIT_ASSERT(flag);
    }

    Y_UNIT_TEST(TestOneTaskAddThreadsAfterExec) {
        NPar::TLocalExecutor e;
        const auto r = NPar::MakeDefaultWaitableRegistry();

        bool flag = false;
        e.Exec(r->Register([&flag](int){ flag = true; }), 0, NPar::TLocalExecutor::HIGH_PRIORITY);

        e.RunAdditionalThreads(3);
        r->Wait();

        UNIT_ASSERT(flag);
    }

    Y_UNIT_TEST(TestMultipleTasksAddThreadsBeforeExec) {
        NPar::TLocalExecutor e;
        const auto r = NPar::MakeDefaultWaitableRegistry();
        e.RunAdditionalThreads(3);

        TVector<bool> flags(10);
        for (auto& flag : flags) {
            e.Exec(r->Register([&flag](int){ flag = true; }), 0, NPar::TLocalExecutor::HIGH_PRIORITY);
        }

        r->Wait();

        for (const auto flag : flags) {
            UNIT_ASSERT(flag);
        }
    }

    Y_UNIT_TEST(TestMultipleTasksAddThreadsAfterExec) {
        NPar::TLocalExecutor e;
        const auto r = NPar::MakeDefaultWaitableRegistry();

        TVector<bool> flags(10);
        for (auto& flag : flags) {
            e.Exec(r->Register([&flag](int){ flag = true; }), 0, NPar::TLocalExecutor::HIGH_PRIORITY);
        }

        e.RunAdditionalThreads(3);
        r->Wait();

        for (const auto flag : flags) {
            UNIT_ASSERT(flag);
        }
    }

    Y_UNIT_TEST(TestMultipleTasksWithDifferentPrioritiesAddThreadsBeforeExec) {
        NPar::TLocalExecutor e;
        const auto r = NPar::MakeDefaultWaitableRegistry();
        e.RunAdditionalThreads(3);

        TVector<bool> flags(10);
        for (size_t i = 0; i < flags.size(); ++i) {
            e.Exec(r->Register([flag = &flags[i]](int){ *flag = true; }), 0, GetPriority(i));
        }

        r->Wait();

        for (const auto flag : flags) {
            UNIT_ASSERT(flag);
        }
    }

    Y_UNIT_TEST(TestMultipleTasksWithDifferentPrioritiesAddThreadsAfterExec) {
        NPar::TLocalExecutor e;
        const auto r = NPar::MakeDefaultWaitableRegistry();

        TVector<bool> flags(10);
        for (size_t i = 0; i < flags.size(); ++i) {
            e.Exec(r->Register([flag = &flags[i]](int){ *flag = true; }), 0, GetPriority(i));
        }

        e.RunAdditionalThreads(3);
        r->Wait();

        for (const auto flag : flags) {
            UNIT_ASSERT(flag);
        }
    }

    Y_UNIT_TEST(TestFromMultipleThreadsMultipleTasksWithDifferentPrioritiesAddThreadsBeforeExec) {
        NPar::TLocalExecutor e;
        const auto r = NPar::MakeDefaultWaitableRegistry();
        e.RunAdditionalThreads(3);

        TVector<TAutoPtr<IThreadPool::IThread>> threads;
        TVector<bool> flagsArr[2];
        flagsArr[0].resize(10);
        flagsArr[1].resize(10);
        for (auto& flags : flagsArr) {
            threads.push_back(SystemThreadPool()->Run([&e, &r, &flags]{
                for (size_t i = 0; i < flags.size(); ++i) {
                    e.Exec(r->Register([flag = &flags[i]](int){ *flag = true; }), 0, GetPriority(i));
                }
            }));
        }

        for (const auto& thread : threads) {
            thread->Join();
        }

        r->Wait();

        for (const auto& flags : flagsArr) {
            for (const auto flag : flags) {
                UNIT_ASSERT(flag);
            }
        }
    }

    Y_UNIT_TEST(TestFromMultipleThreadsMultipleTasksWithDifferentPrioritiesAddThreadsAfterExec) {
        NPar::TLocalExecutor e;
        const auto r = NPar::MakeDefaultWaitableRegistry();

        TVector<TAutoPtr<IThreadPool::IThread>> threads;
        TVector<bool> flagsArr[2];
        flagsArr[0].resize(10);
        flagsArr[1].resize(10);
        for (auto& flags : flagsArr) {
            threads.push_back(SystemThreadPool()->Run([&e, &r, &flags]{
                for (size_t i = 0; i < flags.size(); ++i) {
                    e.Exec(r->Register([flag = &flags[i]](int){ *flag = true; }), 0, GetPriority(i));
                }
            }));
        }

        e.RunAdditionalThreads(3);

        for (const auto& thread : threads) {
            thread->Join();
        }

        r->Wait();

        for (const auto& flags : flagsArr) {
            for (const auto flag : flags) {
                UNIT_ASSERT(flag);
            }
        }
    }

    Y_UNIT_TEST(TestFromMultipleThreadsMultipleTasksWithDifferentPrioritiesAddThreadsBeforeExecAndUnrelatedJobs) {
        NPar::TLocalExecutor e;
        const auto r = NPar::MakeDefaultWaitableRegistry();
        e.RunAdditionalThreads(3);

        TVector<TAutoPtr<IThreadPool::IThread>> threads;
        TVector<bool> flagsArr[2];
        flagsArr[0].resize(10);
        flagsArr[1].resize(10);
        for (auto& flags : flagsArr) {
            threads.push_back(SystemThreadPool()->Run([&e, &r, &flags]{
                for (size_t i = 0; i < flags.size(); ++i) {
                    e.Exec(r->Register([flag = &flags[i]](int){ *flag = true; }), 0, GetPriority(i));
                }
            }));
        }

        // these jobs may be added before or after `Sync` invocation, it's for thread sanitizer to
        // increase number of possible cases
        const auto unrelated = AddUnrelatedJobs(e);

        for (const auto& thread : threads) {
            thread->Join();
        }

        r->Wait();

        for (const auto& flags : flagsArr) {
            for (const auto flag : flags) {
                UNIT_ASSERT(flag);
            }
        }

        unrelated->Join();
    }

    Y_UNIT_TEST(TestFromMultipleThreadsMultipleTasksWithDifferentPrioritiesAddThreadsAfterExecAndUnrelatedJobs) {
        NPar::TLocalExecutor e;
        const auto r = NPar::MakeDefaultWaitableRegistry();

        TVector<TAutoPtr<IThreadPool::IThread>> threads;
        TVector<bool> flagsArr[2];
        flagsArr[0].resize(10);
        flagsArr[1].resize(10);
        for (auto& flags : flagsArr) {
            threads.push_back(SystemThreadPool()->Run([&e, &r, &flags]{
                for (size_t i = 0; i < flags.size(); ++i) {
                    e.Exec(r->Register([flag = &flags[i]](int){ *flag = true; }), 0, GetPriority(i));
                }
            }));
        }

        // these jobs may be added before or after `Wait` invocation, it's for thread sanitizer to
        // increase number of possible cases
        const auto unrelated = AddUnrelatedJobs(e);

        e.RunAdditionalThreads(3);

        for (const auto& thread : threads) {
            thread->Join();
        }

        r->Wait();

        for (const auto& flags : flagsArr) {
            for (const auto flag : flags) {
                UNIT_ASSERT(flag);
            }
        }

        unrelated->Join();
    }

    Y_UNIT_TEST(TestReset) {
        const auto r = NPar::MakeDefaultWaitableRegistry();
        r->Register([](int){});
        r->Reset();

        // If `Reset` is not correct this will deadlock
        r->Wait();

        NPar::TLocalExecutor e;
        e.RunAdditionalThreads(1);
        e.Exec(r->Register([](int){}), 0, NPar::TLocalExecutor::HIGH_PRIORITY);
        r->Wait();
    }
}
