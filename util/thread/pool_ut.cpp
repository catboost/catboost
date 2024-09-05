#include "pool.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/random/fast.h>
#include <util/system/spinlock.h>
#include <util/system/thread.h>
#include <util/system/mutex.h>
#include <util/system/condvar.h>

struct TThreadPoolTest {
    TSpinLock Lock;
    long R = -1;

    struct TTask: public IObjectInQueue {
        TThreadPoolTest* Test = nullptr;
        long Value = 0;

        TTask(TThreadPoolTest* test, int value)
            : Test(test)
            , Value(value)
        {
        }

        void Process(void*) override {
            THolder<TTask> This(this);

            TGuard<TSpinLock> guard(Test->Lock);
            Test->R ^= Value;
        }
    };

    struct TOwnedTask: public IObjectInQueue {
        bool& Processed;
        bool& Destructed;

        TOwnedTask(bool& processed, bool& destructed)
            : Processed(processed)
            , Destructed(destructed)
        {
        }

        ~TOwnedTask() override {
            Destructed = true;
        }

        void Process(void*) override {
            Processed = true;
        }
    };

    inline void TestAnyQueue(IThreadPool* queue, size_t queueSize = 1000) {
        TReallyFastRng32 rand(17);
        const size_t cnt = 1000;

        R = 0;

        for (size_t i = 0; i < cnt; ++i) {
            R ^= (long)rand.GenRand();
        }

        queue->Start(10, queueSize);
        rand = TReallyFastRng32(17);

        for (size_t i = 0; i < cnt; ++i) {
            UNIT_ASSERT(queue->Add(new TTask(this, (long)rand.GenRand())));
        }

        queue->Stop();

        UNIT_ASSERT_EQUAL(0, R);
    }
};

class TFailAddQueue: public IThreadPool {
public:
    bool Add(IObjectInQueue* /*obj*/) override Y_WARN_UNUSED_RESULT {
        return false;
    }

    void Start(size_t, size_t) override {
    }

    void Stop() noexcept override {
    }

    size_t Size() const noexcept override {
        return 0;
    }
};

Y_UNIT_TEST_SUITE(TThreadPoolTest) {
    Y_UNIT_TEST(TestTThreadPool) {
        TThreadPoolTest t;
        TThreadPool q;
        t.TestAnyQueue(&q);
    }

    Y_UNIT_TEST(TestTThreadPoolBlocking) {
        TThreadPoolTest t;
        TThreadPool q(TThreadPool::TParams().SetBlocking(true));
        t.TestAnyQueue(&q, 100);
    }

    // disabled by pg@ long time ago due to test flaps
    // Tried to enable: REVIEW:78772
    Y_UNIT_TEST(TestTAdaptiveThreadPool) {
        if (false) {
            TThreadPoolTest t;
            TAdaptiveThreadPool q;
            t.TestAnyQueue(&q);
        }
    }

    Y_UNIT_TEST(TestAddAndOwn) {
        TThreadPool q;
        q.Start(2);
        bool processed = false;
        bool destructed = false;
        q.SafeAddAndOwn(MakeHolder<TThreadPoolTest::TOwnedTask>(processed, destructed));
        q.Stop();

        UNIT_ASSERT_C(processed, "Not processed");
        UNIT_ASSERT_C(destructed, "Not destructed");
    }

    Y_UNIT_TEST(TestAddFunc) {
        TFailAddQueue queue;
        bool added = queue.AddFunc(
            []() {} // Lambda, I call him 'Lambda'!
        );
        UNIT_ASSERT_VALUES_EQUAL(added, false);
    }

    Y_UNIT_TEST(TestSafeAddFuncThrows) {
        TFailAddQueue queue;
        UNIT_CHECK_GENERATED_EXCEPTION(queue.SafeAddFunc([] {}), TThreadPoolException);
    }

    Y_UNIT_TEST(TestFunctionNotCopied) {
        struct TFailOnCopy {
            TFailOnCopy() {
            }

            TFailOnCopy(TFailOnCopy&&) {
            }

            TFailOnCopy(const TFailOnCopy&) {
                UNIT_FAIL("Don't copy std::function inside TThreadPool");
            }
        };

        TThreadPool queue(TThreadPool::TParams().SetBlocking(false).SetCatching(true));
        queue.Start(2);

        queue.SafeAddFunc([data = TFailOnCopy()]() {});

        queue.Stop();
    }

    Y_UNIT_TEST(TestInfoGetters) {
        TThreadPool queue;

        queue.Start(2, 7);

        UNIT_ASSERT_EQUAL(queue.GetThreadCountExpected(), 2);
        UNIT_ASSERT_EQUAL(queue.GetThreadCountReal(), 2);
        UNIT_ASSERT_EQUAL(queue.GetMaxQueueSize(), 7);

        queue.Stop();

        queue.Start(4, 1);

        UNIT_ASSERT_EQUAL(queue.GetThreadCountExpected(), 4);
        UNIT_ASSERT_EQUAL(queue.GetThreadCountReal(), 4);
        UNIT_ASSERT_EQUAL(queue.GetMaxQueueSize(), 1);

        queue.Stop();
    }

    void TestFixedThreadName(IThreadPool& pool, const TString& expectedName) {
        pool.Start(1);
        TString name;
        pool.SafeAddFunc([&name]() {
            name = TThread::CurrentThreadName();
        });
        pool.Stop();
        if (TThread::CanGetCurrentThreadName()) {
            UNIT_ASSERT_EQUAL(name, expectedName);
            UNIT_ASSERT_UNEQUAL(TThread::CurrentThreadName(), expectedName);
        }
    }

    Y_UNIT_TEST(TestFixedThreadName) {
        const TString expectedName = "HelloWorld";
        {
            TThreadPool pool(TThreadPool::TParams().SetBlocking(true).SetCatching(false).SetThreadName(expectedName));
            TestFixedThreadName(pool, expectedName);
        }
        {
            TAdaptiveThreadPool pool(TThreadPool::TParams().SetThreadName(expectedName));
            TestFixedThreadName(pool, expectedName);
        }
    }

    void TestEnumeratedThreadName(IThreadPool& pool, const THashSet<TString>& expectedNames) {
        pool.Start(expectedNames.size());
        TMutex lock;
        TCondVar allReady;
        size_t readyCount = 0;
        THashSet<TString> names;
        for (size_t i = 0; i < expectedNames.size(); ++i) {
            pool.SafeAddFunc([&]() {
                with_lock (lock) {
                    if (++readyCount == expectedNames.size()) {
                        allReady.BroadCast();
                    } else {
                        while (readyCount != expectedNames.size()) {
                            allReady.WaitI(lock);
                        }
                    }
                    names.insert(TThread::CurrentThreadName());
                }
            });
        }
        pool.Stop();
        if (TThread::CanGetCurrentThreadName()) {
            UNIT_ASSERT_EQUAL(names, expectedNames);
        }
    }

    Y_UNIT_TEST(TestEnumeratedThreadName) {
        const TString namePrefix = "HelloWorld";
        const THashSet<TString> expectedNames = {
            "HelloWorld0",
            "HelloWorld1",
            "HelloWorld2",
            "HelloWorld3",
            "HelloWorld4",
            "HelloWorld5",
            "HelloWorld6",
            "HelloWorld7",
            "HelloWorld8",
            "HelloWorld9",
            "HelloWorld10",
        };
        {
            TThreadPool pool(TThreadPool::TParams().SetBlocking(true).SetCatching(false).SetThreadNamePrefix(namePrefix));
            TestEnumeratedThreadName(pool, expectedNames);
        }
        {
            TAdaptiveThreadPool pool(TThreadPool::TParams().SetThreadNamePrefix(namePrefix));
            TestEnumeratedThreadName(pool, expectedNames);
        }
    }
} // Y_UNIT_TEST_SUITE(TThreadPoolTest)
