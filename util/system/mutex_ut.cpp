#include "mutex.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/thread/pool.h>
#include <util/random/random.h>

class TMutexTest: public TTestBase {
    UNIT_TEST_SUITE(TMutexTest);
    UNIT_TEST(TestBasics)
    UNIT_TEST(TestFake)
    UNIT_TEST(TestRecursive)
    UNIT_TEST_SUITE_END();

    struct TSharedData {
        TSharedData()
            : sharedCounter(0)
            , failed(false)
        {
        }

        volatile ui32 sharedCounter;
        TMutex mutex;
        TFakeMutex fakeMutex;

        bool failed;
    };

    class TThreadTask: public IObjectInQueue {
    public:
        using PFunc = void (TThreadTask::*)(void);

        TThreadTask(PFunc func, TSharedData& data, size_t id)
            : Func_(func)
            , Data_(data)
            , Id_(id)
        {
        }

        void Process(void*) override {
            THolder<TThreadTask> This(this);

            (this->*Func_)();
        }

#define FAIL_ASSERT(cond)    \
    if (!(cond)) {           \
        Data_.failed = true; \
    }
        void RunBasics() {
            Data_.mutex.Acquire();

            ui32 oldCounter = ui32(Data_.sharedCounter + Id_);
            Data_.sharedCounter = oldCounter;
            usleep(10 + RandomNumber<ui32>() % 10);
            FAIL_ASSERT(Data_.sharedCounter == oldCounter);

            Data_.mutex.Release();
        }

        void RunFakeMutex() {
            bool res = Data_.fakeMutex.TryAcquire();
            FAIL_ASSERT(res);
        }

        void RunRecursiveMutex() {
            for (size_t i = 0; i < Id_ + 1; ++i) {
                Data_.mutex.Acquire();
                ++Data_.sharedCounter;
                usleep(1);
            }
            FAIL_ASSERT(Data_.sharedCounter == Id_ + 1);

            bool res = Data_.mutex.TryAcquire();
            FAIL_ASSERT(res);
            Data_.mutex.Release();

            for (size_t i = 0; i < Id_; ++i) {
                --Data_.sharedCounter;
                Data_.mutex.Release();
            }
            FAIL_ASSERT(Data_.sharedCounter == 1);
            --Data_.sharedCounter;
            Data_.mutex.Release();
        }

#undef FAIL_ASSERT

    private:
        PFunc Func_;
        TSharedData& Data_;
        size_t Id_;
    };

private:
#define RUN_CYCLE(what, count)                                              \
    Q_.Start(count);                                                        \
    for (size_t i = 0; i < count; ++i) {                                    \
        UNIT_ASSERT(Q_.Add(new TThreadTask(&TThreadTask::what, Data_, i))); \
    }                                                                       \
    Q_.Stop();                                                              \
    bool b = Data_.failed;                                                  \
    Data_.failed = false;                                                   \
    UNIT_ASSERT(!b);

    void TestBasics() {
        RUN_CYCLE(RunBasics, 5);

        UNIT_ASSERT(Data_.sharedCounter == 10);
        Data_.sharedCounter = 0;
    }

    void TestFake() {
        RUN_CYCLE(RunFakeMutex, 3);
    }

    void TestRecursive() {
        RUN_CYCLE(RunRecursiveMutex, 4);
    }

#undef RUN_CYCLE

private:
    TSharedData Data_;
    TThreadPool Q_;
};

UNIT_TEST_SUITE_REGISTRATION(TMutexTest)
