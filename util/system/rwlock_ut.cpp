#include "rwlock.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/thread/pool.h>
#include <util/random/random.h>

#include <atomic>

class TRWMutexTest: public TTestBase {
    UNIT_TEST_SUITE(TRWMutexTest);
    UNIT_TEST(TestReaders)
    UNIT_TEST(TestReadersWriters)
    UNIT_TEST_SUITE_END();

    struct TSharedData {
        std::atomic<size_t> writersIn = 0;
        std::atomic<size_t> readersIn = 0;

        bool failed = false;

        TRWMutex mutex;
    };

    class TThreadTask: public IObjectInQueue {
    public:
        using PFunc = void (TThreadTask::*)(void);

        TThreadTask(PFunc func, TSharedData& data, size_t id, size_t total)
            : Func_(func)
            , Data_(data)
            , Id_(id)
            , Total_(total)
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
        void RunReaders() {
            Data_.mutex.AcquireRead();

            ++Data_.readersIn;
            usleep(100);
            FAIL_ASSERT(Data_.readersIn.load() == Total_);
            usleep(100);
            --Data_.readersIn;

            Data_.mutex.ReleaseRead();
        }

        void RunReadersWriters() {
            if (Id_ % 2 == 0) {
                for (size_t i = 0; i < 10; ++i) {
                    Data_.mutex.AcquireRead();

                    ++Data_.readersIn;
                    FAIL_ASSERT(Data_.writersIn.load() == 0);
                    usleep(RandomNumber<ui32>() % 5);
                    --Data_.readersIn;

                    Data_.mutex.ReleaseRead();
                }
            } else {
                for (size_t i = 0; i < 10; ++i) {
                    Data_.mutex.AcquireWrite();

                    ++Data_.writersIn;
                    FAIL_ASSERT(Data_.readersIn.load() == 0 && Data_.writersIn.load() == 1);
                    usleep(RandomNumber<ui32>() % 5);
                    --Data_.writersIn;

                    Data_.mutex.ReleaseWrite();
                }
            }
        }
#undef FAIL_ASSERT

    private:
        PFunc Func_;
        TSharedData& Data_;
        size_t Id_;
        size_t Total_;
    };

private:
#define RUN_CYCLE(what, count)                                                     \
    Q_.Start(count);                                                               \
    for (size_t i = 0; i < count; ++i) {                                           \
        UNIT_ASSERT(Q_.Add(new TThreadTask(&TThreadTask::what, Data_, i, count))); \
    }                                                                              \
    Q_.Stop();                                                                     \
    bool b = Data_.failed;                                                         \
    Data_.failed = false;                                                          \
    UNIT_ASSERT(!b);

    void TestReaders() {
        RUN_CYCLE(RunReaders, 1);
    }

    void TestReadersWriters() {
        RUN_CYCLE(RunReadersWriters, 1);
    }

#undef RUN_CYCLE
private:
    TSharedData Data_;
    TThreadPool Q_;
};

UNIT_TEST_SUITE_REGISTRATION(TRWMutexTest)
