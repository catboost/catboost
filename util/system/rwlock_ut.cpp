#include "rwlock.h"
#include "atomic.h"

#include <library/cpp/unittest/registar.h>

#include <util/thread/pool.h>
#include <util/random/random.h>

class TRWMutexTest: public TTestBase {
    UNIT_TEST_SUITE(TRWMutexTest);
    UNIT_TEST(TestReaders)
    UNIT_TEST(TestReadersWriters)
    UNIT_TEST_SUITE_END();

    struct TSharedData {
        TSharedData()
            : writersIn(0)
            , readersIn(0)
            , failed(false)
        {
        }

        TAtomic writersIn;
        TAtomic readersIn;

        bool failed;

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

            AtomicIncrement(Data_.readersIn);
            usleep(100);
            FAIL_ASSERT(Data_.readersIn == long(Total_));
            usleep(100);
            AtomicDecrement(Data_.readersIn);

            Data_.mutex.ReleaseRead();
        }

        void RunReadersWriters() {
            if (Id_ % 2 == 0) {
                for (size_t i = 0; i < 10; ++i) {
                    Data_.mutex.AcquireRead();

                    AtomicIncrement(Data_.readersIn);
                    FAIL_ASSERT(Data_.writersIn == 0);
                    usleep(RandomNumber<ui32>() % 5);
                    AtomicDecrement(Data_.readersIn);

                    Data_.mutex.ReleaseRead();
                }
            } else {
                for (size_t i = 0; i < 10; ++i) {
                    Data_.mutex.AcquireWrite();

                    AtomicIncrement(Data_.writersIn);
                    FAIL_ASSERT(Data_.readersIn == 0 && Data_.writersIn == 1);
                    usleep(RandomNumber<ui32>() % 5);
                    AtomicDecrement(Data_.writersIn);

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
