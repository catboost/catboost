#include "mutex.h"
#include "guard.h"
#include "condvar.h"

#include <library/unittest/registar.h>

#include <util/thread/queue.h>

class TCondVarTest: public TTestBase {
    UNIT_TEST_SUITE(TCondVarTest);
    UNIT_TEST(TestBasics)
    UNIT_TEST(TestSyncronize)
    UNIT_TEST_SUITE_END();

    struct TSharedData {
        TSharedData()
            : stopWaiting(false)
            , in(0)
            , out(0)
            , waited(0)
            , failed(false)
        {
        }

        TMutex mutex;
        TCondVar condVar1;
        TCondVar condVar2;

        volatile bool stopWaiting;

        volatile size_t in;
        volatile size_t out;

        volatile size_t waited;

        bool failed;
    };

    class TThreadTask: public IObjectInQueue {
    public:
        using PFunc = void (TThreadTask::*)(void);

        TThreadTask(PFunc func, size_t id, size_t totalIds, TSharedData& data)
            : Func_(func)
            , Id_(id)
            , TotalIds_(totalIds)
            , Data_(data)
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
            Y_ASSERT(TotalIds_ == 3);

            if (Id_ < 2) {
                TGuard<TMutex> guard(Data_.mutex);
                while (!Data_.stopWaiting) {
                    bool res = Data_.condVar1.TimedWait(Data_.mutex, TDuration::Seconds(1));
                    FAIL_ASSERT(res == true);
                }
            } else {
                usleep(100000);
                Data_.stopWaiting = true;

                TGuard<TMutex> guard(Data_.mutex);
                Data_.condVar1.Signal();
                Data_.condVar1.Signal();
            }
        }

        void RunSyncronize() {
            for (size_t i = 0; i < 10; ++i) {
                TGuard<TMutex> guard(Data_.mutex);
                ++Data_.in;
                if (Data_.in == TotalIds_) {
                    Data_.out = 0;
                    Data_.condVar1.BroadCast();
                } else {
                    ++Data_.waited;
                    while (Data_.in < TotalIds_) {
                        bool res = Data_.condVar1.TimedWait(Data_.mutex, TDuration::Seconds(1));
                        FAIL_ASSERT(res == true);
                    }
                }

                ++Data_.out;
                if (Data_.out == TotalIds_) {
                    Data_.in = 0;
                    Data_.condVar2.BroadCast();
                } else {
                    while (Data_.out < TotalIds_) {
                        bool res = Data_.condVar2.TimedWait(Data_.mutex, TDuration::Seconds(1));
                        FAIL_ASSERT(res == true);
                    }
                }
            }

            FAIL_ASSERT(Data_.waited == (TotalIds_ - 1) * 10);
        }
#undef FAIL_ASSERT

    private:
        PFunc Func_;
        size_t Id_;
        size_t TotalIds_;
        TSharedData& Data_;
    };

private:
#define RUN_CYCLE(what, count)                                                     \
    Q_.Start(count);                                                               \
    for (size_t i = 0; i < count; ++i) {                                           \
        UNIT_ASSERT(Q_.Add(new TThreadTask(&TThreadTask::what, i, count, Data_))); \
    }                                                                              \
    Q_.Stop();                                                                     \
    bool b = Data_.failed;                                                         \
    Data_.failed = false;                                                          \
    UNIT_ASSERT(!b);

    inline void TestBasics() {
        RUN_CYCLE(RunBasics, 3);
    }

    inline void TestSyncronize() {
        RUN_CYCLE(RunSyncronize, 6);
    }
#undef RUN_CYCLE
    TSharedData Data_;
    TMtpQueue Q_;
};

UNIT_TEST_SUITE_REGISTRATION(TCondVarTest);
