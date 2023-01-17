#include "mutex.h"
#include "guard.h"
#include "condvar.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/thread/pool.h>

#include <atomic>

class TCondVarTest: public TTestBase {
    UNIT_TEST_SUITE(TCondVarTest);
    UNIT_TEST(TestBasics)
    UNIT_TEST(TestSyncronize)
    UNIT_TEST_SUITE_END();

    struct TSharedData {
        TMutex mutex;
        TCondVar condVar1;
        TCondVar condVar2;

        std::atomic<bool> stopWaiting = false;

        std::atomic<size_t> in = 0;
        std::atomic<size_t> out = 0;

        std::atomic<size_t> waited = 0;

        bool failed = false;
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
                while (!Data_.stopWaiting.load()) {
                    bool res = Data_.condVar1.WaitT(Data_.mutex, TDuration::Seconds(1));
                    FAIL_ASSERT(res == true);
                }
            } else {
                usleep(100000);
                Data_.stopWaiting.store(true);

                TGuard<TMutex> guard(Data_.mutex);
                Data_.condVar1.Signal();
                Data_.condVar1.Signal();
            }
        }

        void RunBasicsWithPredicate() {
            Y_ASSERT(TotalIds_ == 3);

            if (Id_ < 2) {
                TGuard<TMutex> guard(Data_.mutex);
                const auto res = Data_.condVar1.WaitT(Data_.mutex, TDuration::Seconds(1), [&] {
                    return Data_.stopWaiting.load();
                });
                FAIL_ASSERT(res == true);
            } else {
                usleep(100000);
                Data_.stopWaiting.store(true);

                TGuard<TMutex> guard(Data_.mutex);
                Data_.condVar1.Signal();
                Data_.condVar1.Signal();
            }
        }

        void RunSyncronize() {
            for (size_t i = 0; i < 10; ++i) {
                TGuard<TMutex> guard(Data_.mutex);
                ++Data_.in;
                if (Data_.in.load() == TotalIds_) {
                    Data_.out.store(0);
                    Data_.condVar1.BroadCast();
                } else {
                    ++Data_.waited;
                    while (Data_.in.load() < TotalIds_) {
                        bool res = Data_.condVar1.WaitT(Data_.mutex, TDuration::Seconds(1));
                        FAIL_ASSERT(res == true);
                    }
                }

                ++Data_.out;
                if (Data_.out.load() == TotalIds_) {
                    Data_.in.store(0);
                    Data_.condVar2.BroadCast();
                } else {
                    while (Data_.out.load() < TotalIds_) {
                        bool res = Data_.condVar2.WaitT(Data_.mutex, TDuration::Seconds(1));
                        FAIL_ASSERT(res == true);
                    }
                }
            }

            FAIL_ASSERT(Data_.waited.load() == (TotalIds_ - 1) * 10);
        }

        void RunSyncronizeWithPredicate() {
            for (size_t i = 0; i < 10; ++i) {
                TGuard<TMutex> guard(Data_.mutex);
                ++Data_.in;
                if (Data_.in.load() == TotalIds_) {
                    Data_.out.store(0);
                    Data_.condVar1.BroadCast();
                } else {
                    ++Data_.waited;
                    const auto res = Data_.condVar1.WaitT(Data_.mutex, TDuration::Seconds(1), [&] {
                        return Data_.in.load() >= TotalIds_;
                    });
                    FAIL_ASSERT(res == true);
                }

                ++Data_.out;
                if (Data_.out.load() == TotalIds_) {
                    Data_.in.store(0);
                    Data_.condVar2.BroadCast();
                } else {
                    const auto res = Data_.condVar2.WaitT(Data_.mutex, TDuration::Seconds(1), [&] {
                        return Data_.out.load() >= TotalIds_;
                    });
                    FAIL_ASSERT(res == true);
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

    inline void TestBasicsWithPredicate() {
        RUN_CYCLE(RunBasicsWithPredicate, 3);
    }

    inline void TestSyncronize() {
        RUN_CYCLE(RunSyncronize, 6);
    }

    inline void TestSyncronizeWithPredicate() {
        RUN_CYCLE(RunSyncronizeWithPredicate, 6);
    }
#undef RUN_CYCLE
    TSharedData Data_;
    TThreadPool Q_;
};

UNIT_TEST_SUITE_REGISTRATION(TCondVarTest);
