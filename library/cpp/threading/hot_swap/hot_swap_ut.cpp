#include "hot_swap.h"

#include <library/cpp/unittest/registar.h>

#include <util/datetime/base.h>
#include <util/random/random.h>
#include <util/system/info.h>
#include <util/system/thread.h>

class TDataMock: public TAtomicRefCount<TDataMock> {
public:
    TDataMock() {
        AtomicIncrement(CreatedObjects);
    }

    virtual ~TDataMock() {
        AtomicSet(LiveFlag, 0);
        AtomicIncrement(DestroyedObjects);
    }

    void CheckLive() {
        UNIT_ASSERT(AtomicGet(LiveFlag));
    }

    static void CheckBalance() {
        UNIT_ASSERT_VALUES_EQUAL(AtomicGet(CreatedObjects), AtomicGet(DestroyedObjects));
    }

public:
    static TAtomic CreatedObjects;
    static TAtomic DestroyedObjects;
    TAtomic LiveFlag = 1;
};

using TTestHotSwap = THotSwap<TDataMock>;

class TBalanceChecker {
public:
    TBalanceChecker() {
        TDataMock::CheckBalance();
    }
    ~TBalanceChecker() {
        TDataMock::CheckBalance();
    }
};

class TTestThread: public ISimpleThread {
public:
    TTestThread(TTestHotSwap& ptr, double updateProbability, double nullptrProbability, TDuration workTime)
        : Ptr(ptr)
        , UpdateProbability(updateProbability)
        , NullptrProbability(nullptrProbability)
        , WorkUntil(workTime.ToDeadLine())
    {
    }

    void* ThreadProc() noexcept override {
        while (TInstant::Now() <= WorkUntil) {
            const bool update = RandomNumber<double>() < UpdateProbability;
            if (update) {
                const bool null = NullptrProbability ? RandomNumber<double>() < NullptrProbability : false;
                if (null)
                    Ptr.AtomicStore(nullptr);
                else
                    Ptr.AtomicStore(new TDataMock());
            } else {
                TTestHotSwap::TPtr obj = Ptr.AtomicLoad();
                if (obj)
                    obj->CheckLive();
            }
        }
        return nullptr;
    }

private:
    TTestHotSwap& Ptr;
    double UpdateProbability, NullptrProbability;
    TInstant WorkUntil;
};

TAtomic TDataMock::CreatedObjects = 0;
TAtomic TDataMock::DestroyedObjects = 0;

Y_UNIT_TEST_SUITE(THotSwapTest) {
    Y_UNIT_TEST(NullPtrTest) {
        TBalanceChecker checkBalance;
        {
            TTestHotSwap ptr;

            // Default init
            {
                TTestHotSwap::TPtr p = ptr.AtomicLoad();
                TTestHotSwap::TPtr p2 = p;
                UNIT_ASSERT(!p);
                UNIT_ASSERT(!p2);
                UNIT_ASSERT_EQUAL(p, p2);
            }

            // Explicitly update to nullptr
            {
                ptr.AtomicStore(TTestHotSwap::TPtr(new TDataMock()));
                ptr.AtomicStore(nullptr);
                TTestHotSwap::TPtr p = ptr.AtomicLoad();
                TTestHotSwap::TPtr p2 = p;
                UNIT_ASSERT(!p);
                UNIT_ASSERT(!p2.Get());
                UNIT_ASSERT_EQUAL(p, p2);
            }
        }
    }

    Y_UNIT_TEST(CopyAndMoveTest) {
        TBalanceChecker checkBalance;

        TTestHotSwap ptr(new TDataMock());
        UNIT_ASSERT(ptr.AtomicLoad());

        TTestHotSwap ptr2(std::move(ptr));
        UNIT_ASSERT(!ptr.AtomicLoad());
        UNIT_ASSERT(ptr2.AtomicLoad());

        TTestHotSwap ptr3(ptr2);
        UNIT_ASSERT(ptr2.AtomicLoad());
        UNIT_ASSERT(ptr3.AtomicLoad());

        TTestHotSwap ptr4;
        ptr4 = ptr3;
        UNIT_ASSERT(ptr3.AtomicLoad());
        UNIT_ASSERT(ptr4.AtomicLoad());

        UNIT_ASSERT_EQUAL(ptr3.AtomicLoad(), ptr4.AtomicLoad());
        UNIT_ASSERT_EQUAL(4, ptr3.AtomicLoad().RefCount());

        TTestHotSwap ptr5(TTestHotSwap::TPtr(new TDataMock()));
        UNIT_ASSERT_EQUAL(2, ptr5.AtomicLoad().RefCount());
    }

    void DoMultithreadedTest(double updateProbability, double nullptrProbability) {
        TBalanceChecker checkBalance;
        TTestHotSwap ptr(new TDataMock());

        const TDuration workTime = TDuration::Seconds(1);
        TVector<TSimpleSharedPtr<TTestThread>> threads(NSystemInfo::CachedNumberOfCpus());
        for (TSimpleSharedPtr<TTestThread>& thread : threads) {
            thread = new TTestThread(ptr, updateProbability, nullptrProbability, workTime);
            thread->Start();
        }

        for (TSimpleSharedPtr<TTestThread>& thread : threads)
            thread->Join();
    }

    Y_UNIT_TEST(MultithreadedTest) {
        DoMultithreadedTest(0.5, 0);
        DoMultithreadedTest(0.999, 0);
        DoMultithreadedTest(0.001, 0);
        DoMultithreadedTest(0.5, 0.5);
    }
}
