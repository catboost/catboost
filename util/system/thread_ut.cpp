#include "thread.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TSysThreadTest) {
    struct TIdTester {
        inline TIdTester()
            : Thr(nullptr)
            , Cur(0)
            , Real(0)
        {
        }

        static inline void* DoRun(void* ptr) {
            ((TIdTester*)ptr)->Run();

            return nullptr;
        }

        inline void Run() {
            Cur = TThread::CurrentThreadId();
            Real = Thr->Id();
        }

        TThread* Thr;
        TThread::TId Cur;
        TThread::TId Real;
    };

    SIMPLE_UNIT_TEST(TestThreadId) {
        TIdTester tst;
        TThread thr(tst.DoRun, &tst);

        tst.Thr = &thr;

        thr.Start();
        thr.Join();

        UNIT_ASSERT_EQUAL(tst.Cur, tst.Real);
        UNIT_ASSERT(tst.Cur != 0);
    }

    void* ThreadProc(void*) {
        TThread::CurrentThreadSetName("CurrentThreadSetNameTest");
        return nullptr;
    }

    void* ThreadProc2(void*) {
        return nullptr;
    }

    SIMPLE_UNIT_TEST(TestSetThreadName) {
        TThread thread(&ThreadProc, nullptr);
        // just check it doesn't crash
        thread.Start();
        thread.Join();
    }

    SIMPLE_UNIT_TEST(TestSetThreadName2) {
        TThread thread(TThread::TParams(&ThreadProc, nullptr, 0).SetName("XXX"));

        thread.Start();
        thread.Join();
    }

    SIMPLE_UNIT_TEST(TestDoubleJoin) {
        TThread thread(&ThreadProc, nullptr);

        thread.Start();
        thread.Join();

        UNIT_ASSERT_EQUAL(thread.Join(), nullptr);
    }

    SIMPLE_UNIT_TEST(TestDoubleStart) {
        TThread thread(&ThreadProc, nullptr);

        thread.Start();
        UNIT_ASSERT_EXCEPTION(thread.Start(), yexception);
        thread.Join();
    }

    SIMPLE_UNIT_TEST(TestNoStart) {
        TThread thread(&ThreadProc, nullptr);
    }

    SIMPLE_UNIT_TEST(TestNoStartJoin) {
        TThread thread(&ThreadProc, nullptr);

        UNIT_ASSERT_EQUAL(thread.Join(), nullptr);
    }

    SIMPLE_UNIT_TEST(TestStackPointer) {
        TArrayHolder<char> buf(new char[64000]);
        TThread thr(TThread::TParams(ThreadProc2, nullptr).SetStackPointer(buf.Get()).SetStackSize(64000));

        thr.Start();
        UNIT_ASSERT_VALUES_EQUAL(thr.Join(), nullptr);
    }

    SIMPLE_UNIT_TEST(TestStackLimits) {
        TCurrentThreadLimits sl;

        UNIT_ASSERT(sl.StackBegin);
        UNIT_ASSERT(sl.StackLength > 0);
    }
};
