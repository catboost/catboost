#include "thread.h"

#include <library/cpp/testing/unittest/registar.h>

#include <atomic>

Y_UNIT_TEST_SUITE(TSysThreadTest) {
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
            Numeric = TThread::CurrentThreadNumericId();
        }

        TThread* Thr;
        TThread::TId Cur;
        TThread::TId Real;
        TThread::TId Numeric;
    };

    Y_UNIT_TEST(TestThreadId) {
        TIdTester tst;
        TThread thr(tst.DoRun, &tst);

        tst.Thr = &thr;

        thr.Start();
        thr.Join();

        UNIT_ASSERT_EQUAL(tst.Cur, tst.Real);
        UNIT_ASSERT(tst.Cur != 0);
        UNIT_ASSERT(tst.Numeric != 0);
        UNIT_ASSERT(tst.Numeric != tst.Real);
    }

    void* ThreadProc(void*) {
        TThread::SetCurrentThreadName("CurrentThreadSetNameTest");
        return nullptr;
    }

    void* ThreadProc2(void*) {
        return nullptr;
    }

    void* ThreadProc3(void*) {
        const auto name = TThread::CurrentThreadName();
        Y_FAKE_READ(name);
        return nullptr;
    }

    void* ThreadProc4(void*) {
        const TString setName = "ThreadName";
        TThread::SetCurrentThreadName(setName.data());

        const auto getName = TThread::CurrentThreadName();
        if (TThread::CanGetCurrentThreadName()) {
            UNIT_ASSERT_VALUES_EQUAL(setName, getName);
        } else {
            UNIT_ASSERT_VALUES_EQUAL("", getName);
        }
        return nullptr;
    }

    void* ThreadProcChild(void*) {
        const auto name = TThread::CurrentThreadName();
        const auto defaultName = GetProgramName();

        (void)name;
        (void)defaultName;

#if defined(_darwin_) || defined(_linux_)
        UNIT_ASSERT_VALUES_EQUAL(name, defaultName);
#endif
        return nullptr;
    }

    void* ThreadProcParent(void*) {
        const TString setName = "Parent";
        TThread::SetCurrentThreadName(setName.data());

        TThread thread(&ThreadProcChild, nullptr);

        thread.Start();
        thread.Join();

        const auto getName = TThread::CurrentThreadName();
        if (TThread::CanGetCurrentThreadName()) {
            UNIT_ASSERT_VALUES_EQUAL(setName, getName);
        } else {
            UNIT_ASSERT_VALUES_EQUAL("", getName);
        }
        return nullptr;
    }

    Y_UNIT_TEST(TestSetThreadName) {
        TThread thread(&ThreadProc, nullptr);
        // just check it doesn't crash
        thread.Start();
        thread.Join();
    }

    Y_UNIT_TEST(TestSetThreadName2) {
        TThread thread(TThread::TParams(&ThreadProc, nullptr, 0).SetName("XXX"));

        thread.Start();
        thread.Join();
    }

    Y_UNIT_TEST(TestGetThreadName) {
        TThread thread(&ThreadProc3, nullptr);
        thread.Start();
        thread.Join();
    }

    Y_UNIT_TEST(TestSetGetThreadName) {
        TThread thread(&ThreadProc4, nullptr);
        thread.Start();
        thread.Join();
    }

    Y_UNIT_TEST(TestSetGetThreadNameInChildThread) {
        TThread thread(&ThreadProcParent, nullptr);
        thread.Start();
        thread.Join();
    }

    Y_UNIT_TEST(TestDoubleJoin) {
        TThread thread(&ThreadProc, nullptr);

        thread.Start();
        thread.Join();

        UNIT_ASSERT_EQUAL(thread.Join(), nullptr);
    }

    Y_UNIT_TEST(TestDoubleStart) {
        TThread thread(&ThreadProc, nullptr);

        thread.Start();
        UNIT_ASSERT_EXCEPTION(thread.Start(), yexception);
        thread.Join();
    }

    Y_UNIT_TEST(TestNoStart) {
        TThread thread(&ThreadProc, nullptr);
    }

    Y_UNIT_TEST(TestNoStartJoin) {
        TThread thread(&ThreadProc, nullptr);

        UNIT_ASSERT_EQUAL(thread.Join(), nullptr);
    }

    Y_UNIT_TEST(TestStackPointer) {
        TArrayHolder<char> buf(new char[64000]);
        TThread thr(TThread::TParams(ThreadProc2, nullptr).SetStackPointer(buf.Get()).SetStackSize(64000));

        thr.Start();
        UNIT_ASSERT_VALUES_EQUAL(thr.Join(), nullptr);
    }

    Y_UNIT_TEST(TestStackLimits) {
        TCurrentThreadLimits sl;

        UNIT_ASSERT(sl.StackBegin);
        UNIT_ASSERT(sl.StackLength > 0);
    }

    Y_UNIT_TEST(TestFunc) {
        std::atomic_bool flag = {false};
        TThread thread([&flag]() { flag = true; });

        thread.Start();
        UNIT_ASSERT_VALUES_EQUAL(thread.Join(), nullptr);
        UNIT_ASSERT(flag);
    }

    Y_UNIT_TEST(TestCopyFunc) {
        std::atomic_bool flag = {false};
        auto func = [&flag]() { flag = true; };

        TThread thread(func);
        thread.Start();
        UNIT_ASSERT_VALUES_EQUAL(thread.Join(), nullptr);

        TThread thread2(func);
        thread2.Start();
        UNIT_ASSERT_VALUES_EQUAL(thread2.Join(), nullptr);

        UNIT_ASSERT(flag);
    }

    Y_UNIT_TEST(TestCallable) {
        std::atomic_bool flag = {false};

        struct TCallable: TMoveOnly {
            std::atomic_bool* Flag_;

            TCallable(std::atomic_bool* flag)
                : Flag_(flag)
            {
            }

            void operator()() {
                *Flag_ = true;
            }
        };

        TCallable foo(&flag);
        TThread thread(std::move(foo));

        thread.Start();
        UNIT_ASSERT_VALUES_EQUAL(thread.Join(), nullptr);
        UNIT_ASSERT(flag);
    }
} // Y_UNIT_TEST_SUITE(TSysThreadTest)
