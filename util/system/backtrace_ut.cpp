#include "backtrace.h"

#include <util/generic/array_ref.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/output.h>

using PFunc = int (*)(void**, size_t);

int Dbg1(void** buf, size_t len) {
    volatile int ret = (int)BackTrace(buf, len);
    return ret;
}

int Dbg2(void** buf, size_t len) {
    volatile int ret = (int)BackTrace(buf, len);
    return ret;
}

void FormatBackTraceReplacement(IOutputStream* out, void* const*, size_t) {
    *out << "WorksLikeACharm" << Endl;
}

void SomeMethod() {
    TStringStream out;

    FormatBackTrace(&out);

#if defined(_musl_)
    // musl dladdr broken for us now
    return;
#endif

    UNIT_ASSERT(out.Empty() || out.Str().find("SomeMethod") != TString::npos);
}

class TBackTraceTest: public TTestBase {
    UNIT_TEST_SUITE(TBackTraceTest);
    UNIT_TEST(TestBackTrace)
    UNIT_TEST(TestBackTraceView)
    UNIT_TEST(TestPrintBackTrace)
    UNIT_TEST(TestSetFormatBackTraceFn)
    UNIT_TEST_SUITE_END();

    void TestPrintBackTrace() {
        SomeMethod();
    }

    void TestSetFormatBackTraceFn() {
        TFormatBackTraceFn prevFn = SetFormatBackTraceFn(FormatBackTraceReplacement);
        TStringStream out;
        FormatBackTrace(&out);
        SetFormatBackTraceFn(prevFn);
        UNIT_ASSERT(out.Str().Contains("WorksLikeACharm"));
        TestPrintBackTrace();
    }

    void TestBackTrace() {
        //PrintBackTrace();
        void* buf1[100];
        size_t ret1;

        void* buf2[100];
        size_t ret2;

        volatile PFunc func = &Dbg1;
        ret1 = (*func)(buf1, 100);
        func = &Dbg2;
        ret2 = (*func)(buf2, 100);

        UNIT_ASSERT_EQUAL(ret1, ret2);
    }

    void TestBackTraceView() {
        try {
            throw TWithBackTrace<yexception>();
        } catch (const yexception& e) {
            const TBackTrace bt = *e.BackTrace();
            const TBackTraceView btView = bt;
            UNIT_ASSERT_VALUES_EQUAL(btView.size(), bt.size());
        }
    }
};

UNIT_TEST_SUITE_REGISTRATION(TBackTraceTest);
