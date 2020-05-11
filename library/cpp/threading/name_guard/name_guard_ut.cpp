#include <library/cpp/threading/name_guard/name_guard.h>
#include <library/cpp/unittest/registar.h>

#include <util/system/thread.h>
#include <util/thread/factory.h>

Y_UNIT_TEST_SUITE(ThreadNameGuardTests) {
    Y_UNIT_TEST(Test) {
        const TString nameBefore = "nameBefore";
        const TString nameToSet = "nameToSet";
        SystemThreadFactory()->Run([&] {
            TThread::SetCurrentThreadName(nameBefore.c_str());

            {
                Y_THREAD_NAME_GUARD(nameToSet);
                const auto name = TThread::CurrentThreadName();

                UNIT_ASSERT_VALUES_EQUAL(nameToSet, name);
            }

            const auto name = TThread::CurrentThreadName();
            UNIT_ASSERT_VALUES_EQUAL(nameBefore, name);
        })->Join();
    }
}
