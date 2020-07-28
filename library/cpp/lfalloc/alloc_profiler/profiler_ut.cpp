#include "profiler.h"

#include <library/cpp/testing/unittest/registar.h>

namespace NAllocProfiler {

////////////////////////////////////////////////////////////////////////////////

Y_UNIT_TEST_SUITE(Profiler) {
    Y_UNIT_TEST(StackCollection)
    {
        TStringStream str;

        NAllocProfiler::StartAllocationSampling(true);
        TVector<TAutoPtr<int>> test;
        // Do many allocations and no deallocations
        for (int i = 0; i < 10000; ++i) {
            test.push_back(new int);
        }
        NAllocProfiler::StopAllocationSampling(str);
        //Cout << str.Str() << Endl;

#if !defined(ARCH_AARCH64)
        /* Check that output resembles this:

            STACK #2: 0     Allocs: 10      Frees: 0        CurrentSize: 40
            0000000000492353        ??
            000000000048781F        operator new(unsigned long) +1807
            00000000003733FA        NAllocProfiler::NTestSuiteProfiler::TTestCaseStackCollection::Execute_(NUnitTest::TTestContext&) +218
            00000000004A1938        NUnitTest::TTestBase::Run(std::__y1::function<void ()>, TString, char const*, bool) +120
            0000000000375656        NAllocProfiler::NTestSuiteProfiler::TCurrentTest::Execute() +342
            00000000004A20CF        NUnitTest::TTestFactory::Execute() +847
            000000000049922D        NUnitTest::RunMain(int, char**) +1965
            00007FF665778F45        __libc_start_main +245
        */

        UNIT_ASSERT_STRING_CONTAINS(str.Str(), "StackCollection");
        UNIT_ASSERT_STRING_CONTAINS(str.Str(), "NUnitTest::TTestBase::Run");
        UNIT_ASSERT_STRING_CONTAINS(str.Str(), "NAllocProfiler::NTestSuiteProfiler::TCurrentTest::Execute");
        UNIT_ASSERT_STRING_CONTAINS(str.Str(), "NUnitTest::TTestFactory::Execute");
        UNIT_ASSERT_STRING_CONTAINS(str.Str(), "NUnitTest::RunMain");
#endif
    }

    class TAllocDumper : public NAllocProfiler::TAllocationStatsDumper {
    public:
        explicit TAllocDumper(IOutputStream& out) : NAllocProfiler::TAllocationStatsDumper(out) {}

        TString FormatTag(int tag) override {
            UNIT_ASSERT_VALUES_EQUAL(tag, 42);
            return "TAG_NAME_42";
        }
    };

    Y_UNIT_TEST(TagNames)
    {
        TStringStream str;

        NAllocProfiler::StartAllocationSampling(true);
        TVector<TAutoPtr<int>> test;
        NAllocProfiler::TProfilingScope scope(42);
        // Do many allocations and no deallocations
        for (int i = 0; i < 10000; ++i) {
            test.push_back(new int);
        }

        TAllocDumper dumper(str);
        NAllocProfiler::StopAllocationSampling(dumper);

#if !defined(ARCH_AARCH64)
        UNIT_ASSERT_STRING_CONTAINS(str.Str(), "TAG_NAME_42");
#endif
    }
}

}
