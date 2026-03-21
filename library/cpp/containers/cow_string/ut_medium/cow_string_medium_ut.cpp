#include <library/cpp/containers/cow_string/cow_string.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/strbuf.h>
#include <util/generic/yexception.h>
#include <util/stream/output.h>
#include <util/system/thread.h>

#include <string>
#include <barrier>

static_assert(sizeof(TCowString) == sizeof(const char*), "expect sizeof(TCowString) == sizeof(const char*)");

Y_UNIT_TEST_SUITE(CowPitfalls) {
    Y_UNIT_TEST(ParallelDetach) {
        // best results with thread-sanitizer
        std::vector<std::unique_ptr<TThread>> threads;
        TCowString a = "the string";
        TCowString b = a;
        auto makeRefToA = [&a, &b]() {
            b = a; // make second reference to the same string
        };
        constexpr int nThreads = 8;
#ifdef _tsan_enabled_
        constexpr i64 retries = 1'000;
#else
        constexpr i64 retries = 1'000'000;
#endif
        std::barrier iterationSyncPoint(nThreads, makeRefToA);
        std::atomic<i64> totalLen = 0;
        auto addLen = [](std::string a, std::atomic<i64>& len) {
            len += a.length();
        };
        auto workload = [&a, &addLen, &totalLen, &iterationSyncPoint]() {
            std::atomic<i64> len = 0;
            for (i64 j = 0; j < retries; ++j) {
                addLen(a, len); // possibility of bad implicit conversion
                iterationSyncPoint.arrive_and_wait();
            }
            totalLen += len.load();
        };
        for (int i = 0; i < nThreads; ++i) {
            threads.push_back(std::make_unique<TThread>(workload));
        }
        for (auto& t : threads) {
            t->Start();
        }
        for (auto& t : threads) {
            t->Join();
        }
        UNIT_ASSERT_VALUES_EQUAL(totalLen.load(), b.size() * nThreads * retries);
    }

} // Y_UNIT_TEST_SUITE(CowPitfalls)
