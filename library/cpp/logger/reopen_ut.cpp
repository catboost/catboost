#include "backend.h"
#include "reopen.h"

#include <exception>
#include <util/generic/string.h>
#include <util/generic/array_ref.h>

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/testing/unittest/tests_data.h>

#include <future>
#include <atomic>
#include <barrier>

namespace {

struct TMockLogBackend : public TLogBackend {
    void WriteData(const TLogRecord& rec) override {
        BytesWritten.fetch_add(rec.Len);
    }

    void ReopenLog() override {
        NumReopens.fetch_add(1);
    }

    std::atomic<ui64> BytesWritten{0};
    std::atomic<ui64> NumReopens{0};
};

void WriteData(TReopenLogBackend& log, const TString& data) {
    log.WriteData(TLogRecord(ELogPriority::TLOG_INFO, data.data(), data.size()));
}

}

Y_UNIT_TEST_SUITE(ReopenLogSuite) {
    Y_UNIT_TEST(TestSimple) {
        constexpr ui64 limit = 5;
        const auto testData = {"test", "dshkafhuadshfiasuh", "log", "data"};
        constexpr ui64 expectedReopens = 2; // considering the limit, the first reopen after the second string and one more at the end

        auto mockHolder = MakeHolder<TMockLogBackend>();
        auto& mock = *mockHolder;
        TReopenLogBackend log(std::move(mockHolder), limit);

        ui64 expectedWritten = 0;
        for (const TString str : testData) {
            WriteData(log, str);
            expectedWritten += str.size();
        }

        UNIT_ASSERT(mock.BytesWritten.load() == expectedWritten);
        UNIT_ASSERT(mock.NumReopens.load() == expectedReopens);
    }

    Y_UNIT_TEST(TestSingleThreaded) {
        constexpr ui64 limit = 1_KB;
        constexpr ui64 numLogs = 123;
        constexpr ui64 logSize = 1_KB / 4;

        static_assert((limit / logSize) * logSize == limit); // should be divisible for this test
        constexpr ui64 expectedWritten = numLogs * logSize;
        constexpr ui64 expectedReopens = expectedWritten / limit;

        auto mockHolder = MakeHolder<TMockLogBackend>();
        auto& mock = *mockHolder;
        TReopenLogBackend log(std::move(mockHolder), limit);

        for (ui64 i = 0; i < numLogs; ++i) {
            WriteData(log, TString(logSize, 'a'));
        }

        UNIT_ASSERT(mock.BytesWritten.load() == expectedWritten);
        UNIT_ASSERT(mock.NumReopens.load() == expectedReopens);
    }

    Y_UNIT_TEST(TestMultiThreaded) {
        constexpr ui64 limit = 1_KB;
        constexpr ui64 numLogsPerThread = 123;
        constexpr ui64 numThreads = 12;
        constexpr ui64 logSize = 1_KB / 4;

        static_assert((limit / logSize) * logSize == limit); // should be divisible for this test
        constexpr ui64 expectedWritten = numLogsPerThread * numThreads * logSize;

         // can't guarantee consistent number of reopens every N bytes in multithreaded setting
        constexpr ui64 minExpectedReopens = limit / logSize;
        constexpr ui64 maxExpectedReopens = expectedWritten / limit;

        auto mockHolder = MakeHolder<TMockLogBackend>();
        auto& mock = *mockHolder;
        TReopenLogBackend log(std::move(mockHolder), limit);

        std::barrier barrier(numThreads);
        const auto job = [&]() {
            barrier.arrive_and_wait();

            for (ui64 i = 0; i < numLogsPerThread; ++i) {
                WriteData(log, TString(logSize, 'a'));
            }
        };

        std::vector<std::future<void>> jobs;
        for (ui64 i = 0; i < numThreads; ++i) {
            jobs.emplace_back(std::async(std::launch::async, job));
        }
        for (auto& res : jobs) {
            res.wait();
        }

        UNIT_ASSERT(mock.BytesWritten.load() == expectedWritten);
        UNIT_ASSERT(mock.NumReopens.load() >= minExpectedReopens);
        UNIT_ASSERT(mock.NumReopens.load() <= maxExpectedReopens);
    }

    Y_UNIT_TEST(TestZeroThrows) {
        UNIT_ASSERT_EXCEPTION(TReopenLogBackend(MakeHolder<TMockLogBackend>(), 0), std::exception);
    }
}
