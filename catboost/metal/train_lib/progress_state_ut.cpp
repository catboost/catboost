#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include "learn_cursor.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/str.h>

#include <cmath>

Y_UNIT_TEST_SUITE(TMetalTransientTrainingProgress) {
    Y_UNIT_TEST(OriginalRowsRestoredAfterComposedPreprocessing) {
        // The caller's Pool can itself contain duplicate source rows. Tracking
        // starts at that Pool's own positions, independent of feature storage.
        NCB::TArraySubsetIndexing<ui32> order(NCB::TFullSubset<ui32>(5));
        order = NCB::Compose(order, NCB::TArraySubsetIndexing<ui32>(NCB::TIndexedSubset<ui32>{2, 4, 0, 3, 1}));
        order = NCB::Compose(order, NCB::TArraySubsetIndexing<ui32>(NCB::TIndexedSubset<ui32>{4, 2, 0, 1, 3}));
        TVector<TVector<double>> cursor = {{20, 10, 30, 50, 40}, {2, 1, 3, 5, 4}};
        NCB::RestoreMetalLearnCursorOrder(order, &cursor);
        UNIT_ASSERT(cursor == TVector<TVector<double>>({{10, 20, 30, 40, 50}, {1, 2, 3, 4, 5}}));
    }

    Y_UNIT_TEST(InvalidMappingOrCursorDimensionsAreRejected) {
        TVector<TVector<double>> cursor = {{1, 2}};
        const NCB::TArraySubsetIndexing<ui32> duplicate(NCB::TIndexedSubset<ui32>{0, 0});
        UNIT_ASSERT_EXCEPTION_CONTAINS(NCB::RestoreMetalLearnCursorOrder(duplicate, &cursor), TCatBoostException,
            "must be a permutation");
        const NCB::TArraySubsetIndexing<ui32> wrongSize(NCB::TFullSubset<ui32>(3));
        UNIT_ASSERT_EXCEPTION_CONTAINS(NCB::RestoreMetalLearnCursorOrder(wrongSize, &cursor), TCatBoostException,
            "differs from cursor dimensions");
    }

    Y_UNIT_TEST(UnavailableCursorRemainsUnavailable) {
        TVector<TVector<double>> cursor;
        NCB::RestoreMetalLearnCursorOrder(NCB::TArraySubsetIndexing<ui32>(NCB::TFullSubset<ui32>(5)), &cursor);
        UNIT_ASSERT(cursor.empty());
    }

    Y_UNIT_TEST(TrainingCursorAndDiagnosticsNeverEnterSerializedHistory) {
        TMetricsAndTimeLeftHistory history;
        history.LearnMetricsHistory = {{{"RMSE", 1.5}}};
        history.TestMetricsHistory = {{{{"RMSE", 2.5}}}};
        history.BestIteration = 0;
        history.LearnBestError = {{"RMSE", 1.5}};
        history.TestBestError = {{{"RMSE", 2.5}}};
        TTimeInfo time;
        time.IterationTime = .125;
        time.PassedTime = .125;
        time.RemainingTime = .875;
        history.TimeHistory.push_back(time);

        TStringStream originalBytes;
        ::Save(&originalBytes, history);
        const auto originalMetadata = history.SaveMetrics();

        // Use several dimensions and conspicuous values, so accidentally
        // serializing even one transient field changes the byte comparison.
        history.MetalLearnCursor = {{123456.25, -42.5, .125}, {7.5, 987654.5, -.75}};
        history.MetalInitialLoss = 7654321.25;
        history.MetalObjectiveMetric = "QuerySoftMax:beta=0.375";
        history.MetalResumedIterations = 12345;
        history.MetalKernelDispatches = 987654321;
        history.MetalGpuSeconds = 1234.25;
        TStringStream withCursorBytes;
        ::Save(&withCursorBytes, history);
        UNIT_ASSERT_VALUES_EQUAL(originalBytes.Str(), withCursorBytes.Str());
        UNIT_ASSERT(originalMetadata == history.SaveMetrics());

        TMetricsAndTimeLeftHistory restored;
        ::Load(&withCursorBytes, restored);
        UNIT_ASSERT(restored.MetalLearnCursor.empty());
        UNIT_ASSERT(std::isnan(restored.MetalInitialLoss));
        UNIT_ASSERT(restored.MetalObjectiveMetric.empty());
        UNIT_ASSERT_VALUES_EQUAL(restored.MetalResumedIterations, 0);
        UNIT_ASSERT_VALUES_EQUAL(restored.MetalKernelDispatches, 0);
        UNIT_ASSERT_VALUES_EQUAL(restored.MetalGpuSeconds, 0);
        UNIT_ASSERT(restored.LearnMetricsHistory == history.LearnMetricsHistory);
        UNIT_ASSERT(restored.TestMetricsHistory == history.TestMetricsHistory);
        UNIT_ASSERT_VALUES_EQUAL(restored.TimeHistory.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(restored.TimeHistory[0].PassedTime, .125);
        UNIT_ASSERT(restored.SaveMetrics() == originalMetadata);
    }
}
