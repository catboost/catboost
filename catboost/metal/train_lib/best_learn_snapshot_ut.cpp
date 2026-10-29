#include "snapshot.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/file.h>
#include <util/string/cast.h>
#include <util/system/tempfile.h>

#include <limits>

using namespace NCB;

namespace {
    constexpr ui32 LegacyCursorTag = 0x4D424C31;
    constexpr ui32 IndexedCursorTag = 0x4D424C32;

    struct TSnapshotFile {
        TTempFile File{MakeTempName()};
        TMetalSnapshot Base;

        TSnapshotFile() {
            NJson::TJsonValue options;
            NCatboostOptions::TCatBoostOptions(ETaskType::GPU).Save(&options);
            Base.Params = ToString(options);
            UNIT_ASSERT(NCatboostOptions::IsParamsCompatible(Base.Params, Base.Params));
            Base.Checksum = 0x13579BDF;
            Base.Depths = {1, 1};
            Base.SplitFeatures = {0, 1};
            Base.SplitBins = {1, 2};
            Base.SplitTypes = {0, 0};
            Base.Leaves = {-0.1f, 0.2f, 0.15f, -0.25f};
            Base.Weights = {1, 2, 2, 1};
            Base.Predictions = {0.2f, -0.3f, 0.4f};
            Base.PermutationPredictions = Base.Predictions;
            Base.PermutationMvsLambdas = {0};
            Base.PermutationMvsValid = {0};
            Base.History.TimeHistory.resize(2);
            Base.History.LearnMetricsHistory.resize(2);
            Base.History.LearnMetricsHistory[0]["RMSE"] = 0.8;
            Base.History.LearnMetricsHistory[1]["RMSE"] = 0.6;
            Base.Validate(3, 1, 2);
            Base.Save(File.Name(), "host-only snapshot fixture", nullptr);
            UNIT_ASSERT(TFile(File.Name(), OpenExisting | RdOnly).GetLength() > 0);
        }

        TMetalSnapshot Reader() const {
            TMetalSnapshot result;
            result.Params = Base.Params;
            result.Checksum = Base.Checksum;
            return result;
        }

        template <class TWriter>
        void Append(TWriter&& writer) {
            TFileOutput output(TFile::ForAppend(File.Name()));
            writer(&output);
            output.Finish();
        }

        void AppendLegacy(const TVector<float>& cursor) {
            Append([&](IOutputStream* output) {
                ::SaveMany(output, LegacyCursorTag, cursor);
            });
        }

        void AppendIndexed(i32 iteration, const TVector<float>& cursor) {
            Append([&](IOutputStream* output) {
                ::SaveMany(output, IndexedCursorTag, iteration, cursor);
            });
        }

        void CheckBase(const TMetalSnapshot& restored) const {
            restored.Validate(3, 1, 2);
            UNIT_ASSERT_VALUES_EQUAL(restored.Params, Base.Params);
            UNIT_ASSERT_VALUES_EQUAL(restored.Checksum, Base.Checksum);
            UNIT_ASSERT_VALUES_EQUAL(restored.Depths, Base.Depths);
            UNIT_ASSERT_VALUES_EQUAL(restored.SplitFeatures, Base.SplitFeatures);
            UNIT_ASSERT_VALUES_EQUAL(restored.SplitBins, Base.SplitBins);
            UNIT_ASSERT_VALUES_EQUAL(restored.Leaves, Base.Leaves);
            UNIT_ASSERT_VALUES_EQUAL(restored.Weights, Base.Weights);
            UNIT_ASSERT_VALUES_EQUAL(restored.Predictions, Base.Predictions);
            UNIT_ASSERT(restored.History.LearnMetricsHistory == Base.History.LearnMetricsHistory);
        }
    };

    const TVector<float> BestCursor = {0.125f, -0.25f, 0.375f};
}

Y_UNIT_TEST_SUITE(TMetalBestLearnSnapshotTail) {
    Y_UNIT_TEST(IndexedV2RoundtripPreservesFirstAndLastIterationAndLegacyPrefix) {
        for (i32 iteration : {0, 1}) {
            TSnapshotFile fixture;
            const TString legacyBytes = TFileInput(fixture.File.Name()).ReadAll();
            fixture.Base.BestLearnPredictions = BestCursor;
            fixture.Base.BestLearnIteration = iteration;
            fixture.Base.Save(fixture.File.Name(), "host-only snapshot fixture", nullptr);
            const TString indexedBytes = TFileInput(fixture.File.Name()).ReadAll();
            UNIT_ASSERT(indexedBytes.size() > legacyBytes.size());
            UNIT_ASSERT_VALUES_EQUAL(indexedBytes.substr(0, legacyBytes.size()), legacyBytes);

            auto restored = fixture.Reader();
            UNIT_ASSERT(restored.Load(fixture.File.Name(), nullptr));
            fixture.CheckBase(restored);
            UNIT_ASSERT_VALUES_EQUAL(restored.BestLearnPredictions, BestCursor);
            UNIT_ASSERT_VALUES_EQUAL(restored.BestLearnIteration, iteration);
        }
    }

    Y_UNIT_TEST(LegacyV6WithoutOptionalTailRemainsReadable) {
        TSnapshotFile fixture;
        auto restored = fixture.Reader();
        UNIT_ASSERT(restored.Load(fixture.File.Name(), nullptr));
        fixture.CheckBase(restored);
        UNIT_ASSERT(restored.BestLearnPredictions.empty());
        UNIT_ASSERT_VALUES_EQUAL(restored.BestLearnIteration, -1);
    }

    Y_UNIT_TEST(LegacyV1CursorLoadsWithUnknownIteration) {
        TSnapshotFile fixture;
        fixture.AppendLegacy(BestCursor);
        auto restored = fixture.Reader();
        UNIT_ASSERT(restored.Load(fixture.File.Name(), nullptr));
        fixture.CheckBase(restored);
        UNIT_ASSERT_VALUES_EQUAL(restored.BestLearnPredictions, BestCursor);
        UNIT_ASSERT_VALUES_EQUAL(restored.BestLearnIteration, -1);
    }

    Y_UNIT_TEST(ReusedReaderClearsIndexedStateForLegacyV1AndNoTail) {
        TSnapshotFile indexed;
        indexed.Base.BestLearnPredictions = BestCursor;
        indexed.Base.BestLearnIteration = 1;
        indexed.Base.Save(indexed.File.Name(), "host-only snapshot fixture", nullptr);
        TSnapshotFile legacyCursor;
        const TVector<float> legacyValues = {-0.5f, 0.25f, 0.75f};
        legacyCursor.AppendLegacy(legacyValues);
        TSnapshotFile noTail;

        auto restored = indexed.Reader();
        UNIT_ASSERT(restored.Load(indexed.File.Name(), nullptr));
        UNIT_ASSERT_VALUES_EQUAL(restored.BestLearnIteration, 1);
        UNIT_ASSERT(restored.Load(legacyCursor.File.Name(), nullptr));
        legacyCursor.CheckBase(restored);
        UNIT_ASSERT_VALUES_EQUAL(restored.BestLearnPredictions, legacyValues);
        UNIT_ASSERT_VALUES_EQUAL(restored.BestLearnIteration, -1);

        UNIT_ASSERT(restored.Load(indexed.File.Name(), nullptr));
        UNIT_ASSERT_VALUES_EQUAL(restored.BestLearnPredictions, BestCursor);
        UNIT_ASSERT_VALUES_EQUAL(restored.BestLearnIteration, 1);
        UNIT_ASSERT(restored.Load(noTail.File.Name(), nullptr));
        noTail.CheckBase(restored);
        UNIT_ASSERT(restored.BestLearnPredictions.empty());
        UNIT_ASSERT_VALUES_EQUAL(restored.BestLearnIteration, -1);
    }

    Y_UNIT_TEST(RejectsUnknownOptionalTags) {
        for (ui32 tag : {0u, 0x4D424C33u, std::numeric_limits<ui32>::max()}) {
            TSnapshotFile fixture;
            fixture.Append([&](IOutputStream* output) { ::Save(output, tag); });
            auto restored = fixture.Reader();
            UNIT_ASSERT_EXCEPTION_CONTAINS(restored.Load(fixture.File.Name(), nullptr), TCatBoostException,
                "Unknown Metal best-learn snapshot payload");
        }
    }

    Y_UNIT_TEST(RejectsEveryPartialOptionalTag) {
        for (size_t bytes = 1; bytes < sizeof(IndexedCursorTag); ++bytes) {
            TSnapshotFile fixture;
            fixture.Append([&](IOutputStream* output) { output->Write(&IndexedCursorTag, bytes); });
            auto restored = fixture.Reader();
            UNIT_ASSERT_EXCEPTION_CONTAINS(restored.Load(fixture.File.Name(), nullptr), TCatBoostException,
                "Unknown Metal best-learn snapshot payload");
        }
    }

    Y_UNIT_TEST(RejectsNegativeIndexedIterations) {
        for (i32 iteration : {-1, std::numeric_limits<i32>::min()}) {
            TSnapshotFile fixture;
            fixture.AppendIndexed(iteration, BestCursor);
            auto restored = fixture.Reader();
            UNIT_ASSERT_EXCEPTION_CONTAINS(restored.Load(fixture.File.Name(), nullptr), TCatBoostException,
                "Saved Metal best-learn cursor has an invalid iteration");
        }
    }

    Y_UNIT_TEST(RejectsIndexedIterationsAtOrBeyondCompletedTrees) {
        for (i32 iteration : {2, 3, std::numeric_limits<i32>::max()}) {
            TSnapshotFile fixture;
            fixture.AppendIndexed(iteration, BestCursor);
            auto restored = fixture.Reader();
            UNIT_ASSERT_EXCEPTION_CONTAINS(restored.Load(fixture.File.Name(), nullptr), TCatBoostException,
                "Saved Metal best-learn cursor has an invalid iteration");
        }
    }

    Y_UNIT_TEST(RejectsEmptyShortAndLongCursorsInBothTailVersions) {
        for (bool indexed : {false, true}) {
            for (size_t size : {0u, 2u, 4u}) {
                TSnapshotFile fixture;
                const TVector<float> cursor(size, 0.5f);
                if (indexed) fixture.AppendIndexed(0, cursor);
                else fixture.AppendLegacy(cursor);
                auto restored = fixture.Reader();
                UNIT_ASSERT_EXCEPTION_CONTAINS(restored.Load(fixture.File.Name(), nullptr), TCatBoostException,
                    "Saved Metal best-learn cursor has inconsistent dimensions");
            }
        }
    }

    Y_UNIT_TEST(RejectsNanAndBothInfinitiesInBothTailVersions) {
        for (bool indexed : {false, true}) {
            for (float value : {std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity()}) {
                TSnapshotFile fixture;
                auto cursor = BestCursor;
                cursor[1] = value;
                if (indexed) fixture.AppendIndexed(0, cursor);
                else fixture.AppendLegacy(cursor);
                auto restored = fixture.Reader();
                UNIT_ASSERT_EXCEPTION_CONTAINS(restored.Load(fixture.File.Name(), nullptr), TCatBoostException,
                    "Saved Metal best-learn cursor is nonfinite");
            }
        }
    }

    Y_UNIT_TEST(RejectsTrailingBytesAfterIndexedCursor) {
        TSnapshotFile fixture;
        fixture.AppendIndexed(1, BestCursor);
        fixture.Append([](IOutputStream* output) { output->Write("x", 1); });
        auto restored = fixture.Reader();
        UNIT_ASSERT_EXCEPTION_CONTAINS(restored.Load(fixture.File.Name(), nullptr), TCatBoostException,
            "Saved Metal best-learn cursor has trailing data");
    }

    Y_UNIT_TEST(RejectsTrailingBytesAfterLegacyCursor) {
        TSnapshotFile fixture;
        fixture.AppendLegacy(BestCursor);
        fixture.Append([](IOutputStream* output) { output->Write("x", 1); });
        auto restored = fixture.Reader();
        UNIT_ASSERT_EXCEPTION_CONTAINS(restored.Load(fixture.File.Name(), nullptr), TCatBoostException,
            "Saved Metal best-learn cursor has trailing data");
    }

    Y_UNIT_TEST(RejectsTruncatedIndexedIteration) {
        for (size_t bytes = 0; bytes < sizeof(i32); ++bytes) {
            TSnapshotFile fixture;
            fixture.Append([&](IOutputStream* output) {
                ::Save(output, IndexedCursorTag);
                const i32 iteration = 0;
                output->Write(&iteration, bytes);
            });
            auto restored = fixture.Reader();
            UNIT_ASSERT_EXCEPTION(restored.Load(fixture.File.Name(), nullptr), TLoadEOF);
        }
    }

    Y_UNIT_TEST(RejectsTruncatedCursorBodyInBothTailVersions) {
        for (bool indexed : {false, true}) {
            TSnapshotFile fixture;
            if (indexed) fixture.AppendIndexed(0, BestCursor);
            else fixture.AppendLegacy(BestCursor);
            {
                TFile file(fixture.File.Name(), OpenExisting | WrOnly);
                file.Resize(file.GetLength() - 1);
            }
            auto restored = fixture.Reader();
            UNIT_ASSERT_EXCEPTION(restored.Load(fixture.File.Name(), nullptr), TLoadEOF);
        }
    }
}
