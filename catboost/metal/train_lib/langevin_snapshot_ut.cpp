#include "snapshot.h"

#include <library/cpp/testing/unittest/registar.h>
#include <util/stream/file.h>
#include <util/string/cast.h>
#include <util/system/tempfile.h>

using namespace NCB;

namespace {
    constexpr ui32 LangevinTag = 0x4D4C4731;
    constexpr ui32 BestTag = 0x4D424C32;
    struct TFixture {
        TTempFile File{MakeTempName()};
        TMetalSnapshot Base;
        TFixture() {
            NJson::TJsonValue options;
            NCatboostOptions::TCatBoostOptions(ETaskType::GPU).Save(&options);
            Base.Params = ToString(options); Base.Checksum = 919;
            Base.Depths = {0, 0}; Base.Leaves = {.1f, .2f}; Base.Weights = {3, 3};
            Base.Predictions = {.1f, .2f, .3f};
            Base.PermutationPredictions = Base.Predictions;
            Base.PermutationMvsLambdas = {0}; Base.PermutationMvsValid = {0};
            Base.History.TimeHistory.resize(2);
            Base.Save(File.Name(), "host-only Langevin snapshot", nullptr);
        }
        TMetalSnapshot Reader(bool enabled = true) const {
            TMetalSnapshot result;
            result.Params = Base.Params; result.Checksum = Base.Checksum; result.Langevin = enabled;
            return result;
        }
        template<class TWriter> void Append(TWriter&& writer) {
            TFileOutput output(TFile::ForAppend(File.Name())); writer(&output); output.Finish();
        }
        void AppendRandom(ui64 draws = 65601, ui32 completed = 2, bool cache = true) {
            Append([&](IOutputStream* out) { ::SaveMany(out, LangevinTag, draws, completed, cache); });
        }
    };
}

Y_UNIT_TEST_SUITE(TMetalLangevinSnapshotTail) {
    Y_UNIT_TEST(RoundtripWithAndWithoutBestKeepsLegacyBytesAndExactDraws) {
        for (bool best : {false, true}) for (bool cache : {false, true}) {
            TFixture f;
            const auto legacy = TFileInput(f.File.Name()).ReadAll();
            f.Base.Langevin = true; f.Base.LangevinRandom = {cache ? 65601u : 43u, 2, cache};
            if (best) { f.Base.BestLearnIteration = 0; f.Base.BestLearnPredictions = {.5f, .25f, -.25f}; }
            f.Base.Save(f.File.Name(), "host-only Langevin snapshot", nullptr);
            const auto saved = TFileInput(f.File.Name()).ReadAll();
            UNIT_ASSERT_VALUES_EQUAL(saved.substr(0, legacy.size()), legacy);
            auto result = f.Reader(); UNIT_ASSERT(result.Load(f.File.Name(), nullptr));
            result.Validate(3, 0, 2);
            UNIT_ASSERT_VALUES_EQUAL(result.LangevinRandom.DrawCount, f.Base.LangevinRandom.DrawCount);
            UNIT_ASSERT_VALUES_EQUAL(result.LangevinRandom.CompletedIterations, 2);
            UNIT_ASSERT_VALUES_EQUAL(result.LangevinRandom.WeakSeedCacheInitialized, cache);
            UNIT_ASSERT_VALUES_EQUAL(result.BestLearnPredictions, f.Base.BestLearnPredictions);
        }
    }
    Y_UNIT_TEST(MissingRandomTailCannotSilentlyRestartStream) {
        TFixture f; auto result = f.Reader();
        UNIT_ASSERT_EXCEPTION_CONTAINS(result.Load(f.File.Name(), nullptr), TCatBoostException, "payload is missing");
    }
    Y_UNIT_TEST(LegacySnapshotRemainsValidWhenLangevinIsDisabled) {
        TFixture f; auto result = f.Reader(false);
        UNIT_ASSERT(result.Load(f.File.Name(), nullptr)); result.Validate(3, 0, 2);
        UNIT_ASSERT_VALUES_EQUAL(result.LangevinRandom.DrawCount, 0);
    }
    Y_UNIT_TEST(DisabledConfigurationRejectsRandomTail) {
        TFixture f; f.AppendRandom(); auto result = f.Reader(false);
        UNIT_ASSERT_EXCEPTION_CONTAINS(result.Load(f.File.Name(), nullptr), TCatBoostException, "Unexpected or duplicate");
    }
    Y_UNIT_TEST(RejectsDuplicateTailAndBestAfterRandom) {
        for (bool best : {false, true}) {
            TFixture f; f.AppendRandom();
            if (best) f.Append([&](IOutputStream* out) { ::SaveMany(out, BestTag, i32(0), f.Base.Predictions); });
            else f.AppendRandom();
            auto result = f.Reader(); UNIT_ASSERT_EXCEPTION(result.Load(f.File.Name(), nullptr), TCatBoostException);
        }
    }
    Y_UNIT_TEST(RejectsCompletedIterationMismatch) {
        for (ui32 completed : {0u, 1u, 3u, Max<ui32>()}) {
            TFixture f; f.AppendRandom(65601, completed); auto result = f.Reader();
            UNIT_ASSERT_EXCEPTION_CONTAINS(result.Load(f.File.Name(), nullptr), TCatBoostException, "iteration count differs");
        }
    }
    Y_UNIT_TEST(RejectsEveryTruncatedRandomBody) {
        for (ui64 removed = 1; removed <= sizeof(ui64) + sizeof(ui32) + sizeof(bool); ++removed) {
            TFixture f; f.AppendRandom();
            { TFile file(f.File.Name(), OpenExisting | WrOnly); file.Resize(file.GetLength() - removed); }
            auto result = f.Reader(); UNIT_ASSERT_EXCEPTION(result.Load(f.File.Name(), nullptr), TLoadEOF);
        }
    }
    Y_UNIT_TEST(RejectsUnknownTrailingBytes) {
        TFixture f; f.AppendRandom();
        f.Append([](IOutputStream* out) { out->Write("bad!", 4); });
        auto result = f.Reader(); UNIT_ASSERT_EXCEPTION(result.Load(f.File.Name(), nullptr), TCatBoostException);
    }
    Y_UNIT_TEST(ReusedReaderClearsRandomStateForLegacyAndRestoresAgain) {
        TFixture f; f.AppendRandom(); auto result = f.Reader();
        UNIT_ASSERT(result.Load(f.File.Name(), nullptr));
        TFixture legacy; result.Langevin = false;
        UNIT_ASSERT(result.Load(legacy.File.Name(), nullptr));
        UNIT_ASSERT_VALUES_EQUAL(result.LangevinRandom.DrawCount, 0);
        result.Langevin = true; UNIT_ASSERT(result.Load(f.File.Name(), nullptr));
        UNIT_ASSERT_VALUES_EQUAL(result.LangevinRandom.DrawCount, 65601);
    }
    Y_UNIT_TEST(RejectsChangedOptionsBeforeReadingAnotherTrainersPayload) {
        TFixture f; f.AppendRandom();
        auto result = f.Reader(false);
        result.YetiRank = true; // This path expects a legacy TString tag.
        NCatboostOptions::TCatBoostOptions changed(ETaskType::GPU);
        changed.RandomSeed = 13;
        NJson::TJsonValue options; changed.Save(&options);
        result.Params = ToString(options);
        UNIT_ASSERT_EXCEPTION_CONTAINS(result.Load(f.File.Name(), nullptr), TCatBoostException,
            "parameters differ");
    }
    Y_UNIT_TEST(FeatureParallelUsesDedicatedStateInsteadOfLegacyChooserFields) {
        for (bool ordered : {false, true}) {
            TFixture f; f.Base.Langevin = true; f.Base.LangevinRandom = {65601, 2, true};
            if (ordered) {
                f.Base.PermutationPredictions.clear(); f.Base.PermutationMvsLambdas.clear(); f.Base.PermutationMvsValid.clear();
                f.Base.OrderedDescriptors = {1, 2, 0, 0, 3, 3, 2, 0};
                f.Base.OrderedCursors = {0, 0, .1f, .2f, .3f};
            }
            f.Base.Validate(3, 0, 2, 1, 1, 0, ordered, true);
            f.Base.OrderedRandomDrawCount = 1;
            UNIT_ASSERT_EXCEPTION(f.Base.Validate(3, 0, 2, 1, 1, 0, ordered, true), TCatBoostException);
        }
    }
}
