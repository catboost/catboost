#include "snapshot.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/file.h>
#include <util/stream/str.h>
#include <util/string/cast.h>
#include <util/system/tempfile.h>

#include <limits>

using namespace NCB;

namespace {
    constexpr ui32 ModelBasedTag = 0x4D4D4231;
    constexpr ui32 LegacyBestTag = 0x4D424C31;
    constexpr ui32 IndexedBestTag = 0x4D424C32;
    constexpr ui32 LangevinTag = 0x4D4C4731;
    constexpr ui32 Permutations = 3;
    constexpr ui32 LeafCapacity = 4;
    constexpr ui64 MaximumValues = (ui64(512) << 20) / sizeof(float);

    TVector<float> HistoryLeaves(ui32 dimension = 1) {
        TVector<float> values(2 * Permutations * LeafCapacity * dimension);
        for (size_t i = 0; i < values.size(); ++i) {
            // Unique, exactly representable values expose reordered histories,
            // and negative padding must be preserved just like active leaves.
            values[i] = float(i + 1) / (i % 2 ? -16.f : 16.f);
        }
        return values;
    }

    void WriteModelTail(IOutputStream* out, ui32 permutations, ui32 capacity,
                        ui64 declaredValues, const TVector<float>& values = {}, ui32 dimension = 1) {
        ::SaveMany(out, ModelBasedTag, permutations, capacity, dimension, declaredValues);
        // Write scalars independently of the production SaveArray path. The
        // explicit ui64 count is the only array length on the wire.
        for (float value : values) ::Save(out, value);
    }

    TVector<float> ExpandValues(const TVector<float>& scalar, ui32 dimension) {
        TVector<float> result;
        result.reserve(scalar.size() * dimension);
        for (float value : scalar) for (ui32 channel = 0; channel < dimension; ++channel) {
            result.push_back(value + float(channel) / 128.f);
        }
        return result;
    }

    struct TFixture {
        TTempFile File{MakeTempName()};
        TMetalSnapshot Base;
        const ui32 ApproxDimension;

        explicit TFixture(bool greedy = false, ui32 dimension = 1)
            : ApproxDimension(dimension)
        {
            NJson::TJsonValue options;
            NCatboostOptions::TCatBoostOptions(ETaskType::GPU).Save(&options);
            Base.Params = ToString(options);
            Base.Checksum = 0x2468ACE0;
            Base.Greedy = greedy;
            Base.Depths = {1, 0};
            Base.Predictions = {.25f, -.5f, .75f};
            Base.PermutationPredictions = {-.125f, .25f, -.375f, .5f, -.625f, .75f, .25f, -.5f, .75f};
            Base.Predictions = ExpandValues(Base.Predictions, dimension);
            Base.PermutationPredictions = ExpandValues(Base.PermutationPredictions, dimension);
            if (dimension > 1) Base.OptimizationPredictions.resize(3 * Permutations * dimension, .125f);
            Base.PermutationMvsLambdas = {0, 0, 0};
            Base.PermutationMvsValid = {0, 0, 0};
            Base.History.TimeHistory.resize(2);
            Base.History.LearnMetricsHistory.resize(2);
            Base.History.LearnMetricsHistory[0]["RMSE"] = .75;
            Base.History.LearnMetricsHistory[1]["RMSE"] = .5;
            if (greedy) {
                TMetalGreedyTree first;
                first.Nodes = {{1, 2, 0, 1, 2, Max<ui32>()},
                               {0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0}};
                first.Values = ExpandValues({-.125f, .25f}, dimension);
                first.Weights = {1, 2};
                Base.GreedyTrees.Append(first, 2, 2, dimension);
                TMetalGreedyTree second;
                second.Nodes = {{0, 0, 0, 0, 0, 0}};
                second.Values = ExpandValues({.375f}, dimension);
                second.Weights = {3};
                Base.GreedyTrees.Append(second, 2, 2, dimension);
            } else {
                Base.SplitFeatures = {1, 0, 0, 0};
                Base.SplitBins = {2, 0, 0, 0};
                Base.SplitTypes = {0, 0, 0, 0};
                Base.Leaves = {-.125f, .25f, 0, 0, .375f, 0, 0, 0};
                Base.Leaves = ExpandValues(Base.Leaves, dimension);
                Base.Weights = {1, 2, 0, 0, 3, 0, 0, 0};
            }
            CheckBase(Base);
            Save();
        }

        void Save() const {
            Base.Save(File.Name(), "host-only model-based snapshot", nullptr);
        }

        TMetalSnapshot Reader() const {
            TMetalSnapshot result;
            result.Params = Base.Params;
            result.Checksum = Base.Checksum;
            result.Greedy = Base.Greedy;
            result.Langevin = Base.Langevin;
            return result;
        }

        template <class TWriter>
        void Append(TWriter&& writer) {
            TFileOutput output(TFile::ForAppend(File.Name()));
            writer(&output);
            output.Finish();
        }

        void AppendModel() {
            const auto values = HistoryLeaves();
            Append([&](IOutputStream* out) {
                WriteModelTail(out, Permutations, LeafCapacity, values.size(), values);
            });
        }

        void SetModelHistory() {
            Base.ModelBasedPermutationCount = Permutations;
            Base.ModelBasedLeafCapacity = LeafCapacity;
            Base.ModelBasedApproxDimension = ApproxDimension;
            Base.ModelBasedPermutationLeaves = HistoryLeaves(ApproxDimension);
        }

        void CheckBase(const TMetalSnapshot& restored) const {
            const ui32 optimizerDimension = ApproxDimension == 1 ? 0 : ApproxDimension;
            if (Base.Greedy) restored.ValidateGreedy(3, 2, 1, 2, LeafCapacity, 2, Permutations,
                ApproxDimension, optimizerDimension);
            else restored.Validate(3, 2, 2, ApproxDimension, Permutations, optimizerDimension);
            UNIT_ASSERT_VALUES_EQUAL(restored.Depths, Base.Depths);
            UNIT_ASSERT_VALUES_EQUAL(restored.Leaves, Base.Leaves);
            UNIT_ASSERT_VALUES_EQUAL(restored.Weights, Base.Weights);
            UNIT_ASSERT_VALUES_EQUAL(restored.Predictions, Base.Predictions);
            UNIT_ASSERT_VALUES_EQUAL(restored.PermutationPredictions, Base.PermutationPredictions);
            UNIT_ASSERT_VALUES_EQUAL(restored.GreedyTrees.Nodes, Base.GreedyTrees.Nodes);
            UNIT_ASSERT_VALUES_EQUAL(restored.GreedyTrees.Values, Base.GreedyTrees.Values);
            UNIT_ASSERT(restored.History.LearnMetricsHistory == Base.History.LearnMetricsHistory);
        }

        void CheckModelHistory(const TMetalSnapshot& restored) const {
            restored.ModelBasedValidate(Permutations, LeafCapacity, ApproxDimension);
            UNIT_ASSERT_VALUES_EQUAL(restored.ModelBasedPermutationCount, Permutations);
            UNIT_ASSERT_VALUES_EQUAL(restored.ModelBasedLeafCapacity, LeafCapacity);
            UNIT_ASSERT_VALUES_EQUAL(restored.ModelBasedApproxDimension, ApproxDimension);
            UNIT_ASSERT_VALUES_EQUAL(restored.ModelBasedPermutationLeaves, HistoryLeaves(ApproxDimension));
        }

        void WriteLegacy(const TString& path) const {
            // Independent v6 writer with no new fields, including the existing
            // greedy extension. Absence must preserve these exact bytes.
            TProgressHelper("CatBoost Metal snapshot v6").Write(path, [&](IOutputStream* out) {
                ::SaveMany(out, Base.Params, Base.Checksum, Base.Bias, Base.MvsLambda, Base.MvsLambdaIsSet,
                    Base.Depths, Base.SplitFeatures, Base.SplitBins, Base.SplitTypes, Base.Leaves, Base.Weights,
                    Base.Predictions, Base.History, Base.PermutationPredictions, Base.PermutationMvsLambdas,
                    Base.PermutationMvsValid, Base.UsedFeatures, Base.OptimizationPredictions,
                    Base.OrderedDescriptors, Base.OrderedCursors, Base.OrderedRandomDrawCount,
                    Base.OrderedRandomCompletedIterations, Base.OrderedBootstrapInitialized);
                if (Base.Greedy) ::SaveMany(out, TString("Metal greedy trees v1"), Base.GreedyTrees);
            });
        }
    };
}

Y_UNIT_TEST_SUITE(TMetalModelBasedSnapshotTail) {
    Y_UNIT_TEST(AbsentHistoryPreservesExactLegacyBytesAndLoadsWithoutInventedLeaves) {
        for (bool greedy : {false, true}) {
            TFixture f(greedy);
            TTempFile legacy{MakeTempName()};
            f.WriteLegacy(legacy.Name());
            UNIT_ASSERT_VALUES_EQUAL(TFileInput(f.File.Name()).ReadAll(), TFileInput(legacy.Name()).ReadAll());
            auto restored = f.Reader();
            UNIT_ASSERT(restored.Load(legacy.Name(), nullptr));
            f.CheckBase(restored);
            restored.ModelBasedValidate(Permutations, LeafCapacity);
            UNIT_ASSERT_VALUES_EQUAL(restored.ModelBasedPermutationCount, 0);
            UNIT_ASSERT_VALUES_EQUAL(restored.ModelBasedLeafCapacity, 0);
            UNIT_ASSERT_VALUES_EQUAL(restored.ModelBasedApproxDimension, 0);
            UNIT_ASSERT(restored.ModelBasedPermutationLeaves.empty());
        }
    }

    Y_UNIT_TEST(RoundtripSymmetricAndGreedyAfterOptionalBestAndLangevinRecords) {
        for (bool greedy : {false, true}) for (bool best : {false, true}) for (bool langevin : {false, true}) {
            TFixture f(greedy);
            if (best) {
                f.Base.BestLearnIteration = 1;
                f.Base.BestLearnPredictions = {-.5f, .25f, .125f};
            }
            f.Base.Langevin = langevin;
            if (langevin) f.Base.LangevinRandom = {65601, 2, true};
            f.Save();
            const auto prefix = TFileInput(f.File.Name()).ReadAll();
            f.SetModelHistory();
            f.Save();
            const auto saved = TFileInput(f.File.Name()).ReadAll();
            TStringStream tail;
            WriteModelTail(&tail, Permutations, LeafCapacity, HistoryLeaves().size(), HistoryLeaves());
            UNIT_ASSERT_VALUES_EQUAL(saved.substr(0, prefix.size()), prefix);
            UNIT_ASSERT_VALUES_EQUAL(saved.substr(prefix.size()), tail.Str());

            auto restored = f.Reader();
            UNIT_ASSERT(restored.Load(f.File.Name(), nullptr));
            f.CheckBase(restored);
            f.CheckModelHistory(restored);
            UNIT_ASSERT_VALUES_EQUAL(restored.BestLearnIteration, best ? 1 : -1);
            UNIT_ASSERT_VALUES_EQUAL(restored.BestLearnPredictions, f.Base.BestLearnPredictions);
            UNIT_ASSERT_VALUES_EQUAL(restored.LangevinRandom.DrawCount, langevin ? 65601 : 0);
            UNIT_ASSERT_VALUES_EQUAL(restored.LangevinRandom.CompletedIterations, langevin ? 2 : 0);
            UNIT_ASSERT_VALUES_EQUAL(restored.LangevinRandom.WeakSeedCacheInitialized, langevin);
        }
    }

    Y_UNIT_TEST(LegacyBestCursorCanPrecedeModelHistoryWithOrWithoutLangevin) {
        for (bool greedy : {false, true}) for (bool langevin : {false, true}) {
            TFixture f(greedy);
            f.Append([&](IOutputStream* out) {
                ::SaveMany(out, LegacyBestTag, f.Base.Predictions);
                if (langevin) ::SaveMany(out, LangevinTag, ui64(65601), ui32(2), true);
            });
            f.AppendModel();
            auto restored = f.Reader();
            restored.Langevin = langevin;
            UNIT_ASSERT(restored.Load(f.File.Name(), nullptr));
            f.CheckBase(restored);
            f.CheckModelHistory(restored);
            UNIT_ASSERT_VALUES_EQUAL(restored.BestLearnIteration, -1);
            UNIT_ASSERT_VALUES_EQUAL(restored.BestLearnPredictions, f.Base.Predictions);
            UNIT_ASSERT_VALUES_EQUAL(restored.LangevinRandom.DrawCount, langevin ? 65601 : 0);
        }
    }

    Y_UNIT_TEST(VectorHistoryPreservesEveryChannelAndRequiresTheCurrentDimension) {
        for (bool greedy : {false, true}) for (ui32 dimension : {2u, 3u, 64u}) {
            TFixture f(greedy, dimension);
            const auto prefix = TFileInput(f.File.Name()).ReadAll();
            f.SetModelHistory();
            f.Save();
            TStringStream tail;
            WriteModelTail(&tail, Permutations, LeafCapacity, f.Base.ModelBasedPermutationLeaves.size(),
                f.Base.ModelBasedPermutationLeaves, dimension);
            const auto saved = TFileInput(f.File.Name()).ReadAll();
            UNIT_ASSERT_VALUES_EQUAL(saved.substr(0, prefix.size()), prefix);
            UNIT_ASSERT_VALUES_EQUAL(saved.substr(prefix.size()), tail.Str());
            auto restored = f.Reader();
            UNIT_ASSERT(restored.Load(f.File.Name(), nullptr));
            f.CheckBase(restored);
            f.CheckModelHistory(restored);
            for (ui32 expected : {0u, 1u, dimension + 1}) {
                UNIT_ASSERT_EXCEPTION_CONTAINS(restored.ModelBasedValidate(Permutations, LeafCapacity, expected),
                    TCatBoostException, "differs from current");
            }
            restored.ModelBasedPermutationLeaves.back() = std::numeric_limits<float>::infinity();
            UNIT_ASSERT_EXCEPTION_CONTAINS(restored.ModelBasedValidate(Permutations, LeafCapacity, dimension),
                TCatBoostException, "nonfinite leaf values");
        }
    }

    Y_UNIT_TEST(ReusedReaderClearsAbsentHistoryAndCanRestoreItAgain) {
        for (bool greedy : {false, true}) {
            TFixture present(greedy), absent(greedy);
            present.SetModelHistory();
            present.Save();
            auto restored = present.Reader();
            UNIT_ASSERT(restored.Load(present.File.Name(), nullptr));
            present.CheckModelHistory(restored);
            UNIT_ASSERT(restored.Load(absent.File.Name(), nullptr));
            absent.CheckBase(restored);
            restored.ModelBasedValidate(Permutations, LeafCapacity);
            UNIT_ASSERT_VALUES_EQUAL(restored.ModelBasedPermutationCount, 0);
            UNIT_ASSERT_VALUES_EQUAL(restored.ModelBasedLeafCapacity, 0);
            UNIT_ASSERT_VALUES_EQUAL(restored.ModelBasedApproxDimension, 0);
            UNIT_ASSERT(restored.ModelBasedPermutationLeaves.empty());
            UNIT_ASSERT(restored.Load(present.File.Name(), nullptr));
            present.CheckModelHistory(restored);
        }
    }

    Y_UNIT_TEST(ZeroTreeHistoryIsPresentAtInclusiveDimensionBounds) {
        for (ui32 permutations : {1u, 64u}) for (ui32 capacity : {1u, 65536u}) for (ui32 dimension : {1u, 2u, 3u, 64u}) {
            TFixture f(false, dimension);
            f.Base.Depths.clear();
            f.Base.SplitFeatures.clear();
            f.Base.SplitBins.clear();
            f.Base.SplitTypes.clear();
            f.Base.Leaves.clear();
            f.Base.Weights.clear();
            f.Base.History.TimeHistory.clear();
            f.Base.History.LearnMetricsHistory.clear();
            f.Base.PermutationPredictions.clear();
            for (ui32 p = 0; p < permutations; ++p) {
                f.Base.PermutationPredictions.insert(f.Base.PermutationPredictions.end(),
                    f.Base.Predictions.begin(), f.Base.Predictions.end());
            }
            f.Base.PermutationMvsLambdas.assign(permutations, 0);
            f.Base.PermutationMvsValid.assign(permutations, 0);
            const ui32 optimizerDimension = dimension == 1 ? 0 : dimension;
            f.Base.OptimizationPredictions.assign(3 * permutations * optimizerDimension, .125f);
            f.Save();
            const auto prefix = TFileInput(f.File.Name()).ReadAll();
            f.Base.ModelBasedPermutationCount = permutations;
            f.Base.ModelBasedLeafCapacity = capacity;
            f.Base.ModelBasedApproxDimension = dimension;
            f.Save();
            UNIT_ASSERT_VALUES_EQUAL(TFileInput(f.File.Name()).ReadAll().size(),
                prefix.size() + 4 * sizeof(ui32) + sizeof(ui64));
            auto restored = f.Reader();
            UNIT_ASSERT(restored.Load(f.File.Name(), nullptr));
            restored.Validate(3, 2, 2, dimension, permutations, optimizerDimension);
            restored.ModelBasedValidate(permutations, capacity, dimension);
            UNIT_ASSERT_VALUES_EQUAL(restored.ModelBasedPermutationCount, permutations);
            UNIT_ASSERT_VALUES_EQUAL(restored.ModelBasedLeafCapacity, capacity);
            UNIT_ASSERT_VALUES_EQUAL(restored.ModelBasedApproxDimension, dimension);
            UNIT_ASSERT(restored.ModelBasedPermutationLeaves.empty());
        }
    }

    Y_UNIT_TEST(RejectsCurrentPermutationOrCapacityMismatch) {
        TFixture f;
        f.AppendModel();
        auto restored = f.Reader();
        UNIT_ASSERT(restored.Load(f.File.Name(), nullptr));
        f.CheckModelHistory(restored);
        for (ui32 permutations : {0u, 1u, 2u, 4u, 64u}) {
            UNIT_ASSERT_EXCEPTION(restored.ModelBasedValidate(permutations, LeafCapacity), TCatBoostException);
        }
        for (ui32 capacity : {0u, 1u, 3u, 5u, 65536u}) {
            UNIT_ASSERT_EXCEPTION(restored.ModelBasedValidate(Permutations, capacity), TCatBoostException);
        }
    }

    Y_UNIT_TEST(RejectsInvalidDimensionsBeforeAllocatingDeclaredHistory) {
        struct TDimensions {
            ui32 Permutations;
            ui32 Capacity;
            ui32 Dimension;
        };
        for (const auto& dimensions : TVector<TDimensions>{
            {0, 0, 0}, {0, LeafCapacity, 1}, {65, LeafCapacity, 1}, {Max<ui32>(), LeafCapacity, 1},
            {Permutations, 0, 1}, {Permutations, 65537, 1}, {Permutations, Max<ui32>(), 1},
            {Permutations, LeafCapacity, 0}, {Permutations, LeafCapacity, 65},
            {Permutations, LeafCapacity, Max<ui32>()}})
        {
            TFixture f;
            f.Append([&](IOutputStream* out) {
                const ui64 size = !dimensions.Permutations && !dimensions.Capacity && !dimensions.Dimension ? 0 : Max<ui64>();
                WriteModelTail(out, dimensions.Permutations, dimensions.Capacity, size, {}, dimensions.Dimension);
            });
            auto restored = f.Reader();
            UNIT_ASSERT_EXCEPTION_CONTAINS(restored.Load(f.File.Name(), nullptr), TCatBoostException,
                "invalid permutation count");
            UNIT_ASSERT(restored.ModelBasedPermutationLeaves.empty());
        }
    }

    Y_UNIT_TEST(RejectsWrongOrHostileDeclaredLengthsBeforeAllocatingHistory) {
        for (ui64 size : {ui64(0), ui64(23), ui64(25), MaximumValues, MaximumValues + 1, Max<ui64>()}) {
            TFixture f;
            f.Append([&](IOutputStream* out) { WriteModelTail(out, Permutations, LeafCapacity, size); });
            auto restored = f.Reader();
            UNIT_ASSERT_EXCEPTION_CONTAINS(restored.Load(f.File.Name(), nullptr), TCatBoostException,
                "inconsistent array size");
            UNIT_ASSERT(restored.ModelBasedPermutationLeaves.empty());
        }
    }

    Y_UNIT_TEST(ChecksAggregateByteLimitBeforeAllocationAndIncludesExactBoundary) {
        TFixture f;
        f.Base.Depths.assign(32, 0);
        UNIT_ASSERT_VALUES_EQUAL(f.Base.ModelBasedValueCount(64, 65536), MaximumValues);
        f.Base.Depths.push_back(0);
        UNIT_ASSERT_EXCEPTION_CONTAINS(f.Base.ModelBasedValueCount(64, 65536), TCatBoostException, "exceeds 512 MiB");
        f.Base.SplitFeatures.clear();
        f.Base.SplitBins.clear();
        f.Base.SplitTypes.clear();
        f.Base.Leaves.assign(33, .125f);
        f.Base.Weights.assign(33, 3);
        f.Base.History.TimeHistory.resize(33);
        f.Base.History.LearnMetricsHistory.resize(33);
        f.Base.Validate(3, 0, 33, 1, Permutations);
        f.Save();
        f.Append([](IOutputStream* out) { WriteModelTail(out, 64, 65536, ui64(33) * 64 * 65536); });
        auto restored = f.Reader();
        UNIT_ASSERT_EXCEPTION_CONTAINS(restored.Load(f.File.Name(), nullptr), TCatBoostException, "exceeds 512 MiB");
        UNIT_ASSERT(restored.ModelBasedPermutationLeaves.empty());
    }

    Y_UNIT_TEST(VectorChannelsAreIncludedInTheAggregateByteLimitBeforeAllocation) {
        TFixture f;
        // Two scalar trees at these bounds occupy 32 MiB; their 64 channels
        // would occupy 2 GiB even though each individual dimension is valid.
        UNIT_ASSERT_VALUES_EQUAL(f.Base.ModelBasedValueCount(64, 65536, 16), MaximumValues);
        UNIT_ASSERT_EXCEPTION_CONTAINS(f.Base.ModelBasedValueCount(64, 65536, 64), TCatBoostException,
            "exceeds 512 MiB");
        f.Append([](IOutputStream* out) { WriteModelTail(out, 64, 65536, ui64(2) * 64 * 65536 * 64, {}, 64); });
        auto restored = f.Reader();
        UNIT_ASSERT_EXCEPTION_CONTAINS(restored.Load(f.File.Name(), nullptr), TCatBoostException, "exceeds 512 MiB");
        UNIT_ASSERT(restored.ModelBasedPermutationLeaves.empty());
    }

    Y_UNIT_TEST(RejectsNonfiniteActiveLeavesAndPaddingInDifferentHistories) {
        for (size_t index : {0u, 3u, 23u}) for (float value : {
            std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity()})
        {
            TFixture f;
            f.SetModelHistory();
            f.Base.ModelBasedPermutationLeaves[index] = value;
            UNIT_ASSERT_EXCEPTION(f.Base.ModelBasedValidate(Permutations, LeafCapacity), TCatBoostException);
            UNIT_ASSERT_EXCEPTION(f.Save(), TCatBoostException);
            f.Append([&](IOutputStream* out) {
                WriteModelTail(out, Permutations, LeafCapacity, f.Base.ModelBasedPermutationLeaves.size(),
                    f.Base.ModelBasedPermutationLeaves);
            });
            auto restored = f.Reader();
            UNIT_ASSERT_EXCEPTION_CONTAINS(restored.Load(f.File.Name(), nullptr), TCatBoostException, "nonfinite leaf values");
        }
    }

    Y_UNIT_TEST(RejectsInconsistentAbsentAndPresentStateBeforeSaving) {
        for (ui32 variant : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u}) {
            TFixture f;
            const auto legacy = TFileInput(f.File.Name()).ReadAll();
            if (variant == 0) f.Base.ModelBasedLeafCapacity = LeafCapacity;
            else if (variant == 1) f.Base.ModelBasedPermutationLeaves = HistoryLeaves();
            else if (variant == 5) f.Base.ModelBasedApproxDimension = 1;
            else {
                f.SetModelHistory();
                if (variant == 2) f.Base.ModelBasedPermutationLeaves.clear();
                if (variant == 3) f.Base.ModelBasedPermutationLeaves.pop_back();
                if (variant == 4) f.Base.ModelBasedPermutationLeaves.push_back(.125f);
                if (variant == 6) f.Base.ModelBasedApproxDimension = 0;
                if (variant == 7) f.Base.ModelBasedApproxDimension = 65;
            }
            UNIT_ASSERT_EXCEPTION(f.Base.ModelBasedValidate(Permutations, LeafCapacity), TCatBoostException);
            UNIT_ASSERT_EXCEPTION(f.Save(), TCatBoostException);
            UNIT_ASSERT_VALUES_EQUAL(TFileInput(f.File.Name()).ReadAll(), legacy);
        }
    }

    Y_UNIT_TEST(RejectsDuplicateHistoryAndBothBestVersionsAfterHistory) {
        for (ui32 tag : {ModelBasedTag, LegacyBestTag, IndexedBestTag}) {
            TFixture f;
            f.AppendModel();
            f.Append([&](IOutputStream* out) {
                if (tag == ModelBasedTag) WriteModelTail(out, Permutations, LeafCapacity, HistoryLeaves().size(), HistoryLeaves());
                else if (tag == LegacyBestTag) ::SaveMany(out, tag, f.Base.Predictions);
                else ::SaveMany(out, tag, i32(1), f.Base.Predictions);
            });
            auto restored = f.Reader();
            UNIT_ASSERT_EXCEPTION(restored.Load(f.File.Name(), nullptr), TCatBoostException);
        }
    }

    Y_UNIT_TEST(HistoryCannotPrecedeRequiredLangevinOrPermitAnotherRandomRecord) {
        for (bool earlierRandom : {false, true}) {
            TFixture f;
            f.Base.Langevin = true;
            f.Base.LangevinRandom = {65601, 2, true};
            if (earlierRandom) f.Save();
            f.AppendModel();
            f.Append([](IOutputStream* out) { ::SaveMany(out, LangevinTag, ui64(65602), ui32(2), true); });
            auto restored = f.Reader();
            UNIT_ASSERT_EXCEPTION(restored.Load(f.File.Name(), nullptr), TCatBoostException);
        }
    }

    Y_UNIT_TEST(RejectsEveryTruncatedTagHeaderAndFloatBody) {
        TStringStream tail;
        WriteModelTail(&tail, Permutations, LeafCapacity, HistoryLeaves().size(), HistoryLeaves());
        for (size_t bytes = 1; bytes < tail.Str().size(); ++bytes) {
            TFixture f;
            f.Append([&](IOutputStream* out) { out->Write(tail.Str().data(), bytes); });
            auto restored = f.Reader();
            if (bytes < sizeof(ModelBasedTag)) {
                UNIT_ASSERT_EXCEPTION(restored.Load(f.File.Name(), nullptr), TCatBoostException);
            } else {
                UNIT_ASSERT_EXCEPTION(restored.Load(f.File.Name(), nullptr), TLoadEOF);
            }
        }
    }

    Y_UNIT_TEST(RejectsPartialAndCompleteJunkAfterHistory) {
        for (size_t bytes : {1u, 2u, 3u, 4u, 8u}) {
            TFixture f;
            f.AppendModel();
            f.Append([&](IOutputStream* out) { out->Write("bad!junk", bytes); });
            auto restored = f.Reader();
            UNIT_ASSERT_EXCEPTION_CONTAINS(restored.Load(f.File.Name(), nullptr), TCatBoostException,
                "model-based history has trailing data");
        }
    }
}
