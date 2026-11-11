#include "fixed_splits.h"
#include <catboost/metal/native/metal_fixed_splits.h>
#include <library/cpp/testing/unittest/registar.h>

#include <cmath>
#include <numeric>

namespace {
    struct TFixture {
        NCB::TFeaturesLayout Layout{7, TVector<ui32>{1}, TVector<ui32>{3}, TVector<ui32>{5}, {}};
        NCB::TQuantizedFeaturesInfoPtr Info;

        TFixture() {
            Layout.IgnoreExternalFeature(0);
            Info = MakeIntrusive<NCB::TQuantizedFeaturesInfo>(Layout, TConstArrayRef<ui32>(),
                NCatboostOptions::TBinarizationOptions(EBorderSelectionType::Uniform, 1, ENanMode::Forbidden));
            TVector<float> wide(256);
            std::iota(wide.begin(), wide.end(), 0.f);
            Info->SetBorders(NCB::TFloatFeatureIdx(0), std::move(wide));
            Info->SetBorders(NCB::TFloatFeatureIdx(1), {0.5f});
            Info->SetBorders(NCB::TFloatFeatureIdx(2), {0.25f, 0.75f});
            Info->SetBorders(NCB::TFloatFeatureIdx(3), {});
        }
    };

    CBMFixedSplits Config(const std::vector<uint32_t>& fixed) {
        const uint32_t features[] = {0, 1, 2};
        const uint32_t bins[] = {0, 0, 0};
        CBMFixedSplits result;
        result.Configure(fixed.size(), fixed.data(), 3, 3, features, bins, nullptr);
        return result;
    }
}

Y_UNIT_TEST_SUITE(TMetalFixedBinarySplits) {
    Y_UNIT_TEST(TestOriginalManagerIdsPreserveIgnoredSlicesAndSkipEstimatedSources) {
        TFixture fixture;
        const auto ids = NCB::MakeMetalOriginalFeatureManagerIds(fixture.Layout, *fixture.Info);
        UNIT_ASSERT_VALUES_EQUAL(ids.NextId, 6);
        UNIT_ASSERT(ids.ByFlatIndex[0] == TVector<ui32>({0, 1}));
        UNIT_ASSERT(ids.ByFlatIndex[1] == TVector<ui32>({2}));
        UNIT_ASSERT(ids.ByFlatIndex[2] == TVector<ui32>({3}));
        UNIT_ASSERT(ids.ByFlatIndex[3].empty());
        UNIT_ASSERT(ids.ByFlatIndex[4] == TVector<ui32>({4}));
        UNIT_ASSERT(ids.ByFlatIndex[5].empty());
        UNIT_ASSERT(ids.ByFlatIndex[6] == TVector<ui32>({5}));
        UNIT_ASSERT_VALUES_EQUAL(ids.ByManagerId[1].BorderOffset, 255);
        UNIT_ASSERT_VALUES_EQUAL(ids.ByManagerId[1].BorderCount, 1);
        UNIT_ASSERT_VALUES_EQUAL(ids.ByManagerId[5].BorderCount, 0);
    }

    Y_UNIT_TEST(TestResolveManagerIdToPreparedDenseGridAndRetainRepeats) {
        TFixture fixture;
        const TVector<ui32> dense = {4, 2, 6};
        const TVector<ui32> fixed = {3, 3};
        UNIT_ASSERT(NCB::ResolveMetalFixedBinarySplits(fixed, fixture.Layout, *fixture.Info, dense) == TVector<ui32>({1, 1}));
        for (ui32 invalid : {0u, 1u, 2u, 4u, 5u, 6u, Max<ui32>()}) {
            const TVector<ui32> requested = {invalid};
            UNIT_ASSERT_EXCEPTION(NCB::ResolveMetalFixedBinarySplits(requested, fixture.Layout, *fixture.Info, dense), TCatBoostException);
        }
        UNIT_ASSERT_EXCEPTION(NCB::ResolveMetalFixedBinarySplits(fixed, fixture.Layout, *fixture.Info, TVector<ui32>{4, 6}), TCatBoostException);
    }

    Y_UNIT_TEST(TestCountedConfigurationRejectsMalformedCandidatesWithoutMutation) {
        auto config = Config({1});
        const uint32_t feature[] = {0, 1, 1};
        const uint32_t bins[] = {0, 0, 1};
        const uint32_t request[] = {1};
        UNIT_ASSERT_EXCEPTION(config.Configure(1, request, 3, 3, feature, bins, nullptr), std::invalid_argument);
        UNIT_ASSERT_VALUES_EQUAL(config.At(0, 9).Feature, 1);
        UNIT_ASSERT_VALUES_EQUAL(config.At(0, 9).Leaf, 9);
        UNIT_ASSERT_EXCEPTION(config.Configure(1, nullptr, 3, 3, feature, bins, nullptr), std::invalid_argument);
        const uint8_t oneHot[] = {0, 1, 0};
        UNIT_ASSERT_EXCEPTION(config.Configure(1, request, 3, 2, feature, bins, oneHot), std::invalid_argument);
        config.Configure(0, nullptr, 3, 3, feature, bins, nullptr);
        UNIT_ASSERT(!config.Enabled());
    }

    Y_UNIT_TEST(TestGlobalDepthAndCachedLossguideWinnerAfterFinalFixedLevel) {
        auto config = Config({0, 1});
        CBMFixedSplitSearch search(config);
        uint32_t depths[] = {0, 0, 0};
        uint32_t offsets[] = {0, 16, 16, 16};
        CBMGreedySplit winners[3] = {};
        UNIT_ASSERT(!search.Begin(1, depths, offsets, 4, 1));
        UNIT_ASSERT(search.IsForced() && search.ForceAll());
        search.Merge(winners);
        UNIT_ASSERT_VALUES_EQUAL(winners[0].Feature, 0);
        UNIT_ASSERT(std::isinf(winners[0].Gain) && winners[0].Gain < 0);
        search.Split(0, 1);
        depths[0] = depths[1] = 1; offsets[1] = 8;
        UNIT_ASSERT(!search.Begin(2, depths, offsets, 4, 1));
        UNIT_ASSERT(search.IsForced() && !search.ForceAll());
        search.Merge(winners);
        UNIT_ASSERT_VALUES_EQUAL(winners[0].Feature, 1);
        UNIT_ASSERT_VALUES_EQUAL(winners[1].Feature, 1);
        // Only leaf zero is selected at the last Lossguide fixed level.
        search.Split(0, 2);
        depths[0] = depths[2] = 2; offsets[1] = 4; offsets[2] = 12;
        UNIT_ASSERT(search.Begin(3, depths, offsets, 4, 1));
        UNIT_ASSERT(!search.IsForced());
        winners[0] = {2, 2, 0, 0, -1, 1, 0, 0};
        winners[1] = {2, 2, 0, 0, 7, 1, 1, 1};
        winners[2] = {2, 2, 0, 0, -2, 1, 0, 2};
        search.Merge(winners);
        UNIT_ASSERT_VALUES_EQUAL(winners[1].Feature, 1);
        UNIT_ASSERT_VALUES_EQUAL(winners[1].Error, 0);
        UNIT_ASSERT(winners[1].Gain < winners[2].Gain);
    }

    Y_UNIT_TEST(TestRepeatedForcedSplitKeepsEmptyChildrenTerminal) {
        auto config = Config({0, 0, 1});
        CBMFixedSplitSearch search(config);
        uint32_t depths[] = {0, 1, 0};
        uint32_t offsets[] = {0, 1, 1, 1};
        CBMGreedySplit winners[3] = {};
        // Root is eligible even when its row count is below the minimum.
        UNIT_ASSERT(!search.Begin(1, depths, offsets, 4, 2));
        search.Merge(winners); UNIT_ASSERT(winners[0].Valid);
        search.Split(0, 1); depths[0] = 1;
        UNIT_ASSERT(!search.Begin(2, depths, offsets, 4, 2));
        search.Merge(winners);
        UNIT_ASSERT(!winners[0].Valid && !winners[1].Valid);
        // New trees reconstruct a fresh root from immutable configuration.
        CBMFixedSplitSearch restarted(config); depths[0] = 0;
        restarted.Begin(1, depths, offsets, 4, 2); restarted.Merge(winners);
        UNIT_ASSERT(winners[0].Valid && winners[0].Feature == 0);
    }
}
