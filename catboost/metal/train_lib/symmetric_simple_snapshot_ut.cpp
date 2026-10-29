#include "snapshot.h"

#include <library/cpp/testing/unittest/registar.h>

#include <limits>

using namespace NCB;

namespace {
    TMetalSnapshot CompletedSignedSnapshot(ui32 maxDepth = 1) {
        TMetalSnapshot snapshot;
        snapshot.Depths = {1};
        snapshot.SplitFeatures.resize(maxDepth);
        snapshot.SplitBins.resize(maxDepth);
        snapshot.SplitTypes.resize(maxDepth);
        snapshot.Leaves.resize(1u << maxDepth);
        snapshot.Leaves[0] = -.2f;
        snapshot.Leaves[1] = .1f;
        snapshot.Weights.resize(1u << maxDepth);
        snapshot.Weights[0] = -4;
        snapshot.Weights[1] = 2;
        snapshot.Predictions = {-.2f, -.2f, .1f, .1f};
        snapshot.PermutationPredictions = snapshot.Predictions;
        snapshot.PermutationMvsLambdas = {0};
        snapshot.PermutationMvsValid = {0};
        snapshot.History.TimeHistory.resize(1);
        return snapshot;
    }
}

Y_UNIT_TEST_SUITE(TMetalSymmetricSimpleSignedSnapshotWeights) {
    Y_UNIT_TEST(CompletedResumeRequiresExplicitCurrentConfigurationPermission) {
        const auto snapshot = CompletedSignedSnapshot();
        UNIT_ASSERT_EXCEPTION(snapshot.Validate(4, 1, 1), TCatBoostException);
        snapshot.Validate(4, 1, 1, 1, 1, 0, false, false, true);
        UNIT_ASSERT_EXCEPTION(snapshot.Validate(4, 1, 1, 1, 1, 0, false, false, false), TCatBoostException);
    }

    Y_UNIT_TEST(SignedPermissionStillRejectsNonfiniteAndMalformedActiveWeights) {
        for (float value : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity()}) {
            auto snapshot = CompletedSignedSnapshot();
            snapshot.Weights[1] = value;
            UNIT_ASSERT_EXCEPTION(snapshot.Validate(4, 1, 1, 1, 1, 0, false, false, true), TCatBoostException);
        }
        auto snapshot = CompletedSignedSnapshot();
        snapshot.Weights.pop_back();
        UNIT_ASSERT_EXCEPTION(snapshot.Validate(4, 1, 1, 1, 1, 0, false, false, true), TCatBoostException);
    }

    Y_UNIT_TEST(ValidationUsesActualDepthAndLeavesUnusedPaddingAlone) {
        auto snapshot = CompletedSignedSnapshot(2);
        snapshot.Weights[2] = std::numeric_limits<float>::quiet_NaN();
        snapshot.Weights[3] = -std::numeric_limits<float>::infinity();
        snapshot.Validate(4, 2, 1, 1, 1, 0, false, false, true);
        snapshot.Weights[0] = 4;
        snapshot.Validate(4, 2, 1);
        snapshot.Depths[0] = 3;
        UNIT_ASSERT_EXCEPTION(snapshot.Validate(4, 2, 1, 1, 1, 0, false, false, true), TCatBoostException);
    }
}
