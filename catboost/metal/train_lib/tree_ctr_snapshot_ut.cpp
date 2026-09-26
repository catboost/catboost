#include "tree_ctr_session.h"

#include <library/cpp/testing/unittest/registar.h>

using namespace NCB;

namespace {
    struct TCompletedSnapshot {
        TVector<ui32> StaticCounts = {0, 11};
        TMetalTreeCtrBatch Registry;
        TMetalSnapshot Snapshot;

        TCompletedSnapshot() {
            Registry.FirstFeature = 2;
            Registry.CtrUniqueValues = {37, 101};
            Registry.RegisteredCtrFlags = {1, 0};
            Snapshot.TreeCtrs = true;
            Snapshot.TreeCtrState = "registry decoded before metadata validation";
            Snapshot.Depths = {2};
            Snapshot.TreeCtrCounts = {0, 11, 37, 101};
            Snapshot.TreeCtrWeights = {1, 1, 1, 1};
            Snapshot.TreeCtrFlags = {2, 2, 3, 1};
            Snapshot.TreeCtrUsed = {0, 0, 1, 0};
            Snapshot.TreeCtrActive = {1, 1, 1, 0};
            Snapshot.UsedFeatures = Snapshot.TreeCtrUsed;
        }

        void Validate(bool ordered = false) const {
            ValidateMetalTreeCtrSnapshot(Registry, StaticCounts, Snapshot, ordered);
        }
    };
}

Y_UNIT_TEST_SUITE(TMetalTreeCtrCompletedSnapshotMetadata) {
    Y_UNIT_TEST(ValidPlainAndOrderedPayloadsNeedNoRuntime) {
        TCompletedSnapshot state;
        state.Validate();
        state.Snapshot.UsedFeatures.clear();
        state.Validate(true);
    }

    Y_UNIT_TEST(TerminalDynamicActivityCanIncludeInactiveRegisteredAndActiveTransientColumns) {
        TCompletedSnapshot state;
        state.Snapshot.TreeCtrActive = {1, 1, 0, 1};
        state.Validate();
    }

    Y_UNIT_TEST(RejectsMetadataWithValidShapesButDifferentRegistryCounts) {
        for (ui32 feature : {1u, 2u}) {
            TCompletedSnapshot state;
            ++state.Snapshot.TreeCtrCounts[feature];
            state.Snapshot.ValidateTreeCtrMetadata();
            UNIT_ASSERT_EXCEPTION(state.Validate(), TCatBoostException);
        }
    }

    Y_UNIT_TEST(RejectsFiniteWeightsThatDifferFromPreparedFeatures) {
        for (ui32 feature : {1u, 2u}) {
            TCompletedSnapshot state;
            state.Snapshot.TreeCtrWeights[feature] = 0.5f;
            state.Snapshot.ValidateTreeCtrMetadata();
            UNIT_ASSERT_EXCEPTION(state.Validate(), TCatBoostException);
        }
    }

    Y_UNIT_TEST(RejectsRegistrationMetadataThatDisagreesWithRestoredGridIdentity) {
        TCompletedSnapshot state;
        state.Snapshot.TreeCtrFlags[3] = 3;
        state.Snapshot.ValidateTreeCtrMetadata();
        UNIT_ASSERT_EXCEPTION(state.Validate(), TCatBoostException);
    }

    Y_UNIT_TEST(RejectsStaticColumnsClaimingDynamicIdentityOrInactivity) {
        TCompletedSnapshot state;
        state.Snapshot.TreeCtrFlags[1] = 3;
        state.Snapshot.ValidateTreeCtrMetadata();
        UNIT_ASSERT_EXCEPTION(state.Validate(), TCatBoostException);
        state.Snapshot.TreeCtrFlags[1] = 2;
        state.Snapshot.TreeCtrActive[1] = 0;
        state.Snapshot.ValidateTreeCtrMetadata();
        UNIT_ASSERT_EXCEPTION(state.Validate(), TCatBoostException);
    }

    Y_UNIT_TEST(RejectsDisagreeingPlainUsedFeaturePayloads) {
        TCompletedSnapshot state;
        state.Snapshot.UsedFeatures[2] = 0;
        state.Snapshot.ValidateTreeCtrMetadata();
        UNIT_ASSERT_EXCEPTION(state.Validate(), TCatBoostException);
    }

    Y_UNIT_TEST(RejectsWrongStaticPrefixAndRegistryCardinality) {
        TCompletedSnapshot state;
        --state.Registry.FirstFeature;
        UNIT_ASSERT_EXCEPTION(state.Validate(), TCatBoostException);
        ++state.Registry.FirstFeature;
        state.Registry.RegisteredCtrFlags.pop_back();
        UNIT_ASSERT_EXCEPTION(state.Validate(), TCatBoostException);
    }

    Y_UNIT_TEST(RejectsCoherentlyTruncatedMetadataThatOmitsARegistryColumn) {
        TCompletedSnapshot state;
        state.Snapshot.TreeCtrCounts.pop_back();
        state.Snapshot.TreeCtrWeights.pop_back();
        state.Snapshot.TreeCtrFlags.pop_back();
        state.Snapshot.TreeCtrUsed.pop_back();
        state.Snapshot.TreeCtrActive.pop_back();
        state.Snapshot.UsedFeatures.pop_back();
        state.Snapshot.ValidateTreeCtrMetadata();
        UNIT_ASSERT_EXCEPTION(state.Validate(), TCatBoostException);
    }
}
