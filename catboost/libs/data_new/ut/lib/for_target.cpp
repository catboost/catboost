#include "for_target.h"

#include <util/generic/mapfindptr.h>

#include <library/unittest/registar.h>


namespace NCB {
    namespace NDataNewUT {

    #define COMPARE_TARGET_FIELD(FIELD) \
        UNIT_ASSERT_EQUAL( \
            lhs.Get##FIELD(), \
            rhs.Get##FIELD() \
        );


    void CompareTargets(const TBinClassTarget& lhs, const TBinClassTarget& rhs) {
        UNIT_ASSERT_EQUAL(*lhs.GetObjectsGrouping(), *rhs.GetObjectsGrouping());

        COMPARE_TARGET_FIELD(Target);
        COMPARE_TARGET_FIELD(Weights);
        COMPARE_TARGET_FIELD(Baseline);
    }

    void CompareTargets(const TMultiClassTarget& lhs, const TMultiClassTarget& rhs) {
        UNIT_ASSERT_EQUAL(*lhs.GetObjectsGrouping(), *rhs.GetObjectsGrouping());

        COMPARE_TARGET_FIELD(ClassCount);
        COMPARE_TARGET_FIELD(Target);
        COMPARE_TARGET_FIELD(Weights);
        COMPARE_TARGET_FIELD(Baseline);

    }

    void CompareTargets(const TRegressionTarget& lhs, const TRegressionTarget& rhs) {
        UNIT_ASSERT_EQUAL(*lhs.GetObjectsGrouping(), *rhs.GetObjectsGrouping());

        COMPARE_TARGET_FIELD(Target);
        COMPARE_TARGET_FIELD(Weights);
        COMPARE_TARGET_FIELD(Baseline);
    }

    void CompareTargets(const TGroupwiseRankingTarget& lhs, const TGroupwiseRankingTarget& rhs) {
        UNIT_ASSERT_EQUAL(*lhs.GetObjectsGrouping(), *rhs.GetObjectsGrouping());

        COMPARE_TARGET_FIELD(Target);
        COMPARE_TARGET_FIELD(Weights);
        COMPARE_TARGET_FIELD(Baseline);
        COMPARE_TARGET_FIELD(GroupInfo);
    }

    void CompareTargets(const TGroupPairwiseRankingTarget& lhs, const TGroupPairwiseRankingTarget& rhs) {
        UNIT_ASSERT_EQUAL(*lhs.GetObjectsGrouping(), *rhs.GetObjectsGrouping());

        COMPARE_TARGET_FIELD(Baseline);
        COMPARE_TARGET_FIELD(GroupInfo);
    }

    #undef COMPARE_TARGET_FIELD


    void CompareTargetDataProviders(const TTargetDataProviders& lhs, const TTargetDataProviders& rhs) {
        UNIT_ASSERT_VALUES_EQUAL(lhs.size(), rhs.size());

        for (const auto& lhsSpecAndDataProvider : lhs) {
            auto* lhsTarget = lhsSpecAndDataProvider.second.Get();
            auto* rhsPtr = MapFindPtr(rhs, lhsSpecAndDataProvider.first);
            UNIT_ASSERT(rhsPtr);
            auto* rhsTarget = rhsPtr->Get();
            UNIT_ASSERT_VALUES_EQUAL(lhsTarget->GetSpecification(), rhsTarget->GetSpecification());

    #define COMPARE_TYPE_CASE(targetType) \
            if (dynamic_cast<targetType*>(lhsTarget)) { \
                CompareTargets( \
                    dynamic_cast<targetType&>(*lhsTarget), \
                    dynamic_cast<targetType&>(*rhsTarget) \
                ); \
                continue; \
            }

            COMPARE_TYPE_CASE(TBinClassTarget)
            COMPARE_TYPE_CASE(TMultiClassTarget)
            COMPARE_TYPE_CASE(TRegressionTarget)
            COMPARE_TYPE_CASE(TGroupwiseRankingTarget)
            COMPARE_TYPE_CASE(TGroupPairwiseRankingTarget)

    #undef COMPARE_TYPE_CASE

        }
    }

    }

}


