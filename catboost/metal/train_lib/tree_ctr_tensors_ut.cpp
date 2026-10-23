#include "tree_ctr_tensors.h"

#include <library/cpp/testing/unittest/registar.h>

#include <limits>

namespace {
    TFeatureCombination Tensor(TVector<int> cats, TVector<TFloatSplit> floats = {},
                                TVector<TOneHotSplit> oneHot = {}) {
        TFeatureCombination result;
        result.CatFeatures = std::move(cats);
        result.BinFeatures = std::move(floats);
        result.OneHotFeatures = std::move(oneHot);
        return result;
    }

    TModelSplit CtrSplit(const TFeatureCombination& projection, float prior = 0.5f) {
        TModelCtrSplit split;
        split.Ctr.Base.Projection = projection;
        split.Ctr.Base.CtrType = ECtrType::Borders;
        split.Ctr.PriorNum = prior;
        split.Ctr.PriorDenom = 1;
        split.Ctr.TargetBorderIdx = 0;
        split.Border = 0.25f;
        return TModelSplit(split);
    }
}

// Source-level fixtures for CUDA tree_ctrs.cpp's separate CurrentTensor and
// PureTreeCtrTensorTracker transitions. Run model_ut +TMetalTreeCtrTensors;
// these cases need neither a CUDA device nor a trained model.
Y_UNIT_TEST_SUITE(TMetalTreeCtrTensors) {
    Y_UNIT_TEST(TestPureTreePacksReplaceEarlierPredicates) {
        NCB::TMetalTreeCtrTensorScheduler scheduler(TVector<int>{4, 2, 4}, 2);
        scheduler.BeginTree();
        UNIT_ASSERT(scheduler.GetActiveTensors().empty());
        scheduler.AddSplit(TModelSplit(TFloatSplit(7, 0.5f)));
        UNIT_ASSERT_VALUES_EQUAL(scheduler.GetActiveTensors().size(), 2);
        UNIT_ASSERT(scheduler.GetActiveTensors().contains(Tensor({2}, {{7, 0.5f}})));
        scheduler.AddSplit(TModelSplit(TOneHotSplit(3, -123)));
        UNIT_ASSERT_VALUES_EQUAL(scheduler.GetActiveTensors().size(), 2);
        UNIT_ASSERT(!scheduler.GetActiveTensors().contains(Tensor({2}, {{7, 0.5f}})));
        UNIT_ASSERT(scheduler.GetActiveTensors().contains(Tensor({2}, {{7, 0.5f}}, {{3, -123}})));
        scheduler.AddSplit(TModelSplit(TFloatSplit(8, 1.5f)));
        UNIT_ASSERT(scheduler.GetActiveTensors().contains(Tensor({4}, {{7, 0.5f}, {8, 1.5f}}, {{3, -123}})));
        for (const auto& tensor : scheduler.GetActiveTensors()) {
            UNIT_ASSERT_VALUES_EQUAL(NCB::TMetalTreeCtrTensorScheduler::Complexity(tensor), 2);
        }
    }

    Y_UNIT_TEST(TestCtrProjectionMergePacksPersistIndependently) {
        NCB::TMetalTreeCtrTensorScheduler scheduler(TVector<int>{1, 2, 3, 4}, 3);
        scheduler.AddSplit(CtrSplit(Tensor({1})));
        UNIT_ASSERT_VALUES_EQUAL(scheduler.GetActiveTensors().size(), 3);
        scheduler.AddSplit(TModelSplit(TFloatSplit(7, 0.5f)));
        UNIT_ASSERT_VALUES_EQUAL(scheduler.GetActiveTensors().size(), 7);
        scheduler.AddSplit(CtrSplit(Tensor({2})));
        // Older {1}+category packs persist; the new merged {1,2} base adds
        // only categories 3 and 4, with no pure-tree float predicate mixed in.
        UNIT_ASSERT(scheduler.GetActiveTensors().contains(Tensor({1, 3})));
        UNIT_ASSERT(scheduler.GetActiveTensors().contains(Tensor({1, 2, 3})));
        UNIT_ASSERT(scheduler.GetActiveTensors().contains(Tensor({1, 2, 4})));
        UNIT_ASSERT(!scheduler.GetActiveTensors().contains(Tensor({1, 2}, {{7, 0.5f}})));
        scheduler.AddSplit(TModelSplit(TFloatSplit(8, 1.0f)));
        UNIT_ASSERT(scheduler.GetActiveTensors().contains(Tensor({1, 3})));
        UNIT_ASSERT(!scheduler.GetActiveTensors().contains(Tensor({1}, {{7, 0.5f}})));
        UNIT_ASSERT(scheduler.GetActiveTensors().contains(Tensor({1}, {{7, 0.5f}, {8, 1.0f}})));
    }

    Y_UNIT_TEST(TestRejectedBaseDoesNotAdvanceCurrentTensor) {
        NCB::TMetalTreeCtrTensorScheduler scheduler(TVector<int>{0, 1, 2, 3}, 3);
        scheduler.AddSplit(CtrSplit(Tensor({0})));
        scheduler.AddSplit(CtrSplit(Tensor({1, 2})));
        // {0,1,2} has complexity equal to the maximum, so it cannot become
        // CurrentTensor. The next accepted merge must still start from {0}.
        scheduler.AddSplit(CtrSplit(Tensor({1})));
        UNIT_ASSERT(scheduler.GetActiveTensors().contains(Tensor({0, 1, 3})));
        for (const auto& tensor : scheduler.GetActiveTensors()) {
            UNIT_ASSERT(tensor.CatFeatures.size() <= 3);
        }
    }

    Y_UNIT_TEST(TestProjectionComponentsAreCanonicalAndDeduplicated) {
        NCB::TMetalTreeCtrTensorScheduler scheduler(TVector<int>{9, 2, 9}, 4);
        scheduler.AddSplit(CtrSplit(Tensor({3, 1, 3}, {{7, -0.0f}, {5, 1.0f}, {7, 0.0f}},
                                           {{4, -123}, {4, -123}})));
        const auto expected = Tensor({1, 2, 3}, {{5, 1.0f}, {7, 0.0f}}, {{4, -123}});
        UNIT_ASSERT(scheduler.GetActiveTensors().contains(expected));
        const auto before = scheduler.GetActiveTensors();
        // Different prior / selected CTR threshold does not change its tensor.
        scheduler.AddSplit(CtrSplit(Tensor({1, 3}, {{5, 1.0f}, {7, 0.0f}}, {{4, -123}}), 1.0f));
        UNIT_ASSERT(scheduler.GetActiveTensors() == before);
    }

    Y_UNIT_TEST(TestEstimatedSplitInvalidatesOnlyPureTreePacks) {
        NCB::TMetalTreeCtrTensorScheduler scheduler(TVector<int>{0, 1, 2}, 3);
        scheduler.AddSplit(CtrSplit(Tensor({0})));
        const auto persistent = scheduler.GetActiveTensors();
        scheduler.AddSplit(TModelSplit(TFloatSplit(7, 0.5f)));
        UNIT_ASSERT(scheduler.GetActiveTensors().size() > persistent.size());
        TModelSplit estimated;
        estimated.Type = ESplitType::EstimatedFeature;
        scheduler.AddSplit(estimated);
        UNIT_ASSERT(scheduler.GetActiveTensors() == persistent);
        scheduler.AddSplit(TModelSplit(TFloatSplit(8, 1.5f)));
        UNIT_ASSERT(scheduler.GetActiveTensors() == persistent);
        scheduler.AddSplit(CtrSplit(Tensor({1})));
        UNIT_ASSERT(scheduler.GetActiveTensors().contains(Tensor({0, 1, 2})));
    }

    Y_UNIT_TEST(TestEstimatedFloatProjectionAndTreeReset) {
        NCB::TMetalTreeCtrTensorScheduler scheduler(TVector<int>{0, 1}, 3, TVector<int>{7});
        scheduler.AddSplit(CtrSplit(Tensor({0}, {{7, 0.5f}})));
        UNIT_ASSERT(scheduler.GetActiveTensors().empty());
        scheduler.BeginTree();
        scheduler.AddSplit(TModelSplit(TFloatSplit(7, 0.5f)));
        UNIT_ASSERT(scheduler.GetActiveTensors().empty());
        scheduler.BeginTree();
        scheduler.AddSplit(TModelSplit(TFloatSplit(8, 0.5f)));
        UNIT_ASSERT_VALUES_EQUAL(scheduler.GetActiveTensors().size(), 2);
        scheduler.BeginTree();
        UNIT_ASSERT(scheduler.GetActiveTensors().empty());
    }

    Y_UNIT_TEST(TestNoEligibleCategoriesOrComplexityOne) {
        for (const auto& cats : {TVector<int>{}, TVector<int>{1, 2}}) {
            NCB::TMetalTreeCtrTensorScheduler scheduler(cats, 1);
            scheduler.AddSplit(TModelSplit(TFloatSplit(3, 0.5f)));
            scheduler.AddSplit(CtrSplit(Tensor({1})));
            UNIT_ASSERT(scheduler.GetActiveTensors().empty());
        }
    }

    Y_UNIT_TEST(TestInvalidModelIndicesAndNonFiniteBorders) {
        UNIT_ASSERT_EXCEPTION(NCB::TMetalTreeCtrTensorScheduler(TVector<int>{-1}, 2), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(NCB::TMetalTreeCtrTensorScheduler(TVector<int>{1}, 0), TCatBoostException);
        NCB::TMetalTreeCtrTensorScheduler scheduler(TVector<int>{1}, 2);
        UNIT_ASSERT_EXCEPTION(scheduler.AddSplit(TModelSplit(TFloatSplit(-1, 0.5f))), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(scheduler.AddSplit(TModelSplit(TFloatSplit(0, std::numeric_limits<float>::quiet_NaN()))),
                              TCatBoostException);
        UNIT_ASSERT(scheduler.GetActiveTensors().empty());
    }
}
