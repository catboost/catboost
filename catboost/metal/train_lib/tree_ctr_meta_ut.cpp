#include "tree_ctr_meta.h"
#include "tree_ctr_tensors.h"

#include <library/cpp/testing/unittest/registar.h>

#include <numeric>

namespace {
    TModelSplit Ctr(const TVector<int>& cats) {
        TModelCtrSplit split;
        split.Ctr.Base.Projection.CatFeatures = cats;
        split.Ctr.Base.CtrType = ECtrType::Borders;
        split.Ctr.PriorNum = 0.5f;
        split.Ctr.PriorDenom = 1;
        split.Border = 0.5f;
        return TModelSplit(split);
    }
}

Y_UNIT_TEST_SUITE(TMetalTreeCtrMeta) {
    Y_UNIT_TEST(TestBaseHashUsesSourceManagerSlicesAndOriginalPerfectHashBins) {
        NCB::TFeaturesLayout layout(6, TVector<ui32>{1, 4}, TVector<ui32>{2}, TVector<ui32>{5}, {});
        layout.IgnoreExternalFeature(0);
        auto info = MakeIntrusive<NCB::TQuantizedFeaturesInfo>(layout, TConstArrayRef<ui32>(),
            NCatboostOptions::TBinarizationOptions(EBorderSelectionType::Uniform, 1, ENanMode::Forbidden));
        TVector<float> wide(256);
        std::iota(wide.begin(), wide.end(), 0.f);
        info->SetBorders(NCB::TFloatFeatureIdx(0), std::move(wide));
        info->SetBorders(NCB::TFloatFeatureIdx(1), {0.25f, 0.75f});
        NCB::TCatFeaturePerfectHash perfect;
        perfect.Map[0xffffffffu] = {0, 3};
        perfect.Map[7] = {1, 3}; // Hash sorting would reverse these bin IDs.
        info->UpdateCategoricalFeaturesPerfectHash(NCB::TCatFeatureIdx(0), std::move(perfect));
        const auto ids = NCB::MakeMetalOriginalFeatureManagerIds(layout, *info);
        TFeatureCombination projection;
        projection.CatFeatures = {1};
        projection.BinFeatures = {{1, 0.75f}, {0, 255.f}};
        projection.OneHotFeatures = {{0, -1}};
        NCatboostCuda::TFeatureTensor expected;
        expected.AddCatFeature(4);
        expected.AddBinarySplit({1, 0, NCatboostCuda::EBinSplitType::TakeGreater});
        expected.AddBinarySplit({3, 1, NCatboostCuda::EBinSplitType::TakeGreater});
        expected.AddBinarySplit({2, 0, NCatboostCuda::EBinSplitType::TakeBin});
        const auto actual = NCB::MakeMetalCudaTreeCtrTensor(projection, layout, *info, ids);
        UNIT_ASSERT(actual == expected);
        UNIT_ASSERT_VALUES_EQUAL(actual.GetHash(), expected.GetHash());
        projection.OneHotFeatures = {{0, 7}};
        UNIT_ASSERT(NCB::MakeMetalCudaTreeCtrTensor(projection, layout, *info, ids).GetHash() != expected.GetHash());
        projection.OneHotFeatures = {{0, 99}};
        UNIT_ASSERT_EXCEPTION(NCB::MakeMetalCudaTreeCtrTensor(projection, layout, *info, ids), TCatBoostException);
    }

    Y_UNIT_TEST(TestLogicalBasesSurviveCurrentProjectionMergesAndPureReplacement) {
        NCB::TMetalTreeCtrTensorScheduler scheduler(TVector<int>{0, 1, 2}, 3);
        UNIT_ASSERT(scheduler.GetActiveTensorPacks().empty());
        scheduler.AddSplit(Ctr({0}));
        auto packs = scheduler.GetActiveTensorPacks();
        UNIT_ASSERT_VALUES_EQUAL(packs.size(), 1);
        UNIT_ASSERT(packs[0].Base.CatFeatures == TVector<int>({0}));
        UNIT_ASSERT_VALUES_EQUAL(packs[0].Tensors.size(), 2);
        scheduler.AddSplit(TModelSplit(TFloatSplit(3, 0.5f)));
        scheduler.AddSplit(Ctr({1}));
        packs = scheduler.GetActiveTensorPacks();
        UNIT_ASSERT_VALUES_EQUAL(packs.size(), 3);
        TSet<TFeatureCombination> combined;
        for (const auto& pack : packs) combined.insert(pack.Tensors.begin(), pack.Tensors.end());
        UNIT_ASSERT(combined == scheduler.GetActiveTensors());
        UNIT_ASSERT(packs.back().Base.CatFeatures.empty());
        UNIT_ASSERT(packs.back().Base.BinFeatures == TVector<TFloatSplit>({{3, 0.5f}}));
        TModelSplit estimated;
        estimated.Type = ESplitType::EstimatedFeature;
        scheduler.AddSplit(estimated);
        UNIT_ASSERT_VALUES_EQUAL(scheduler.GetActiveTensorPacks().size(), 2);
        scheduler.BeginTree();
        UNIT_ASSERT(scheduler.GetActiveTensorPacks().empty());
    }

    Y_UNIT_TEST(TestConfiguredPolicyBounds) {
        UNIT_ASSERT_VALUES_EQUAL(NCB::MetalTreeCtrConfiguredPolicy(1), 0);
        for (ui32 borders : {2u, 7u, 15u}) UNIT_ASSERT_VALUES_EQUAL(NCB::MetalTreeCtrConfiguredPolicy(borders), 1);
        for (ui32 borders : {16u, 31u, 255u}) UNIT_ASSERT_VALUES_EQUAL(NCB::MetalTreeCtrConfiguredPolicy(borders), 2);
        UNIT_ASSERT_EXCEPTION(NCB::MetalTreeCtrConfiguredPolicy(0), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(NCB::MetalTreeCtrConfiguredPolicy(256), TCatBoostException);
    }
}
