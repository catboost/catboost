#include <catboost/libs/model/model.h>
#include <catboost/libs/model/model_calcer.h>
#include <library/unittest/registar.h>

using namespace std;

TFullModel SimpleFloatModel() {
    TFullModel model;
    model.Borders = {{1.f, 2.f}, {0.5f}, {0.5f}};
    {
        TTensorStructure3 tree;
        tree.Add(TModelSplit(TBinFeature(0, 1)));
        tree.Add(TModelSplit(TBinFeature(1, 0)));
        tree.Add(TModelSplit(TBinFeature(2, 0)));
        model.TreeStruct.push_back(tree);
        yvector<yvector<double>> leafs = {
            {0., 1., 2., 3.,
             4., 5., 6., 7.}};
        model.LeafValues.emplace_back(std::move(leafs));
    }
    return model;
}

TFullModel MultiValueFloatModel() {
    TFullModel model;
    model.Borders = {{0.5f}, {0.5f}};
    {
        TTensorStructure3 tree;
        tree.Add(TModelSplit(TBinFeature(0, 0)));
        tree.Add(TModelSplit(TBinFeature(1, 0)));
        model.TreeStruct.push_back(tree);
        yvector<yvector<double>> leafs = {
            {00., 01., 02., 03.},
            {10., 11., 12., 13.},
            {20., 21., 22., 23.},
        };
        model.LeafValues.emplace_back(std::move(leafs));
    }
    return model;
}

SIMPLE_UNIT_TEST_SUITE(TModelCalcer) {
    SIMPLE_UNIT_TEST(TestFlatCalcFloat) {
        NCatBoost::TModelCalcer modelCalcer(SimpleFloatModel());
        yvector<double> result(8);
        yvector<NArrayRef::TConstArrayRef<float>> features = {
            {0.f, 0.f, 0.f},
            {3.f, 0.f, 0.f},
            {0.f, 1.f, 0.f},
            {3.f, 1.f, 0.f},
            {0.f, 0.f, 1.f},
            {3.f, 0.f, 1.f},
            {0.f, 1.f, 1.f},
            {3.f, 1.f, 1.f},
        };
        modelCalcer.CalcFlat(
            features,
            result);
        yvector<double> canonVals = {
            0., 1., 2., 3.,
            4., 5., 6., 7.};
        UNIT_ASSERT_EQUAL(canonVals, result);
    }

    SIMPLE_UNIT_TEST(TestFlatCalcMultiVal) {
        NCatBoost::TModelCalcer modelCalcer(MultiValueFloatModel());
        yvector<NArrayRef::TConstArrayRef<float>> features = {
            {0.f, 0.f},
            {1.f, 0.f},
            {0.f, 1.f},
            {1.f, 1.f}};
        yvector<double> result(features.size() * 3);
        modelCalcer.CalcFlat(
            features,
            result);
        yvector<double> canonVals = {
            00., 10., 20.,
            01., 11., 21.,
            02., 12., 22.,
            03., 13., 23.,
        };
        UNIT_ASSERT_EQUAL(canonVals, result);
    }
}
