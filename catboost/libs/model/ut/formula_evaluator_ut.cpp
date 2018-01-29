#include <catboost/libs/model/model.h>
#include <catboost/libs/model/formula_evaluator.h>
#include <library/unittest/registar.h>

using namespace std;

TFullModel SimpleFloatModel() {
    TFullModel model;
    model.ObliviousTrees.FloatFeatures = {
        TFloatFeature{
            false, 0, 0,
            {1.f, 2.f}, // bin splits 0, 1
            ""
        },
        TFloatFeature{
            false, 1, 1,
            {0.5f}, // bin split 2
            ""
        },
        TFloatFeature{
            false, 2, 2,
            {0.5f}, // bin split 3
            ""
        }
    };
    {
        TVector<int> tree = {1, 2, 3};
        model.ObliviousTrees.AddBinTree(tree);
        model.ObliviousTrees.LeafValues = {
            {0., 1., 2., 3., 4., 5., 6., 7.}
        };
    }
    model.UpdateDynamicData();
    return model;
}

TFullModel MultiValueFloatModel() {
    TFullModel model;
    model.ObliviousTrees.FloatFeatures = {
        TFloatFeature{
            false, 0, 0,
            {0.5f}, // bin split 0
            ""
        },
        TFloatFeature{
            false, 1, 1,
            {0.5f}, // bin split 1
            ""
        }
    };
    {
        TVector<int> tree = {0, 1};
        model.ObliviousTrees.AddBinTree(tree);
        model.ObliviousTrees.LeafValues = {
            {00., 10., 20.,
             01., 11., 21.,
             02., 12., 22.,
             03., 13., 23.}
        };
        model.ObliviousTrees.ApproxDimension = 3;
    }
    model.UpdateDynamicData();
    return model;
}

SIMPLE_UNIT_TEST_SUITE(TObliviousTreeModel) {
    SIMPLE_UNIT_TEST(TestFlatCalcFloat) {
        auto modelCalcer = SimpleFloatModel();
        TVector<double> result(8);
        TVector<TConstArrayRef<float>> features = {
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
        TVector<double> canonVals = {
            0., 1., 2., 3.,
            4., 5., 6., 7.};
        UNIT_ASSERT_EQUAL(canonVals, result);
    }

    SIMPLE_UNIT_TEST(TestFlatCalcMultiVal) {
        auto modelCalcer = MultiValueFloatModel();
        TVector<TConstArrayRef<float>> features = {
            {0.f, 0.f},
            {1.f, 0.f},
            {0.f, 1.f},
            {1.f, 1.f}};
        TVector<double> result(features.size() * 3);
        modelCalcer.CalcFlat(
            features,
            result);
        TVector<double> canonVals = {
            00., 10., 20.,
            01., 11., 21.,
            02., 12., 22.,
            03., 13., 23.,
        };
        UNIT_ASSERT_EQUAL(canonVals, result);
    }
}
