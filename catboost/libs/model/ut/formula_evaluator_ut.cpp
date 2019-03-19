#include "model_test_helpers.h"

#include <catboost/libs/data_new/data_provider_builders.h>
#include <catboost/libs/model/formula_evaluator.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(TObliviousTreeModel) {
    Y_UNIT_TEST(TestFlatCalcFloat) {
        auto modelCalcer = SimpleFloatModel();
        TVector<double> result(8);
        TVector<TVector<float>> data = {
            {0.f, 0.f, 0.f},
            {3.f, 0.f, 0.f},
            {0.f, 1.f, 0.f},
            {3.f, 1.f, 0.f},
            {0.f, 0.f, 1.f},
            {3.f, 0.f, 1.f},
            {0.f, 1.f, 1.f},
            {3.f, 1.f, 1.f},
        };
        TVector<TConstArrayRef<float>> features(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            features[i] = data[i];
        };
        modelCalcer.CalcFlat(
            features,
            result);
        TVector<double> canonVals = {
            0., 1., 2., 3.,
            4., 5., 6., 7.};
        UNIT_ASSERT_EQUAL(canonVals, result);
        modelCalcer.ObliviousTrees.ConvertObliviousToAsymmetric();
        modelCalcer.CalcFlat(
                features,
                result);
        UNIT_ASSERT_EQUAL(canonVals, result);
    }

    Y_UNIT_TEST(TestFlatCalcMultiVal) {
        auto modelCalcer = MultiValueFloatModel();
        TVector<TVector<float>> data = {
            {0.f, 0.f},
            {1.f, 0.f},
            {0.f, 1.f},
            {1.f, 1.f}};
        TVector<TConstArrayRef<float>> features(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            features[i] = data[i];
        };
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

    Y_UNIT_TEST(TestCatOnlyModel) {
        const auto model = TrainCatOnlyModel();

        const auto applySingle = [&] {
            const TVector<TStringBuf> f[] = {{"a", "b", "c"}};
            double result = 0.;
            model.Calc({}, f, MakeArrayRef(&result, 1));
        };
        UNIT_ASSERT_NO_EXCEPTION(applySingle());

        const auto applyBatch = [&] {
            const TVector<TStringBuf> f[] = {{"a", "b", "c"}, {"d", "e", "f"}, {"g", "h", "k"}};
            double results[3];
            model.Calc({}, f, results);
        };
        UNIT_ASSERT_NO_EXCEPTION(applyBatch());
    }
}

Y_UNIT_TEST_SUITE(TNonSymmetricTreeModel) {
    Y_UNIT_TEST(TestFlatCalcFloat) {
        auto modelCalcer = SimpleAsymmetricModel();
        TVector<double> result(8);
        TVector<TVector<float>> data = {
            {0.f, 0.f, 0.f},
            {3.f, 0.f, 0.f},
            {0.f, 1.f, 0.f},
            {3.f, 1.f, 0.f},
            {0.f, 0.f, 1.f},
            {3.f, 0.f, 1.f},
            {0.f, 1.f, 1.f},
            {3.f, 1.f, 1.f},
        };
        TVector<TConstArrayRef<float>> features(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            features[i] = data[i];
        };
        modelCalcer.CalcFlat(
            features,
            result);
        TVector<double> canonVals = {
            101., 203., 102., 303.,
            111., 213., 112., 313.};
        UNIT_ASSERT_EQUAL(canonVals, result);
        TStringStream strStream;
        modelCalcer.Save(&strStream);
        TFullModel deserializedModel;
        deserializedModel.Load(&strStream);
        deserializedModel.CalcFlat(
            features,
            result);
        UNIT_ASSERT_EQUAL(canonVals, result);
    }
}
