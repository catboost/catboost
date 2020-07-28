#include <catboost/libs/model/ut/lib/model_test_helpers.h>

#include <catboost/libs/model/model_export/export_helpers.h>

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestExportHelpers) {
    Y_UNIT_TEST(TestOutputBorders) {
        TFullModel model;
        TFloatFeature feature;
        feature.Borders = {0, 0.5};
        model.ModelTrees.GetMutable()->AddFloatFeature(feature);
        auto result = NCatboostModelExportHelpers::OutputBorders(model, true);
        UNIT_ASSERT_EQUAL(result, "0.f, 0.5f");
    }

    Y_UNIT_TEST(TestOutputArrayInitializer) {
        UNIT_ASSERT_EQUAL(NCatboostModelExportHelpers::OutputArrayInitializer(TVector<int>({1, -23})), "1, -23");
    }
}
