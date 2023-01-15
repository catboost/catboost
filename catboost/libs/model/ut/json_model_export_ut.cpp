#include <catboost/libs/model/ut/lib/model_test_helpers.h>

#include <catboost/libs/model/model_export/model_exporter.h>

#include <library/cpp/testing/unittest/registar.h>

using namespace std;
using namespace NCB;

Y_UNIT_TEST_SUITE(TJsonModelExport) {
    Y_UNIT_TEST(TestWithCatFeatures) {
        TFullModel model = TrainFloatCatboostModel();
        ExportModel(model, "model.json", EModelType::Json);
        auto model2 = ReadModel("model.json", EModelType::Json);
        UNIT_ASSERT(model.ModelTrees->GetDimensionsCount() == model2.ModelTrees->GetDimensionsCount());
        UNIT_ASSERT(model.ModelTrees->GetModelTreeData()->GetTreeSplits() == model2.ModelTrees->GetModelTreeData()->GetTreeSplits());
        UNIT_ASSERT(model.ModelTrees->GetModelTreeData()->GetTreeSizes() == model2.ModelTrees->GetModelTreeData()->GetTreeSizes());
        UNIT_ASSERT(model.ModelTrees->GetModelTreeData()->GetTreeStartOffsets() == model2.ModelTrees->GetModelTreeData()->GetTreeStartOffsets());
        UNIT_ASSERT(model.ModelTrees->GetCatFeatures() == model2.ModelTrees->GetCatFeatures());
        UNIT_ASSERT(model.ModelTrees->GetFloatFeatures() == model2.ModelTrees->GetFloatFeatures());
        UNIT_ASSERT(model.ModelTrees->GetOneHotFeatures() == model2.ModelTrees->GetOneHotFeatures());
        UNIT_ASSERT(model.ModelTrees->GetCtrFeatures() == model2.ModelTrees->GetCtrFeatures());
        UNIT_ASSERT(model.ModelTrees->GetModelTreeData()->GetLeafValues().ysize() == model2.ModelTrees->GetModelTreeData()->GetLeafValues().ysize());
        for (int idx = 0; idx < model.ModelTrees->GetModelTreeData()->GetLeafValues().ysize(); ++idx) {
            UNIT_ASSERT_DOUBLES_EQUAL(model.ModelTrees->GetModelTreeData()->GetLeafValues()[idx], model2.ModelTrees->GetModelTreeData()->GetLeafValues()[idx], 1e-9);
        }
    }
    Y_UNIT_TEST(TestEmptyLeafWeights) {
        TFullModel model = TrainFloatCatboostModel();
        model.ModelTrees.GetMutable()->ClearLeafWeights();
        ExportModel(model, "model.json", EModelType::Json);
        model = ReadModel("model.json", EModelType::Json);
        UNIT_ASSERT(model.ModelTrees->GetModelTreeData()->GetLeafWeights().empty());
    }
}
