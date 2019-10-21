#include <catboost/libs/model/ut/lib/model_test_helpers.h>

#include <catboost/libs/model/model_export/model_exporter.h>

#include <library/unittest/registar.h>

using namespace std;
using namespace NCB;

Y_UNIT_TEST_SUITE(TJsonModelExport) {
    Y_UNIT_TEST(TestWithCatFeatures) {
        TFullModel model = TrainFloatCatboostModel();
        ExportModel(model, "model.json", EModelType::Json);
        auto model2 = ReadModel("model.json", EModelType::Json);
        UNIT_ASSERT(model.ObliviousTrees->GetDimensionsCount() == model2.ObliviousTrees->GetDimensionsCount());
        UNIT_ASSERT(model.ObliviousTrees->GetTreeSplits() == model2.ObliviousTrees->GetTreeSplits());
        UNIT_ASSERT(model.ObliviousTrees->GetTreeSizes() == model2.ObliviousTrees->GetTreeSizes());
        UNIT_ASSERT(model.ObliviousTrees->GetTreeStartOffsets() == model2.ObliviousTrees->GetTreeStartOffsets());
        UNIT_ASSERT(model.ObliviousTrees->GetCatFeatures() == model2.ObliviousTrees->GetCatFeatures());
        UNIT_ASSERT(model.ObliviousTrees->GetFloatFeatures() == model2.ObliviousTrees->GetFloatFeatures());
        UNIT_ASSERT(model.ObliviousTrees->GetOneHotFeatures() == model2.ObliviousTrees->GetOneHotFeatures());
        UNIT_ASSERT(model.ObliviousTrees->GetCtrFeatures() == model2.ObliviousTrees->GetCtrFeatures());
        UNIT_ASSERT(model.ObliviousTrees->GetLeafValues().ysize() == model2.ObliviousTrees->GetLeafValues().ysize());
        for (int idx = 0; idx < model.ObliviousTrees->GetLeafValues().ysize(); ++idx) {
            UNIT_ASSERT_DOUBLES_EQUAL(model.ObliviousTrees->GetLeafValues()[idx], model2.ObliviousTrees->GetLeafValues()[idx], 1e-9);
        }
    }
    Y_UNIT_TEST(TestEmptyLeafWeights) {
        TFullModel model = TrainFloatCatboostModel();
        model.ObliviousTrees.GetMutable()->ClearLeafWeights();
        ExportModel(model, "model.json", EModelType::Json);
        model = ReadModel("model.json", EModelType::Json);
        UNIT_ASSERT(model.ObliviousTrees->GetLeafWeights().empty());
    }
}
