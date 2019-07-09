#include "model_test_helpers.h"

#include <library/unittest/registar.h>

using namespace std;

Y_UNIT_TEST_SUITE(TJsonModelExport) {
    Y_UNIT_TEST(TestWithCatFeatures) {
        TFullModel model = TrainFloatCatboostModel();
        ExportModel(model, "model.json", EModelType::Json);
        auto model2 = ReadModel("model.json", EModelType::Json);
        UNIT_ASSERT(model.ObliviousTrees->ApproxDimension == model2.ObliviousTrees->ApproxDimension);
        UNIT_ASSERT(model.ObliviousTrees->TreeSplits == model2.ObliviousTrees->TreeSplits);
        UNIT_ASSERT(model.ObliviousTrees->TreeSizes == model2.ObliviousTrees->TreeSizes);
        UNIT_ASSERT(model.ObliviousTrees->TreeStartOffsets == model2.ObliviousTrees->TreeStartOffsets);
        UNIT_ASSERT(model.ObliviousTrees->CatFeatures == model2.ObliviousTrees->CatFeatures);
        UNIT_ASSERT(model.ObliviousTrees->FloatFeatures == model2.ObliviousTrees->FloatFeatures);
        UNIT_ASSERT(model.ObliviousTrees->OneHotFeatures == model2.ObliviousTrees->OneHotFeatures);
        UNIT_ASSERT(model.ObliviousTrees->CtrFeatures == model2.ObliviousTrees->CtrFeatures);
        UNIT_ASSERT(model.ObliviousTrees->LeafValues.ysize() == model2.ObliviousTrees->LeafValues.ysize());
        for (int idx = 0; idx < model.ObliviousTrees->LeafValues.ysize(); ++idx) {
            UNIT_ASSERT_DOUBLES_EQUAL(model.ObliviousTrees->LeafValues[idx], model2.ObliviousTrees->LeafValues[idx], 1e-9);
        }
    }
    Y_UNIT_TEST(TestEmptyLeafWeights) {
        TFullModel model = TrainFloatCatboostModel();
        model.ObliviousTrees.GetMutable()->LeafWeights[0].clear();
        ExportModel(model, "model.json", EModelType::Json);
        model = ReadModel("model.json", EModelType::Json);
        UNIT_ASSERT(model.ObliviousTrees->LeafWeights[0].empty());
        UNIT_ASSERT(!model.ObliviousTrees->LeafWeights[1].empty());
    }
}
