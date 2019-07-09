#include "model_test_helpers.h"

#include <library/unittest/registar.h>

using namespace std;

void DoSerializeDeserialize(const TFullModel& model) {
    TStringStream strStream;
    model.Save(&strStream);
    TFullModel deserializedModel;
    deserializedModel.Load(&strStream);
    UNIT_ASSERT_EQUAL(model, deserializedModel);
}

Y_UNIT_TEST_SUITE(TModelSerialization) {
    Y_UNIT_TEST(TestSerializeDeserializeFullModel) {
        TFullModel trainedModel = TrainFloatCatboostModel();
        DoSerializeDeserialize(trainedModel);
        trainedModel.ObliviousTrees.GetMutable()->ConvertObliviousToAsymmetric();
        DoSerializeDeserialize(trainedModel);
    }

    Y_UNIT_TEST(TestSerializeDeserializeCoreML) {
        TFullModel trainedModel = TrainFloatCatboostModel();
        TStringStream strStream;
        trainedModel.Save(&strStream);
        ExportModel(trainedModel, "model.coreml", EModelType::AppleCoreML);
        TFullModel deserializedModel = ReadModel("model.coreml", EModelType::AppleCoreML);
        UNIT_ASSERT_EQUAL(trainedModel.ObliviousTrees->LeafValues, deserializedModel.ObliviousTrees->LeafValues);
        UNIT_ASSERT_EQUAL(trainedModel.ObliviousTrees->TreeSplits, deserializedModel.ObliviousTrees->TreeSplits);
    }
}
