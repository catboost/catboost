#include <catboost/libs/model/ut/lib/model_test_helpers.h>

#include <catboost/libs/model/model_export/model_exporter.h>

#include <library/unittest/registar.h>

using namespace std;
using namespace NCB;

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
        trainedModel.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        DoSerializeDeserialize(trainedModel);
    }

    Y_UNIT_TEST(TestSerializeDeserializeCoreML) {
        TFullModel trainedModel = TrainFloatCatboostModel();
        TStringStream strStream;
        trainedModel.Save(&strStream);
        ExportModel(trainedModel, "model.coreml", EModelType::AppleCoreML);
        TFullModel deserializedModel = ReadModel("model.coreml", EModelType::AppleCoreML);
        UNIT_ASSERT_EQUAL(trainedModel.ModelTrees->GetLeafValues(), deserializedModel.ModelTrees->GetLeafValues());
        UNIT_ASSERT_EQUAL(trainedModel.ModelTrees->GetTreeSplits(), deserializedModel.ModelTrees->GetTreeSplits());
    }
}
