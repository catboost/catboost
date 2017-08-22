#include "model_test_helpers.h"

#include <library/unittest/registar.h>

using namespace std;

SIMPLE_UNIT_TEST_SUITE(TModelSerialization) {
    SIMPLE_UNIT_TEST(TestSerializeDeserializeFullModel) {
        TFullModel trainedModel = TrainFloatCatboostModel();
        TStringStream strStream;
        trainedModel.Save(&strStream);
        TFullModel deserializedModel;
        deserializedModel.Load(&strStream);
        UNIT_ASSERT_EQUAL(trainedModel, deserializedModel);
    }
}
