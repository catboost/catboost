#include <catboost/libs/model/ut/lib/model_test_helpers.h>

#include <catboost/libs/model/model_export/model_exporter.h>

#include <library/cpp/json/json_reader.h>
#include <library/cpp/json/json_writer.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/file.h>

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
    Y_UNIT_TEST(TestMalformedLeafValuesCountRejected) {
        // A JSON model whose oblivious tree has a leaf_values count that is not a
        // multiple of 2^depth must be rejected instead of silently truncating the
        // approx dimension (which later causes out-of-bounds leaf reads at apply time).
        TFullModel model = TrainFloatCatboostModel();
        ExportModel(model, "model.json", EModelType::Json);

        NJson::TJsonValue jsonModel;
        {
            TIFStream in("model.json");
            NJson::ReadJsonTree(&in, &jsonModel, /*throwOnError*/ true);
        }
        NJson::TJsonValue& trees = jsonModel["oblivious_trees"];
        UNIT_ASSERT(trees.GetArray().size() > 0);

        // Drop one leaf value from the first tree so the count is no longer a
        // multiple of 2^depth.
        NJson::TJsonValue& leafValues = trees[0]["leaf_values"];
        UNIT_ASSERT(leafValues.GetArray().size() > 1);
        NJson::TJsonValue truncated(NJson::JSON_ARRAY);
        const auto& src = leafValues.GetArray();
        for (size_t i = 0; i + 1 < src.size(); ++i) {
            truncated.AppendValue(src[i]);
        }
        leafValues = truncated;

        {
            TOFStream out("model_malformed.json");
            NJson::WriteJson(&out, &jsonModel);
        }

        UNIT_ASSERT_EXCEPTION(
            ReadModel("model_malformed.json", EModelType::Json),
            TCatBoostException);
    }
}
