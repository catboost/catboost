#include <catboost/libs/model/ut/lib/model_test_helpers.h>

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/model/hash.h>
#include <catboost/libs/model/model_export/json_model_helpers.h>
#include <catboost/libs/model/model_export/model_exporter.h>
#include <catboost/libs/model/static_ctr_provider.h>

#include <library/cpp/json/json_reader.h>
#include <library/cpp/json/json_writer.h>
#include <library/cpp/testing/unittest/registar.h>

using namespace std;
using namespace NCB;

namespace {
    void CheckMeanCtrJsonRoundTrip(ECtrType ctrType) {
        // Construct a categorical model directly. No trainer is needed to
        // exercise the importer, CTR lookup, or original-category inference.
        NJson::TJsonValue json;
        UNIT_ASSERT(NJson::ReadJsonTree(R"({
            "features_info": {
                "categorical_features": [{"feature_index":0,"flat_feature_index":0,"feature_id":"kind"}],
                "ctrs": [{
                    "elements":[{"cat_feature_index":0,"combination_element":"cat_feature_value"}],
                    "prior_numerator":0,"prior_denomerator":1,"shift":0,"scale":1,
                    "target_border_idx":0,"borders":[-0.3,0.1,0.6]
                }]
            },
            "oblivious_trees": [
                {"splits":[{"split_type":"OnlineCtr","border":-0.3,"ctr_target_border_idx":0,"split_index":0}],
                 "leaf_values":[0,1],"leaf_weights":[1,1]},
                {"splits":[{"split_type":"OnlineCtr","border":0.1,"ctr_target_border_idx":0,"split_index":1}],
                 "leaf_values":[0,2],"leaf_weights":[1,1]},
                {"splits":[{"split_type":"OnlineCtr","border":0.6,"ctr_target_border_idx":0,"split_index":2}],
                 "leaf_values":[0,4],"leaf_weights":[1,1]}
            ],
            "scale_and_bias":[1,[0]]
        })", &json));
        TModelCtrBase ctrBase;
        ctrBase.Projection.CatFeatures = {0};
        ctrBase.CtrType = ctrType;
        const TString identifier = ModelCtrBaseToStr(ctrBase);
        auto& descriptor = json["features_info"]["ctrs"][0];
        descriptor["identifier"] = identifier;
        descriptor["ctr_type"] = ToString(ctrType);

        // The integer JSON value checks backwards compatibility. The two
        // fractional values used to be read as zero by GetInteger().
        TVector<TStringBuf> categories = {"fractional", "small", "integer"};
        TVector<NJson::TJsonValue> sums = {1.5, 0.25, 2};
        TVector<double> expectedPredictions = {7, 3, 7};
        if (ctrType == ECtrType::FloatTargetMeanValue) {
            categories.push_back("negative");
            sums.push_back(-1.5);
            expectedPredictions.push_back(0);
        }
        auto& table = json["ctr_data"][identifier];
        table["hash_stride"] = 3;
        table["counter_denominator"] = 0;
        TVector<ui64> hashes;
        TVector<TVector<TStringBuf>> features;
        for (size_t index = 0; index < categories.size(); ++index) {
            const ui64 hash = CalcHash(0, static_cast<ui64>(static_cast<int>(CalcCatFeatureHash(categories[index]))));
            hashes.push_back(hash);
            table["hash_map"].AppendValue(ToString(hash));
            table["hash_map"].AppendValue(sums[index]);
            table["hash_map"].AppendValue(1);
            features.push_back({categories[index]});
        }
        features.push_back({"unseen"});
        expectedPredictions.push_back(1);

        for (size_t roundTrip = 0; roundTrip < 2; ++roundTrip) {
            TFullModel model;
            ConvertJsonToCatboostModel(json, &model);
            const auto* provider = dynamic_cast<const TStaticCtrProvider*>(model.CtrProvider.Get());
            UNIT_ASSERT(provider);
            const auto& importedTable = provider->CtrData.LearnCtrs.at(ctrBase);
            const auto history = importedTable.GetTypedArrayRefForBlobData<TCtrMeanHistory>();
            const auto indexViewer = importedTable.GetIndexHashViewer();
            for (size_t index = 0; index < hashes.size(); ++index) {
                const auto bucket = indexViewer.GetIndex(hashes[index]);
                UNIT_ASSERT(bucket != NCatboost::TDenseIndexHashView::NotFoundIndex);
                UNIT_ASSERT_DOUBLES_EQUAL(history[bucket].Sum, sums[index].GetDouble(), 0.0);
                UNIT_ASSERT_VALUES_EQUAL(history[bucket].Count, 1);
            }
            TVector<double> predictions(features.size());
            model.Calc({}, features, predictions);
            UNIT_ASSERT_VALUES_EQUAL(predictions, expectedPredictions);
            // Parse the serialized export too, so integer/double JSON numeric
            // representations are exercised across both directions.
            const auto exported = ConvertModelToJson(model);
            UNIT_ASSERT(NJson::ReadJsonTree(NJson::WriteJson(exported), &json));
        }
    }
}

Y_UNIT_TEST_SUITE(TJsonModelExport) {
    Y_UNIT_TEST(TestFloatTargetMeanCtrJsonRoundTrip) {
        CheckMeanCtrJsonRoundTrip(ECtrType::FloatTargetMeanValue);
    }

    Y_UNIT_TEST(TestBinarizedTargetMeanCtrJsonRoundTrip) {
        CheckMeanCtrJsonRoundTrip(ECtrType::BinarizedTargetMeanValue);
    }

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
