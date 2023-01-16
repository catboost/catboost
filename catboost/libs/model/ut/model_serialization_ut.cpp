#include <catboost/libs/model/ut/lib/model_test_helpers.h>
#include <catboost/libs/model/features.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/model_build_helper.h>
#include <catboost/libs/model/model_export/json_model_helpers.h>
#include <catboost/libs/model/model_export/model_exporter.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/private/libs/algo/apply.h>
#include <catboost/private/libs/algo/learn_context.h>

#include <library/cpp/json/json_writer.h>

#include <library/cpp/testing/unittest/registar.h>

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

    Y_UNIT_TEST(TestSerializeDeserializeFullModelWithScaleAndBias) {
        TFullModel trainedModel = TrainFloatCatboostModel();
        trainedModel.SetScaleAndBias({0.5, {0.125}});
        DoSerializeDeserialize(trainedModel);
        trainedModel.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        DoSerializeDeserialize(trainedModel);
    }

    Y_UNIT_TEST(TestSerializeDeserializeFullModelNonOwning) {
        auto check = [&](const TFullModel& model) {
            TStringStream strStream;
            model.Save(&strStream);
            TFullModel deserializedModel;
            deserializedModel.InitNonOwning(strStream.Data(), strStream.Size());
            UNIT_ASSERT_EQUAL(model, deserializedModel);
        };
        check(TrainFloatCatboostModel());
        check(TrainCatOnlyNoOneHotModel());
    }

    Y_UNIT_TEST(TestSerializeDeserializeCoreML) {
        TFullModel trainedModel = TrainFloatCatboostModel();
        TStringStream strStream;
        trainedModel.Save(&strStream);
        ExportModel(trainedModel, "model.coreml", EModelType::AppleCoreML);
        TFullModel deserializedModel = ReadModel("model.coreml", EModelType::AppleCoreML);
        UNIT_ASSERT_EQUAL(trainedModel.ModelTrees->GetModelTreeData()->GetLeafValues(), deserializedModel.ModelTrees->GetModelTreeData()->GetLeafValues());
        UNIT_ASSERT_EQUAL(trainedModel.ModelTrees->GetModelTreeData()->GetTreeSplits(), deserializedModel.ModelTrees->GetModelTreeData()->GetTreeSplits());
    }

    Y_UNIT_TEST(TestNonSymmetricJsonApply) {
        auto pool = GetAdultPool();
        TDataProviders dataProviders;
        dataProviders.Learn = pool;
        dataProviders.Test.push_back(pool);

        THolder<TLearnProgress> learnProgress;
        NJson::TJsonValue params;
        params.InsertValue("learning_rate", 0.01);
        params.InsertValue("iterations", 100);
        params.InsertValue("random_seed", 1);
        TFullModel trainedModel;
        TEvalResult evalResult;

        TrainModel(
            params,
            nullptr,
            Nothing(),
            Nothing(),
            Nothing(),
            dataProviders,
            Nothing(),
            &learnProgress,
            "",
            &trainedModel,
            {&evalResult});

        ExportModel(trainedModel, "oblivious_model.json", EModelType::Json);
        trainedModel.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        ExportModel(trainedModel, "nonsymmetric_model.json", EModelType::Json);

        TFullModel obliviousModel = ReadModel("oblivious_model.json", EModelType::Json);
        TFullModel nonSymmetricModel = ReadModel("nonsymmetric_model.json", EModelType::Json);
        auto result1 = ApplyModelMulti(obliviousModel, *pool);
        auto result2 = ApplyModelMulti(nonSymmetricModel, *pool);
        UNIT_ASSERT_EQUAL(result1, result2);
    }

    static TString RemoveWhitespacesAndNewLines(const TString& str) {
        TStringBuilder out;
        for (char c : str) {
            if (!EqualToOneOf(c, ' ', '\n')) {
                out << c;
            }
        }
        return out;
    }

    Y_UNIT_TEST(TestNonSymmetricJsonFormat) {
        TFullModel model;
        model.UpdateDynamicData();
        TFloatFeature f0(false, 0, 0, {0.5, 1.5, 2.5});
        TFloatFeature f1(false, 1, 1, {5, 10, 20});
        TFloatFeature f2(false, 2, 2, {5, 15, 25, 35});
        TNonSymmetricTreeModelBuilder builder({f0, f1, f2}, {}, {}, {}, 1);
        {
            auto head = MakeHolder<TNonSymmetricTreeNode>();
            head->SplitCondition = TModelSplit(TFloatSplit(0, 0.5));
            {
                auto left = MakeHolder<TNonSymmetricTreeNode>();
                left->Value = 1.0;
                left->NodeWeight = 10;
                head->Left = std::move(left);
            }
            {
                auto right = MakeHolder<TNonSymmetricTreeNode>();
                right->Value = 2.0;
                right->NodeWeight = 20;
                head->Right = std::move(right);
            }
            builder.AddTree(std::move(head));
        }
        {
            auto head = MakeHolder<TNonSymmetricTreeNode>();
            head->SplitCondition = TModelSplit(TFloatSplit(1, 10));
            {
                auto left = MakeHolder<TNonSymmetricTreeNode>();
                left->SplitCondition = TModelSplit(TFloatSplit(2, 25));
                {
                    auto leftLeft = MakeHolder<TNonSymmetricTreeNode>();
                    leftLeft->Value = 3.0;
                    leftLeft->NodeWeight = 30;
                    left->Left = std::move(leftLeft);
                }
                {
                    auto leftRight = MakeHolder<TNonSymmetricTreeNode>();
                    leftRight->Value = 4.0;
                    leftRight->NodeWeight = 40;
                    left->Right = std::move(leftRight);
                }
                head->Left = std::move(left);
            }
            {
                auto right = MakeHolder<TNonSymmetricTreeNode>();
                right->Value = 5.0;
                right->NodeWeight = 50;
                head->Right = std::move(right);
            }
            builder.AddTree(std::move(head));
        }
        builder.Build(model.ModelTrees.GetMutable());
        const auto json = ConvertModelToJson(model, nullptr, nullptr);
        const TString jsonTreesStr = NJson::WriteJson(&json["trees"], false, true);
        const TString expectedJsonTrees =
            R"([
                {
                "left":
                    {
                    "value":1,
                    "weight":10
                    },
                "right":
                    {
                    "value":2,
                    "weight":20
                    },
                "split":
                    {
                    "border":0.5,
                    "float_feature_index":0,
                    "split_index":0,
                    "split_type":"FloatFeature"
                    }
                },
                {
                "left":
                    {
                    "left":
                        {
                        "value":3,
                        "weight":30
                        },
                    "right":
                        {
                        "value":4,
                        "weight":40
                        },
                    "split":
                        {
                        "border":25,
                        "float_feature_index":2,
                        "split_index":2,
                        "split_type":"FloatFeature"
                        }
                    },
                "right":
                    {
                    "value":5,
                    "weight":50
                    },
                "split":
                    {
                    "border":10,
                    "float_feature_index":1,
                    "split_index":1,
                    "split_type":"FloatFeature"
                    }
                }
            ])";
        UNIT_ASSERT_EQUAL(jsonTreesStr, RemoveWhitespacesAndNewLines(expectedJsonTrees));
    }

    Y_UNIT_TEST(TestNonSymmetricMultiJsonFormat) {
        TFullModel model;
        model.UpdateDynamicData();
        TFloatFeature f0(false, 0, 0, {0.5, 1.5, 2.5});
        TNonSymmetricTreeModelBuilder builder({f0}, {}, {}, {}, 2);
        {
            auto head = MakeHolder<TNonSymmetricTreeNode>();
            head->SplitCondition = TModelSplit(TFloatSplit(0, 0.5));
            {
                auto left = MakeHolder<TNonSymmetricTreeNode>();
                left->Value = TVector<double>{1.0, 2.0};
                left->NodeWeight = 10;
                head->Left = std::move(left);
            }
            {
                auto right = MakeHolder<TNonSymmetricTreeNode>();
                right->Value = TVector<double>{3.0, 4.0};
                right->NodeWeight = 20;
                head->Right = std::move(right);
            }
            builder.AddTree(std::move(head));
        }
        builder.Build(model.ModelTrees.GetMutable());
        const auto json = ConvertModelToJson(model, nullptr, nullptr);
        const TString jsonTreesStr = NJson::WriteJson(&json["trees"], false, true);
        const TString expectedJsonTrees =
            R"([
                {
                "left":
                    {
                    "value":[1, 2],
                    "weight":10
                    },
                "right":
                    {
                    "value":[3,4],
                    "weight":20
                    },
                "split":
                    {
                    "border":0.5,
                    "float_feature_index":0,
                    "split_index":0,
                    "split_type":"FloatFeature"
                    }
                }
            ])";
        assert(jsonTreesStr == RemoveWhitespacesAndNewLines(expectedJsonTrees));
    }
}
