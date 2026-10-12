#include <catboost/libs/model/ut/lib/model_test_helpers.h>

#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/objects.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/model_export/model_exporter.h>
#include <catboost/libs/model/model_export/onnx_helpers.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/private/libs/algo/apply.h>

#include <contrib/libs/onnx/onnx/checker.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/folder/tempdir.h>
#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>

#include <limits>


using namespace NCB;


namespace {
    struct TTestColumn {
        TMaybe<TVector<float>> FloatValues;
        TMaybe<TVector<TString>> CatValues;
    };

    TTestColumn FloatColumn(TVector<float> values) {
        return TTestColumn{std::move(values), Nothing()};
    }

    TTestColumn CatColumn(TVector<TString> values) {
        return TTestColumn{Nothing(), std::move(values)};
    }

    const TVector<float> TARGET = {1.0f, 0.0f, 0.2f, 0.7f, 0.3f, 0.9f};
}


static TDataProviderPtr MakeDataProvider(const TVector<TTestColumn>& columns, const TVector<float>& target = TARGET) {
    return CreateDataProvider(
        [&] (IRawFeaturesOrderDataVisitor* visitor) {
            TVector<ui32> catFeatureIndices;
            for (auto flatFeatureIdx : xrange(columns.size())) {
                if (columns[flatFeatureIdx].CatValues) {
                    catFeatureIndices.push_back(flatFeatureIdx);
                }
            }

            TDataMetaInfo metaInfo;
            metaInfo.TargetType = ERawTargetType::Float;
            metaInfo.TargetCount = 1;
            metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                (ui32)columns.size(),
                catFeatureIndices,
                TVector<ui32>{},
                TVector<ui32>{},
                TVector<TString>{});

            visitor->Start(metaInfo, target.size(), EObjectsOrder::Undefined, {});

            for (auto flatFeatureIdx : xrange(columns.size())) {
                const auto& column = columns[flatFeatureIdx];
                if (column.CatValues) {
                    visitor->AddCatFeature(flatFeatureIdx, TConstArrayRef<TString>(*column.CatValues));
                } else {
                    visitor->AddFloatFeature(
                        flatFeatureIdx,
                        MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>(*column.FloatValues))
                    );
                }
            }

            visitor->AddTarget(MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>(target)));

            visitor->Finish();
        }
    );
}


static TFullModel TrainOnDataProvider(TDataProviderPtr learnData, NJson::TJsonValue params = NJson::TJsonValue()) {
    TDataProviders dataProviders;
    dataProviders.Learn = learnData;
    dataProviders.Test.push_back(learnData);

    TTempDir trainDir;
    params.InsertValue("iterations", 5);
    params.InsertValue("random_seed", 1);
    params.InsertValue("train_dir", trainDir.Name());
    if (!params.Has("one_hot_max_size")) {
        params.InsertValue("one_hot_max_size", 255);
    }

    TFullModel model;
    TEvalResult evalResult;
    TrainModel(
        params,
        nullptr,
        {},
        {},
        Nothing(),
        std::move(dataProviders),
        /*initModel*/ Nothing(),
        /*initLearnProgress*/ nullptr,
        "",
        &model,
        {&evalResult}
    );
    return model;
}


static TFullModel ExportToOnnxAndImport(const TFullModel& model, const THashMap<ui32, TString>* catFeaturesHashToString) {
    const TString onnxProto = ConvertTreeToOnnxProto(model, "", catFeaturesHashToString);
    UNIT_ASSERT(!onnxProto.empty());
    onnx::ModelProto proto;
    UNIT_ASSERT(proto.ParseFromString(onnxProto));
    onnx::checker::check_model(proto);
    if (!model.ModelTrees->GetOneHotFeatures().empty() && proto.graph().output(0).name() == "predictions") {
        const auto& outputShape = proto.graph().output(0).type().tensor_type().shape();
        UNIT_ASSERT_VALUES_EQUAL(outputShape.dim_size(), 2);
        UNIT_ASSERT_VALUES_EQUAL(outputShape.dim(1).dim_value(), 1);
    }
    return ReadModel(onnxProto.data(), onnxProto.size(), EModelType::Onnx);
}


static onnx::ModelProto MakeCategoricalOnnxModel() {
    const auto data = MakeDataProvider({CatColumn({"a", "a", "b", "b", "a", "b"})});
    const auto model = TrainOnDataProvider(data);
    UNIT_ASSERT(!model.ModelTrees->GetOneHotFeatures().empty());
    const auto mapping = MergeCatFeaturesHashToString(*data->ObjectsData);
    onnx::ModelProto result;
    UNIT_ASSERT(result.ParseFromString(ConvertTreeToOnnxProto(model, "", &mapping)));
    return result;
}


static onnx::NodeProto& GetNode(onnx::ModelProto& model, const TString& opType) {
    for (auto& node : *model.mutable_graph()->mutable_node()) {
        if (node.op_type() == opType) {
            return node;
        }
    }
    ythrow TCatBoostException() << "Missing node " << opType;
}


static onnx::AttributeProto& GetAttribute(onnx::NodeProto& node, const TString& name) {
    for (auto& attribute : *node.mutable_attribute()) {
        if (attribute.name() == name) {
            return attribute;
        }
    }
    ythrow TCatBoostException() << "Missing attribute " << name;
}


static void CheckPredictionsEqual(
    const TFullModel& originalModel,
    const TObjectsDataProvider& originalData,
    const TFullModel& importedModel,
    const TObjectsDataProvider& importedModelData
) {
    const auto originalPred = ApplyModelMulti(originalModel, originalData);
    const auto importedPred = ApplyModelMulti(importedModel, importedModelData);

    UNIT_ASSERT_VALUES_EQUAL(originalPred.size(), importedPred.size());
    for (auto i : xrange(originalPred.size())) {
        UNIT_ASSERT_VALUES_EQUAL(originalPred[i].size(), importedPred[i].size());
        for (auto j : xrange(originalPred[i].size())) {
            UNIT_ASSERT_DOUBLES_EQUAL(originalPred[i][j], importedPred[i][j], 1e-5);
        }
    }
}


Y_UNIT_TEST_SUITE(TestOnnxExport) {
    Y_UNIT_TEST(TestImportInvalidLabelEncoderValues) {
        const auto original = MakeCategoricalOnnxModel();
        for (float value : {0.5f, std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::quiet_NaN(), 2147483648.0f, -2147483904.0f}) {
            auto model = original;
            auto& encoder = GetNode(model, "LabelEncoder");
            GetAttribute(encoder, "values_floats").set_floats(0, value);
            TFullModel imported;
            UNIT_ASSERT_EXCEPTION_CONTAINS(
                NOnnx::ConvertOnnxToCatboostModel(model, &imported), TCatBoostException, "must be an integer"
            );
        }
    }

    Y_UNIT_TEST(TestImportAmbiguousLabelEncoder) {
        const auto original = MakeCategoricalOnnxModel();
        {
            auto model = original;
            auto& encoder = GetNode(model, "LabelEncoder");
            auto& values = GetAttribute(encoder, "values_floats");
            auto& keys = GetAttribute(encoder, "keys_strings");
            keys.add_strings("another_category");
            values.add_floats(values.floats(0));
            TFullModel imported;
            UNIT_ASSERT_EXCEPTION_CONTAINS(
                NOnnx::ConvertOnnxToCatboostModel(model, &imported), TCatBoostException, "values_floats must be unique"
            );
        }
        {
            auto model = original;
            auto& encoder = GetNode(model, "LabelEncoder");
            auto& values = GetAttribute(encoder, "values_floats");
            auto& keys = GetAttribute(encoder, "keys_strings");
            keys.add_strings(keys.strings(0));
            values.add_floats(100.0f);
            TFullModel imported;
            UNIT_ASSERT_EXCEPTION_CONTAINS(
                NOnnx::ConvertOnnxToCatboostModel(model, &imported), TCatBoostException, "keys_strings must be unique"
            );
        }
        {
            auto model = original;
            auto& encoder = GetNode(model, "LabelEncoder");
            GetAttribute(encoder, "default_float").set_f(GetAttribute(encoder, "values_floats").floats(0));
            TFullModel imported;
            UNIT_ASSERT_EXCEPTION_CONTAINS(
                NOnnx::ConvertOnnxToCatboostModel(model, &imported), TCatBoostException, "default_float must differ"
            );
        }
        {
            auto model = original;
            auto& encoder = GetNode(model, "LabelEncoder");
            GetAttribute(encoder, "default_float").set_f(std::numeric_limits<float>::quiet_NaN());
            TFullModel imported;
            UNIT_ASSERT_EXCEPTION_CONTAINS(
                NOnnx::ConvertOnnxToCatboostModel(model, &imported), TCatBoostException, "default_float must not be NaN"
            );
        }
    }

    Y_UNIT_TEST(TestImportConcatAxis) {
        auto model = MakeCategoricalOnnxModel();
        auto& axis = GetAttribute(GetNode(model, "Concat"), "axis");
        axis.set_i(0);
        TFullModel imported;
        UNIT_ASSERT_EXCEPTION_CONTAINS(
            NOnnx::ConvertOnnxToCatboostModel(model, &imported), TCatBoostException, "axis 1 or -1"
        );
        axis.set_i(-1);
        NOnnx::ConvertOnnxToCatboostModel(model, &imported);
        UNIT_ASSERT_VALUES_EQUAL(imported.GetNumCatFeatures(), 1);
    }

    Y_UNIT_TEST(TestImportNumericalSplitOnEncodedCategory) {
        auto model = MakeCategoricalOnnxModel();
        GetAttribute(GetNode(model, "TreeEnsembleRegressor"), "nodes_modes").set_strings(0, "BRANCH_GT");
        TFullModel imported;
        UNIT_ASSERT_EXCEPTION_CONTAINS(
            NOnnx::ConvertOnnxToCatboostModel(model, &imported),
            TCatBoostException, "both categorical and numerical splits"
        );
    }

    Y_UNIT_TEST(TestImportFeatureIndexOutOfRange) {
        const auto original = MakeCategoricalOnnxModel();
        for (i64 featureIdx : {-1LL, 1LL, 4294967296LL}) {
            auto model = original;
            GetAttribute(GetNode(model, "TreeEnsembleRegressor"), "nodes_featureids").set_ints(0, featureIdx);
            TFullModel imported;
            UNIT_ASSERT_EXCEPTION_CONTAINS(
                NOnnx::ConvertOnnxToCatboostModel(model, &imported), TCatBoostException, "outside the input tensor"
            );
        }
    }

    Y_UNIT_TEST(TestFloatOnlyModelExportImport) {
        const TVector<TTestColumn> columns = {
            FloatColumn({0.1f, 0.5f, 0.3f, 0.9f, 0.2f, 0.7f}),
            FloatColumn({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}),
            FloatColumn({0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f})
        };
        const auto data = MakeDataProvider(columns);
        const auto model = TrainOnDataProvider(data);
        UNIT_ASSERT(!model.ModelTrees->GetBinFeatures().empty());

        const auto importedModel = ExportToOnnxAndImport(model, nullptr);
        UNIT_ASSERT_VALUES_EQUAL(importedModel.GetTreeCount(), model.GetTreeCount());
        UNIT_ASSERT_VALUES_EQUAL(importedModel.GetNumFloatFeatures(), columns.size());
        UNIT_ASSERT_VALUES_EQUAL(importedModel.GetNumCatFeatures(), 0);

        CheckPredictionsEqual(model, *data->ObjectsData, importedModel, *data->ObjectsData);
    }

    Y_UNIT_TEST(TestCatOnlyModelExportImport) {
        const TVector<TTestColumn> columns = {
            CatColumn({"a", "a", "b", "b", "a", "b"}),
            CatColumn({"d", "c", "d", "c", "c", "d"}),
            CatColumn({"e", "f", "f", "e", "e", "f"})
        };
        const auto data = MakeDataProvider(columns);
        const auto model = TrainOnDataProvider(data);
        UNIT_ASSERT(!model.ModelTrees->GetOneHotFeatures().empty());

        const auto catFeaturesHashToString = MergeCatFeaturesHashToString(*data->ObjectsData);
        const auto importedModel = ExportToOnnxAndImport(model, &catFeaturesHashToString);
        UNIT_ASSERT_VALUES_EQUAL(importedModel.GetTreeCount(), model.GetTreeCount());
        UNIT_ASSERT_VALUES_EQUAL(importedModel.GetNumFloatFeatures(), 0);
        UNIT_ASSERT_VALUES_EQUAL(
            importedModel.GetNumCatFeatures(),
            model.ModelTrees->GetOneHotFeatures().size()
        );

        // all categorical features are used in the model, so the layout is the same
        UNIT_ASSERT_VALUES_EQUAL(importedModel.GetNumCatFeatures(), columns.size());
        CheckPredictionsEqual(model, *data->ObjectsData, importedModel, *data->ObjectsData);
    }

    Y_UNIT_TEST(TestMixedFeaturesModelExportImport) {
        const TVector<TTestColumn> columns = {
            CatColumn({"a", "a", "b", "b", "a", "b"}),
            FloatColumn({0.1f, 0.5f, 0.3f, 0.9f, 0.2f, 0.7f}),
            CatColumn({"d", "c", "d", "c", "c", "d"}),
            FloatColumn({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f})
        };
        const auto data = MakeDataProvider(columns);
        const auto model = TrainOnDataProvider(data);
        UNIT_ASSERT_VALUES_EQUAL(model.ModelTrees->GetOneHotFeatures().size(), 2);
        UNIT_ASSERT(!model.ModelTrees->GetFloatFeatures().empty());

        const auto catFeaturesHashToString = MergeCatFeaturesHashToString(*data->ObjectsData);
        const auto importedModel = ExportToOnnxAndImport(model, &catFeaturesHashToString);
        UNIT_ASSERT_VALUES_EQUAL(importedModel.GetTreeCount(), model.GetTreeCount());
        UNIT_ASSERT_VALUES_EQUAL(importedModel.GetNumFloatFeatures(), 2);
        UNIT_ASSERT_VALUES_EQUAL(importedModel.GetNumCatFeatures(), 2);

        // imported model expects categorical features to be placed after float features
        const auto importedModelData = MakeDataProvider({columns[1], columns[3], columns[0], columns[2]});
        CheckPredictionsEqual(model, *data->ObjectsData, importedModel, *importedModelData->ObjectsData);
    }

    Y_UNIT_TEST(TestUnusedCatFeatureModelExportImport) {
        // A categorical feature that is present in the dataset but not used in the model
        // must not shift ONNX input indices of the used categorical features.
        const TVector<TTestColumn> columns = {
            CatColumn({"a", "a", "b", "b", "a", "b"}),
            CatColumn({"d", "c", "d", "c", "c", "d"}),
            CatColumn({"e", "f", "f", "e", "e", "f"})
        };
        const auto data = MakeDataProvider(columns);

        NJson::TJsonValue params;
        NJson::TJsonValue ignoredFeatures(NJson::EJsonValueType::JSON_ARRAY);
        ignoredFeatures.AppendValue(0);
        params.InsertValue("ignored_features", ignoredFeatures);
        const auto model = TrainOnDataProvider(data, params);
        UNIT_ASSERT_VALUES_EQUAL(model.ModelTrees->GetOneHotFeatures().size(), 2);

        const auto catFeaturesHashToString = MergeCatFeaturesHashToString(*data->ObjectsData);
        const auto importedModel = ExportToOnnxAndImport(model, &catFeaturesHashToString);
        UNIT_ASSERT_VALUES_EQUAL(importedModel.GetNumFloatFeatures(), 0);
        UNIT_ASSERT_VALUES_EQUAL(importedModel.GetNumCatFeatures(), 2);

        // imported model contains only the used categorical features (1 and 2)
        const auto importedModelData = MakeDataProvider({columns[1], columns[2]});
        CheckPredictionsEqual(model, *data->ObjectsData, importedModel, *importedModelData->ObjectsData);
    }

    Y_UNIT_TEST(TestExportAfterDroppingUnusedFeatures) {
        const TVector<TTestColumn> columns = {
            CatColumn({"ignored", "ignored", "ignored", "ignored", "ignored", "ignored"}),
            FloatColumn({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}),
            CatColumn({"a", "a", "b", "b", "a", "b"}),
            FloatColumn({0.1f, 0.5f, 0.3f, 0.9f, 0.2f, 0.7f})
        };
        const auto data = MakeDataProvider(columns);
        auto model = TrainOnDataProvider(data);
        model.ModelTrees.GetMutable()->DropUnusedFeatures();
        UNIT_ASSERT_VALUES_EQUAL(model.ModelTrees->GetCatFeatures().size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(model.ModelTrees->GetCatFeatures()[0].Position.Index, 1);
        UNIT_ASSERT_VALUES_EQUAL(model.ModelTrees->GetFloatFeatures().size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(model.ModelTrees->GetFloatFeatures()[0].Position.Index, 1);

        const auto mapping = MergeCatFeaturesHashToString(*data->ObjectsData);
        const auto importedModel = ExportToOnnxAndImport(model, &mapping);
        const auto importedModelData = MakeDataProvider({columns[3], columns[2]});
        CheckPredictionsEqual(model, *data->ObjectsData, importedModel, *importedModelData->ObjectsData);
    }

    Y_UNIT_TEST(TestExportWithoutCatFeaturesHashToString) {
        const auto data = MakeDataProvider({
            CatColumn({"a", "a", "b", "b", "a", "b"}),
            FloatColumn({0.1f, 0.5f, 0.3f, 0.9f, 0.2f, 0.7f})
        });
        const auto model = TrainOnDataProvider(data);
        UNIT_ASSERT(!model.ModelTrees->GetOneHotFeatures().empty());

        UNIT_ASSERT_EXCEPTION_CONTAINS(
            ConvertTreeToOnnxProto(model, "", nullptr),
            TCatBoostException,
            "hash to string mapping is required"
        );

        const THashMap<ui32, TString> emptyMapping;
        UNIT_ASSERT_EXCEPTION_CONTAINS(
            ConvertTreeToOnnxProto(model, "", &emptyMapping),
            TCatBoostException,
            "Cannot find string value"
        );
    }

    Y_UNIT_TEST(TestExportWithCtrFeatures) {
        const auto data = MakeDataProvider({
            CatColumn({"a", "a", "b", "b", "a", "b"}),
            FloatColumn({0.1f, 0.5f, 0.3f, 0.9f, 0.2f, 0.7f})
        });
        NJson::TJsonValue params;
        params.InsertValue("one_hot_max_size", 0);
        const auto model = TrainOnDataProvider(data, params);
        UNIT_ASSERT(model.ModelTrees->GetOneHotFeatures().empty());
        UNIT_ASSERT(!model.ModelTrees->GetCatFeatures().empty());
        UNIT_ASSERT(!model.ModelTrees->GetCtrFeatures().empty() || !model.ModelTrees->GetEstimatedFeatures().empty());

        const auto catFeaturesHashToString = MergeCatFeaturesHashToString(*data->ObjectsData);
        UNIT_ASSERT_EXCEPTION_CONTAINS(
            ConvertTreeToOnnxProto(model, "", &catFeaturesHashToString),
            TCatBoostException,
            "one_hot_max_size"
        );
    }
}
