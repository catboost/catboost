#include <catboost/libs/model/ut/lib/model_test_helpers.h>

#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/objects.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/model_export/model_exporter.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/private/libs/algo/apply.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/folder/tempdir.h>
#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>


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
    return ReadModel(onnxProto.data(), onnxProto.size(), EModelType::Onnx);
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
