#include <catboost/libs/model/ut/lib/model_test_helpers.h>

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/model/model_export/model_exporter.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/objects.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/private/libs/algo/apply.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/file.h>
#include <util/folder/tempdir.h>

using namespace NCB;

static void CheckPredictionsEqual(
    const TFullModel& originalModel,
    const TFullModel& importedModel,
    TDataProviderPtr testData
) {
    auto originalPred = ApplyModelMulti(originalModel, *testData->ObjectsData);
    auto importedPred = ApplyModelMulti(importedModel, *testData->ObjectsData);

    UNIT_ASSERT_VALUES_EQUAL(originalPred.size(), importedPred.size());
    for (size_t i = 0; i < originalPred.size(); ++i) {
        UNIT_ASSERT_VALUES_EQUAL(originalPred[i].size(), importedPred[i].size());
        for (size_t j = 0; j < originalPred[i].size(); ++j) {
            UNIT_ASSERT_DOUBLES_EQUAL(originalPred[i][j], importedPred[i][j], 1e-5);
        }
    }
}

Y_UNIT_TEST_SUITE(TestOnnxExport) {
    Y_UNIT_TEST(TestFloatOnlyModelExportImport) {
        // Use a trained float model — just verify export/import doesn't crash
        auto model = TrainFloatCatboostModel(5, 123);

        TString onnxProto = NCB::ConvertTreeToOnnxProto(model);
        UNIT_ASSERT(!onnxProto.empty());

        TTempDir tempDir;
        TString onnxPath = tempDir.Name() + "/model.onnx";
        {
            TFileOutput out(onnxPath);
            out.Write(onnxProto);
        }

        TFullModel importedModel = ReadModel(onnxPath, EModelType::Onnx);
        UNIT_ASSERT(importedModel.GetTreeCount() == model.GetTreeCount());
    }

    Y_UNIT_TEST(TestCatOnlyModelExportImport) {
        TDataProviders dataProviders;
        dataProviders.Learn = CreateDataProvider(
            [&] (IRawFeaturesOrderDataVisitor* visitor) {
                TDataMetaInfo metaInfo;
                metaInfo.TargetType = ERawTargetType::Float;
                metaInfo.TargetCount = 1;
                metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                    (ui32)3,
                    TVector<ui32>{0, 1, 2},
                    TVector<ui32>{},
                    TVector<ui32>{},
                    TVector<TString>{});

                visitor->Start(metaInfo, 3, EObjectsOrder::Undefined, {});
                visitor->AddCatFeature(0, TConstArrayRef<TStringBuf>{"a", "a", "b"});
                visitor->AddCatFeature(1, TConstArrayRef<TStringBuf>{"d", "c", "d"});
                visitor->AddCatFeature(2, TConstArrayRef<TStringBuf>{"e", "f", "f"});
                visitor->AddTarget(
                    MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{1.0f, 0.0f, 0.2f})
                );
                visitor->Finish();
            }
        );
        dataProviders.Test.push_back(dataProviders.Learn);

        THashMap<ui32, TString> catFeaturesHashToString = MergeCatFeaturesHashToString(*dataProviders.Learn->ObjectsData);

        TFullModel model;
        TEvalResult evalResult;
        NJson::TJsonValue params;
        params.InsertValue("iterations", 5);
        params.InsertValue("random_seed", 1);
        TTempDir trainDir;
        params.InsertValue("train_dir", trainDir.Name());
        TrainModel(
            params,
            nullptr,
            {},
            {},
            Nothing(),
            std::move(dataProviders),
            Nothing(),
            nullptr,
            "",
            &model,
            {&evalResult}
        );

        TString onnxProto = NCB::ConvertTreeToOnnxProto(model, "", &catFeaturesHashToString);
        UNIT_ASSERT(!onnxProto.empty());

        TTempDir tempDir;
        TString onnxPath = tempDir.Name() + "/model.onnx";
        {
            TFileOutput out(onnxPath);
            out.Write(onnxProto);
        }

        TFullModel importedModel = ReadModel(onnxPath, EModelType::Onnx);

        TDataProviders testDataProviders;
        testDataProviders.Learn = CreateDataProvider(
            [&] (IRawFeaturesOrderDataVisitor* visitor) {
                TDataMetaInfo metaInfo;
                metaInfo.TargetType = ERawTargetType::Float;
                metaInfo.TargetCount = 1;
                metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                    (ui32)3,
                    TVector<ui32>{0, 1, 2},
                    TVector<ui32>{},
                    TVector<ui32>{},
                    TVector<TString>{});

                visitor->Start(metaInfo, 3, EObjectsOrder::Undefined, {});
                visitor->AddCatFeature(0, TConstArrayRef<TStringBuf>{"a", "a", "b"});
                visitor->AddCatFeature(1, TConstArrayRef<TStringBuf>{"d", "c", "d"});
                visitor->AddCatFeature(2, TConstArrayRef<TStringBuf>{"e", "f", "f"});
                visitor->AddTarget(
                    MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{1.0f, 0.0f, 0.2f})
                );
                visitor->Finish();
            }
        );

        CheckPredictionsEqual(model, importedModel, testDataProviders.Learn);
    }
}
