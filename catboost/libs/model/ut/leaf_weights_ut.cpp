#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/libs/model/model_export/model_exporter.h>

#include <util/generic/xrange.h>

#include <library/cpp/testing/unittest/registar.h>

using namespace std;
using namespace NCB;

enum EWeightsMode {
    WITH_WEIGHTS = true,
    WITHOUT_WEIGHTS = false
};

enum ETargetDimMode {
    MULTICLASS = true,
    SCALAR = false
};

enum EExportFormat {
    CBM = true,
    COREML = false
};

static TDataProviderPtr SmallFloatPool(EWeightsMode addWeights, ETargetDimMode multiclass) {
    return CreateDataProvider(
        [&] (IRawFeaturesOrderDataVisitor* visitor) {
            TDataMetaInfo metaInfo;
            metaInfo.TargetType = multiclass ? ERawTargetType::String : ERawTargetType::Float;
            metaInfo.TargetCount = 1;
            metaInfo.HasWeights = addWeights;
            metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                (ui32)3,
                TVector<ui32>{},
                TVector<ui32>{},
                TVector<ui32>{},
                TVector<TString>{});

            visitor->Start(metaInfo, 3, EObjectsOrder::Undefined, {});

            visitor->AddFloatFeature(
                0,
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{+0.5f, +1.5f, -2.5f})
            );
            visitor->AddFloatFeature(
                1,
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{+0.7f, +6.4f, +2.4f})
            );
            visitor->AddFloatFeature(
                2,
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{-2.0f, -1.0f, +6.0f})
            );

            if (multiclass) {
                visitor->AddTarget(TVector<TString>{"1", "0", "2"});
            } else {
                visitor->AddTarget(
                    MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{1.0f, 0.0f, 0.2f})
                );
            }
            if (addWeights) {
                visitor->AddWeights({1.0f, 2.0f, 0.5f});
            }
            visitor->Finish();
        }
    );
}

static TFullModel TrainModelOnPool(TDataProviderPtr pool, ETargetDimMode multiclass) {
    TFullModel model;
    TEvalResult evalResult;
    NJson::TJsonValue params;
    params.InsertValue("iterations", 5);
    if (multiclass) {
        params.InsertValue("loss_function", "MultiClass");
    }
    TrainModel(
        params,
        nullptr,
        Nothing(),
        Nothing(),
        Nothing(),
        TDataProviders{pool, {pool}},
        /*initModel*/ Nothing(),
        /*initLearnProgress*/ nullptr,
        "",
        &model,
        {&evalResult});

    return model;
}

static TFullModel SaveLoadCBM(const TFullModel& trainedModel) {
    TStringStream strStream;
    trainedModel.Save(&strStream);
    TFullModel deserializedModel;
    deserializedModel.Load(&strStream);
    return deserializedModel;
}

static TFullModel SaveLoadCoreML(const TFullModel& trainedModel) {
    TStringStream strStream;
    trainedModel.Save(&strStream);
    ExportModel(trainedModel, "model.coreml", EModelType::AppleCoreML);
    TFullModel deserializedModel = ReadModel("model.coreml", EModelType::AppleCoreML);
    return deserializedModel;
}

static void CheckWeights(const TWeights<float>& docWeights, const TFullModel& model) {
    if (model.ModelTrees->GetModelTreeData()->GetLeafWeights().empty()) {
        return;
    }

    double trueWeightSum = 0;
    for (auto i : xrange(docWeights.GetSize())) {
        trueWeightSum += docWeights[i];
    }

    const auto weights = model.ModelTrees->GetModelTreeData()->GetLeafWeights();
    const auto treeSizes = model.ModelTrees->GetModelTreeData()->GetTreeSizes();
    const int approxDimension = model.ModelTrees->GetDimensionsCount();
    auto applyData = model.ModelTrees->GetApplyData();
    auto leafOffsetPtr = applyData->TreeFirstLeafOffsets.data();
    for (size_t treeIdx = 0; treeIdx < model.GetTreeCount(); ++treeIdx) {
        double weightSumInTree = 0;
        const size_t offset = leafOffsetPtr[treeIdx] / approxDimension;
        for (size_t leafId = offset; leafId < offset + (1u << treeSizes[treeIdx]); ++leafId) {
            weightSumInTree += weights[leafId];
        }
        UNIT_ASSERT_EQUAL(trueWeightSum, weightSumInTree);
    }
}

static void RunTestWithParams(EWeightsMode addWeights, ETargetDimMode multiclass, EExportFormat exportToCBM, bool clearWeightsInModel = false) {
    TDataProviderPtr floatPool = SmallFloatPool(addWeights, multiclass);
    TFullModel trainedModel = TrainModelOnPool(floatPool, multiclass);
    if (clearWeightsInModel) {
        trainedModel.ModelTrees.GetMutable()->ClearLeafWeights();
    }
    TFullModel deserializedModel;
    if (exportToCBM) {
        deserializedModel = SaveLoadCBM(trainedModel);
    } else {
        deserializedModel = SaveLoadCoreML(trainedModel);
    }
    if (exportToCBM) {
        CheckWeights(floatPool->RawTargetData.GetWeights(), deserializedModel);
    } else {
        UNIT_ASSERT(deserializedModel.ModelTrees->GetModelTreeData()->GetLeafWeights().empty());
    }
}

Y_UNIT_TEST_SUITE(TLeafWeights) {
        Y_UNIT_TEST(TestLeafWeightsSumAfterExportNoWeights) {
            RunTestWithParams(EWeightsMode::WITHOUT_WEIGHTS, ETargetDimMode::SCALAR, EExportFormat::CBM);
        }

        Y_UNIT_TEST(TestLeafWeightsSumAfterExportWithWeights) {
            RunTestWithParams(EWeightsMode::WITH_WEIGHTS, ETargetDimMode::SCALAR, EExportFormat::CBM);
        }

        Y_UNIT_TEST(TestLeafWeightsSumAfterExportWithWeightsMulticlass) {
            RunTestWithParams(EWeightsMode::WITH_WEIGHTS, ETargetDimMode::MULTICLASS, EExportFormat::CBM);
        }

        Y_UNIT_TEST(TestEmptyLeafWeightsAfterCoreMLExportNoWeights) {
            RunTestWithParams(EWeightsMode::WITHOUT_WEIGHTS, ETargetDimMode::SCALAR, EExportFormat::COREML);
        }

        Y_UNIT_TEST(TestEmptyLeafWeightsAfterCoreMLExportWithWeights) {
            RunTestWithParams(EWeightsMode::WITH_WEIGHTS, ETargetDimMode::SCALAR, EExportFormat::COREML);
        }

        Y_UNIT_TEST(TestEmptyLeafWeightsAfteExportWithoutWeights) {
            RunTestWithParams(EWeightsMode::WITHOUT_WEIGHTS, ETargetDimMode::SCALAR, EExportFormat::CBM, /*clearWeightsInModel*/ true);
        }
}
