#include <catboost/libs/data_new/data_provider_builders.h>
#include <catboost/libs/train_lib/train_model.h>

#include <util/generic/xrange.h>

#include <library/unittest/registar.h>

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
            metaInfo.HasTarget = true;
            metaInfo.HasWeights = addWeights;
            metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                (ui32)3,
                TVector<ui32>{},
                TVector<TString>{},
                nullptr);

            visitor->Start(metaInfo, 3, EObjectsOrder::Undefined, {});

            visitor->AddFloatFeature(
                0,
                TMaybeOwningConstArrayHolder<float>::CreateOwning(TVector<float>{+0.5f, +1.5f, -2.5f})
            );
            visitor->AddFloatFeature(
                1,
                TMaybeOwningConstArrayHolder<float>::CreateOwning(TVector<float>{+0.7f, +6.4f, +2.4f})
            );
            visitor->AddFloatFeature(
                2,
                TMaybeOwningConstArrayHolder<float>::CreateOwning(TVector<float>{-2.0f, -1.0f, +6.0f})
            );

            if (multiclass) {
                visitor->AddTarget(TVector<TString>{"1", "0", "2"});
            } else {
                visitor->AddTarget(TVector<float>{1.0f, 0.0f, 0.2f});
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
        TDataProviders{pool, {pool}},
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

static void CheckWeights(const TWeights<float>& docWeights, const TVector<TVector<double>>& leafWeights) {
    double trueWeightSum = 0;
    for (auto i : xrange(docWeights.GetSize())) {
        trueWeightSum += docWeights[i];
    }
    for (const auto& leafWeightsInTree : leafWeights) {
        double weightSumInTree = 0;
        for (const auto& weight : leafWeightsInTree) {
            weightSumInTree += weight;
        }
        UNIT_ASSERT_EQUAL(trueWeightSum, weightSumInTree);
    }
}

static void RunTestWithParams(EWeightsMode addWeights, ETargetDimMode multiclass, EExportFormat exportToCBM, bool clearWeightsInModel = false) {
    TDataProviderPtr floatPool = SmallFloatPool(addWeights, multiclass);
    TFullModel trainedModel = TrainModelOnPool(floatPool, multiclass);
    if (clearWeightsInModel) {
        trainedModel.ObliviousTrees.LeafWeights.clear();
    }
    TFullModel deserializedModel;
    if (exportToCBM) {
        deserializedModel = SaveLoadCBM(trainedModel);
    } else {
        deserializedModel = SaveLoadCoreML(trainedModel);
    }
    if (exportToCBM) {
        CheckWeights(floatPool->RawTargetData.GetWeights(), deserializedModel.ObliviousTrees.LeafWeights);
    } else {
        UNIT_ASSERT(deserializedModel.ObliviousTrees.LeafWeights.empty());
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
