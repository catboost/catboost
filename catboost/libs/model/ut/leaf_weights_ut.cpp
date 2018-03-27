#include <catboost/libs/train_lib/train_model.h>

#include <library/unittest/registar.h>

using namespace std;

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

static TPool SmallFloatPool(EWeightsMode addWeights, ETargetDimMode multiclass) {
    TPool pool;
    pool.Docs.Resize(/*doc count*/3, /*factors count*/ 3, /*baseline dimension*/ 0, /*has queryId*/ false, /*has subgroupId*/ false);
    pool.Docs.Factors[0] = {+0.5f, +1.5f, -2.5f};
    pool.Docs.Factors[1] = {+0.7f, +6.4f, +2.4f};
    pool.Docs.Factors[2] = {-2.0f, -1.0f, +6.0f};
    if (multiclass) {
        pool.Docs.Target = {1, 0, 2};
    } else {
        pool.Docs.Target = {1.0f, 0.0f, 0.2f};
    }
    if (addWeights) {
        pool.Docs.Weight = {1.0f, 2.0f, 0.5f};
        pool.MetaInfo.HasWeights = true;
    }
    return pool;
}

static TFullModel TrainModelOnPool(TPool* pool, ETargetDimMode multiclass) {
    TFullModel model;
    TEvalResult evalResult;
    NJson::TJsonValue params;
    params.InsertValue("iterations", 5);
    if (multiclass) {
        params.InsertValue("loss_function", "MultiClass");
    }
    TrainModel(params, Nothing(), Nothing(), *pool, false, *pool, "", &model, &evalResult);

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

static void CheckWeights(const TVector<float>& docWeights, const TVector<TVector<double>>& leafWeights) {
    double trueWeightSum = 0;
    for (auto weight : docWeights) {
        trueWeightSum += weight;
    }
    for (const auto& leafWeightsInTree : leafWeights) {
        double weightSumInTree = 0;
        for (const auto& weight : leafWeightsInTree) {
            weightSumInTree += weight;
        }
        UNIT_ASSERT_EQUAL(trueWeightSum, weightSumInTree);
    }
}

static void RunTestWithParams(EWeightsMode addWeights, ETargetDimMode multiclass, EExportFormat exportToCBM) {
    TPool floatPool = SmallFloatPool(addWeights, multiclass);
    TFullModel trainedModel = TrainModelOnPool(&floatPool, multiclass);
    TFullModel deserializedModel;
    if (exportToCBM) {
        deserializedModel = SaveLoadCBM(trainedModel);
    } else {
        deserializedModel = SaveLoadCoreML(trainedModel);
    }
    if (exportToCBM) {
        CheckWeights(floatPool.Docs.Weight, deserializedModel.ObliviousTrees.LeafWeights);
    } else {
        UNIT_ASSERT(deserializedModel.ObliviousTrees.LeafWeights.empty());
    }
}

SIMPLE_UNIT_TEST_SUITE(TLeafWeights) {
        SIMPLE_UNIT_TEST(TestLeafWeightsSumAfterExportNoWeights) {
            RunTestWithParams(EWeightsMode::WITHOUT_WEIGHTS, ETargetDimMode::SCALAR, EExportFormat::CBM);
        }

        SIMPLE_UNIT_TEST(TestLeafWeightsSumAfterExportWithWeights) {
            RunTestWithParams(EWeightsMode::WITH_WEIGHTS, ETargetDimMode::SCALAR, EExportFormat::CBM);
        }

        SIMPLE_UNIT_TEST(TestLeafWeightsSumAfterExportWithWeightsMulticlass) {
            RunTestWithParams(EWeightsMode::WITH_WEIGHTS, ETargetDimMode::MULTICLASS, EExportFormat::CBM);
        }

        SIMPLE_UNIT_TEST(TestEmptyLeafWeightsAfterCoreMLExportNoWeights) {
            RunTestWithParams(EWeightsMode::WITHOUT_WEIGHTS, ETargetDimMode::SCALAR, EExportFormat::COREML);
        }

        SIMPLE_UNIT_TEST(TestEmptyLeafWeightsAfterCoreMLExportWithWeights) {
            RunTestWithParams(EWeightsMode::WITH_WEIGHTS, ETargetDimMode::SCALAR, EExportFormat::COREML);
        }
}
