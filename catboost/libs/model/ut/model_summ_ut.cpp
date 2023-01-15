#include <catboost/libs/model/ut/lib/model_test_helpers.h>

#include <catboost/private/libs/algo/apply.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/cpp/unittest/registar.h>

using namespace std;
using namespace NCB;

static void AssertModelSumEqualSliced(TDataProviderPtr dataProvider, bool changeScale = false, bool changeBias = false) {
    NJson::TJsonValue params;
    params.InsertValue("learning_rate", 0.01);
    params.InsertValue("iterations", 100);
    params.InsertValue("random_seed", 1);
    TFullModel bigModel;
    TEvalResult evalResult;

    TDataProviders dataProviders;
    dataProviders.Learn = dataProvider;
    dataProviders.Test.push_back(dataProvider);

    THolder<TLearnProgress> learnProgress;

    TrainModel(
        params,
        nullptr,
        Nothing(),
        Nothing(),
        dataProviders,
        Nothing(),
        &learnProgress,
        "",
        &bigModel,
        {&evalResult});

    if (changeScale) {
        bigModel.SetScaleAndBias({0.5, bigModel.GetScaleAndBias().Bias});
    }
    if (changeBias) {
        bigModel.SetScaleAndBias({bigModel.GetScaleAndBias().Scale, 0.125});
    }
    auto bigResult = ApplyModelMulti(bigModel, *(dataProvider->ObjectsData));
    bigModel.ModelTrees.GetMutable()->DropUnusedFeatures();
    TVector<TFullModel> partModels;
    TVector<const TFullModel*> modelPtrs;
    TVector<double> modelWeights(5, 1.0);
    for (size_t i = 0; i < 5; ++i) {
        auto partModel = bigModel.CopyTreeRange(i * 20, (i + 1) * 20);
        partModel.ModelTrees.GetMutable()->DropUnusedFeatures();
        partModels.emplace_back(partModel);
    }
    for (auto& model : partModels) {
        modelPtrs.push_back(&model);
    }
    auto mergedModel = SumModels(modelPtrs, modelWeights);
    if (!changeScale && !changeBias) {
        UNIT_ASSERT_EQUAL(*mergedModel.ModelTrees, *bigModel.ModelTrees);
    }

    auto mergedResult = ApplyModelMulti(mergedModel, *dataProvider->ObjectsData);
    UNIT_ASSERT_VALUES_EQUAL(bigResult.ysize(), mergedResult.ysize());
    for (int idx = 0; idx < bigResult.ysize(); ++idx) {
        if (!changeScale && !changeBias) {
            UNIT_ASSERT_VALUES_EQUAL(bigResult[idx], mergedResult[idx]);
        } else {
            for (int valueIdx = 0; valueIdx < bigResult[idx].ysize(); ++valueIdx) {
                UNIT_ASSERT_DOUBLES_EQUAL(bigResult[idx][valueIdx], mergedResult[idx][valueIdx], 1e-15);
            }
        }
    }
}

Y_UNIT_TEST_SUITE(TModelSummTests) {
    Y_UNIT_TEST(SimpleModelMerge) {
        const auto model1 = SimpleFloatModel();
        const auto model2 = SimpleFloatModel();
        const auto largeModel = SumModels({&model1, &model2}, {2.0, 1.0});
    }

    Y_UNIT_TEST(FloatModelMergeTest) {
        auto bigModel = TrainFloatCatboostModel(40);
        bigModel.ModelTrees.GetMutable()->DropUnusedFeatures();
        TVector<TFullModel> partModels;
        TVector<const TFullModel*> modelPtrs;
        TVector<double> modelWeights(5, 1.0);
        for (size_t i = 0; i < 5; ++i) {
            auto partModel = bigModel.CopyTreeRange(i * 8, (i + 1) * 8);
            partModel.ModelTrees.GetMutable()->DropUnusedFeatures();
            partModels.emplace_back(partModel);
        }
        for (auto& model : partModels) {
            modelPtrs.push_back(&model);
        }
        auto mergedModel = SumModels(modelPtrs, modelWeights);
        UNIT_ASSERT_EQUAL(*mergedModel.ModelTrees, *bigModel.ModelTrees);
    }

    Y_UNIT_TEST(SumEqualSliced) {
        AssertModelSumEqualSliced(GetAdultPool());
        AssertModelSumEqualSliced(GetMultiClassPool());
    }

    Y_UNIT_TEST(SumEqualSlicedTestWithScaleAndBias) {
        AssertModelSumEqualSliced(GetAdultPool(), true, false);
        AssertModelSumEqualSliced(GetAdultPool(), false, true);
    }
}
